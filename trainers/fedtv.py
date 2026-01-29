import os.path as osp

import torch
import torch.nn as nn
from pandas.core.nanops import bottleneck_switch
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

# from Dassl.dassl.engine import TRAINER_REGISTRY, TrainerX
from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
from Dassl.dassl.metrics import compute_accuracy
from torch.cuda import device
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from collections import OrderedDict

from sklearn.decomposition import PCA
import numpy as np

_tokenizer = _Tokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    design_details = {"trainer": 'FedTV',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.FEDTV.N_CTX
        ctx_init = cfg.TRAINER.FEDTV.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        bottleneck = cfg.TRAINER.FEDTV.BOTTLENECK
        self.N = cfg.TRAINER.FEDTV.N
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.FEDTV.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")

                sigma = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)

                ctx_local = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)

                nn.init.normal_(sigma, std=0.02)
                nn.init.normal_(ctx_local, std=0.02)

                # U: (N, n_ctx, n_ctx), S: (N, min), Vh: (N, ctx_dim, ctx_dim)
                U, S, Vh = torch.linalg.svd(ctx_local.float(), full_matrices=False)

                U_low = U[:, :, :bottleneck].to(dtype)
                S_low = S[:, :bottleneck].to(dtype)
                V_low = Vh[:, :bottleneck, :].to(dtype)
                for i in range(self.N):
                    nn.init.orthogonal_(U_low[i])
                    nn.init.orthogonal_(V_low[i])
                nn.init.normal_(S_low,std=0.02)


            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")


        prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.B = nn.Parameter(U_low)
        self.S = nn.Parameter(S_low)
        self.A = nn.Parameter(V_low)
        self.sigma = nn.Parameter(sigma)
        #self.ctx_local = nn.Parameter(ctx_local)


        print("classnames:", classnames)
        classnames = [name.replace("_", " ") if isinstance(name, str) else name for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) if isinstance(name, str) else 0 for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1)
        # tokenized_prompts3.view(3,100,77)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("embedding", embedding)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.FEDPGP.CLASS_TOKEN_POSITION

    def forward(self):
        B = self.B
        S = self.S
        A = self.A

        # U: (N, n_ctx, r), S: (N, r), V: (N, r, ctx_dim)
        SV = S.unsqueeze(2) * A
        BA = torch.matmul(B, SV)

        sigma = self.sigma
        # full rank
        #ctx = ctx_local+ sigma
        ctx = BA + self.sigma# low rank adaptation
        embedding = self.embedding

        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1)

        ctx = ctx.permute(1, 0, 2, 3)
        ctx = ctx.contiguous().view(self.N * self.n_cls, self.n_ctx, ctx.shape[3])

        if BA.dim() == 3:
            BA = BA.unsqueeze(0).expand(self.n_cls, -1, -1,-1)

        BA = BA.permute(1, 0, 2, 3)
        BA = BA.contiguous().view(self.N * self.n_cls, self.n_ctx, BA.shape[3])

        if sigma.dim() == 3:
            sigma = sigma.unsqueeze(0).expand(self.n_cls, -1, -1,-1)

        sigma = sigma.permute(1, 0, 2, 3)
        sigma = sigma.contiguous().view(self.N * self.n_cls, self.n_ctx, sigma.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":

            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_sigma = torch.cat(
                [
                    prefix,  # (n_cls, -1, dim)
                    sigma,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_BA = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    BA,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            #prompts_local_ctx= torch.cat([prefix,ctx_local,suffix],dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return embedding, prompts_sigma, prompts_BA, prompts

class Adapter(nn.Module):
    def __init__(self, clip_model):
        super(Adapter, self).__init__()

        feature_dim = clip_model.ln_final.weight.shape[0]  # 512

        self.layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim,bias=False),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim,bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.layer(x)
        return output


class VisualDynamicFusion(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()

        self.w_query = nn.Linear(feature_dim, hidden_dim)
        self.w_key = nn.Linear(feature_dim, hidden_dim)

    def forward(self, x_orig, x_glo, x_loc):

        Q = self.w_query(x_orig) # [Batch, Hidden]
        
        K_orig = self.w_key(x_orig) # [Batch, Hidden]
        K_glo = self.w_key(x_glo)   # [Batch, Hidden]
        K_loc = self.w_key(x_loc)   # [Batch, Hidden]
        

        score_orig = (Q * K_orig).sum(dim=1, keepdim=True)
        score_glo = (Q * K_glo).sum(dim=1, keepdim=True)
        score_loc = (Q * K_loc).sum(dim=1, keepdim=True)

        scores = torch.cat([score_orig, score_glo, score_loc], dim=1)
        weights = F.softmax(scores, dim=1)
        
        w_orig = weights[:, 0].unsqueeze(1)
        w_glo = weights[:, 1].unsqueeze(1)
        w_loc = weights[:, 2].unsqueeze(1)

        final_features = w_orig * x_orig + w_glo * x_glo + w_loc * x_loc
        
        return final_features

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.glo_adapter = Adapter(clip_model).to(clip_model.dtype)
        self.loc_adapter = Adapter(clip_model).to(clip_model.dtype)
        

        feature_dim = clip_model.ln_final.weight.shape[0]

        self.visual_fusion = VisualDynamicFusion(feature_dim).to(clip_model.dtype)


    def forward(self, image):
        embedding, prompts_sigma, prompts_BA, prompts = self.prompt_learner()

        image_features_0 = self.image_encoder(image.type(self.dtype))

        image_features_att_glo = self.glo_adapter(image_features_0)
        image_features_att_loc = self.loc_adapter(image_features_0)

        image_features_glo = torch.mul(image_features_att_glo, image_features_0)
        image_features_loc = torch.mul(image_features_att_loc, image_features_0)

        image_features= self.visual_fusion(image_features_0, image_features_glo, image_features_loc)


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)


        tokenized_prompts = self.tokenized_prompts

        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if self.training == True:
            text_features_0 = self.text_encoder(embedding, tokenized_prompts)
            text_features_sigma = self.text_encoder(prompts_sigma, tokenized_prompts)

            text_features_0 = text_features_0 / text_features_0.norm(dim=-1, keepdim=True)
            text_features_sigma = text_features_sigma / text_features_sigma.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features@text_features.t()
            return text_features_0, text_features_sigma, text_features,logits

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features@text_features.t()

        return logits


# @TRAINER_REGISTRY.register()
class FedPGP(TrainerX):
    """
    It is based on CoOp.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.FEDTV.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):

        cfg = self.cfg
        self.mu = cfg.TRAINER.FEDTV.mu

        self.temp = cfg.TRAINER.FEDTV.temp
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.FEDPGP.PREC == "fp32" or cfg.TRAINER.FEDPGP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "glo_adapter" not in name and "loc_adapter" not in name and "visual_fusion" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)


        self.model.to(self.device)


        self.optim = build_optimizer(self.model, cfg.OPTIM)

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model("FedTV", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.FEDPGP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        cos = torch.nn.CosineSimilarity(dim=-1)
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.FEDTV.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)

            self.scaler.update()
        else:
            text_features_0, text_features_sigma, text_features, output= self.model(image)

            cls_loss = F.cross_entropy(output, label)


            posi = cos(text_features_0, text_features_sigma)
            nega = cos(text_features_0, text_features)

            logits = torch.cat((posi.reshape(-1, 1), nega.reshape(-1, 1)), dim=1)
            logits /= self.temp
            target = torch.zeros(logits.size(0)).to(self.device).long()
            contrastive_loss = F.cross_entropy(logits, target)


            ortho_reg = 0.0
            ortho_weight = 0.1
            U = self.model.prompt_learner.B
            V = self.model.prompt_learner.A

            for i in range(U.size(0)):
                U_i = U[i]
                gram_U = torch.matmul(U_i.transpose(0, 1), U_i)
                identity_r = torch.eye(gram_U.size(0), device=gram_U.device)
                ortho_reg += torch.norm(gram_U - identity_r, p='fro')

                V_i = V[i]
                gram_V = torch.matmul(V_i, V_i.transpose(0, 1))
                ortho_reg += torch.norm(gram_V - identity_r, p='fro')

            # 使用自适应权重
            reg_loss =ortho_weight * ortho_reg
            loss = cls_loss + reg_loss+contrastive_loss

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            "cls_loss": cls_loss.item() if prec != "amp" else 0,
            "cont_loss": contrastive_loss.item() if prec != "amp" else 0,
            "reg_loss": reg_loss.item() if prec != "amp" else 0,
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)