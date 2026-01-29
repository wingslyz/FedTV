import argparse
import torch
from sklearn.manifold import TSNE

from Dassl.dassl.utils import setup_logger, set_random_seed, collect_env_info
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer
import time

import os
import gc
import copy
from prettytable import PrettyTable
import numpy as np
from fed_utils import average_weights, similarity_weighted_aggregation, cluster_weights, count_parameters, calculate_client_similarity_matrix

import os




def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg, args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    # Config for FEDPGP
    cfg.TRAINER.FEDTV = CN()
    cfg.TRAINER.FEDTV.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.FEDTV.CSC = False  # class-specific context
    cfg.TRAINER.FEDTV.CTX_INIT = args.ctx_init  # initialization words
    cfg.TRAINER.FEDTV.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.FEDTV.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.FEDTV.BOTTLENECK = args.bottleneck
    cfg.TRAINER.FEDTV.N = args.num_prompt # number of prompts
    cfg.TRAINER.FEDTV.FEATURE = args.feature
    cfg.TRAINER.FEDTV.mu = args.mu
    cfg.TRAINER.FEDTV.temp = args.temp
    cfg.RANK = args.rank

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp32"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = args.subsample  # all, base or new
    cfg.DATASET.USERS = args.num_users  # number of clients
    cfg.DATASET.IID = args.iid  # is iid
    cfg.DATASET.PARTITION = args.partition
    cfg.DATASET.USEALL = args.useall # use all data for training instead of few shot
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.DATASET.BETA = args.beta
    cfg.DATASET.REPEATRATE = 0.0 # repeat rate on each client
    cfg.DATALOADER.TRAIN_X.N_DOMAIN = args.num_domain # number of domain
    cfg.DATASET.IMBALANCE_TRAIN = args.imbalance_train # is adding label skew to feature skew datasets
    cfg.DATASET.SPLIT_CLIENT = args.split_client # is adding label skew to feature skew datasets and split one domain to multi clients
    cfg.OPTIM.ROUND = args.round # global round
    cfg.OPTIM.MAX_EPOCH = args.local_epoch # local epoch
    cfg.OPTIM.GAMMA = args.gamma # gamma of single-step
    cfg.OPTIM.LR = args.lr #learning rate
    cfg.DATASET.TARGET_DOMAIN = args.target_domain
    

    cfg.MODEL.BACKBONE.PRETRAINED = True


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        # print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    if cfg.DATASET.USEALL == True:
        setup_logger(os.path.join(cfg.OUTPUT_DIR,cfg.DATASET.SUBSAMPLE_CLASSES))
    else:
        setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    results = []
    client_acc = [[] for i in range(args.num_users)]

    if args.eval_only:
        # Build trainer but skip training loader construction
        local_trainer = build_trainer(cfg)

        print(f"Loading model from: {args.model_dir}")
        model_path = os.path.join(args.model_dir, "save.pt")  # or "fedadapter_best.pt"
        checkpoint = torch.load(model_path, map_location="cpu",weights_only=True)

        for idx in range(cfg.DATASET.USERS):
            local_trainer.model.load_state_dict(checkpoint[idx], strict=False)
            print(f"Evaluating client {idx}")
            result = local_trainer.test(idx=idx)
            client_acc[idx].append(result[0])
            results.append(result)
        global_test_acc = [r[0] for r in results]
        print(f"Global Test Accuracy: {np.mean(global_test_acc):.2f}")

        return

    print_args(args, cfg)
    local_weights= [[] for i in range(args.num_users)]
    local_weights_0= [[] for i in range(args.num_users)]
    local_weights_1= [[] for i in range(args.num_users)]
    local_weights_2 = [[] for i in range(args.num_users)]
    local_weights_3 = [[] for i in range(args.num_users)]
    local_weights_4 = [[]for i in range(args.num_users)]
    local_weights_5 = [[]for i in range(args.num_users)]
    local_weights_6 = [[] for i in range(args.num_users)]
    local_weights_7 = [[] for i in range(args.num_users)]
    local_weights_8 = [[] for i in range(args.num_users)]
    local_weights_9 = [[] for i in range(args.num_users)]
    local_weights_10 = [[] for i in range(args.num_users)]
    local_weights_per = [{} for i in range(args.num_users)]

    local_weights_save = [{} for i in range(args.num_users)]


    client_acc = [[] for i in range(args.num_users)]

    local_trainer = build_trainer(cfg)
    local_trainer.fed_before_train()
    count_parameters(local_trainer.model,"prompt_learner")
    count_parameters(local_trainer.model, "image_encoder")
    count_parameters(local_trainer.model, "text_encoder")

    # local_trainers = {net_i: None for net_i in range(cfg.DATASET.USERS)}
    datanumber_client = []
    if args.trainer == 'CLIP':
        global_weights = copy.deepcopy(local_trainer.model.state_dict())
    else:
        for net_i in range(cfg.DATASET.USERS):
            # local_trainer = build_trainer(cfg)
            datanumber_client.append(len(local_trainer.fed_train_loader_x_dict[net_i].dataset))
            # local_trainer.fed_before_train()
            # local_trainers[net_i] = local_trainer
            # local_weights[net_i] = copy.deepcopy(local_trainer.model.state_dict())
        global_weights = copy.deepcopy(local_trainer.model.state_dict())

    # Training
    start_epoch = 0
    max_epoch = cfg.OPTIM.ROUND
    # global_trainer.before_train()
    global_test_acc_list = []
    global_test_error_list = []
    global_test_f1_list = []
    global_epoch_list = []
    global_time_list = []
    cluster_group = []
    start = time.time()
    n_cls = len(local_trainer.dm.dataset.classnames)
    prompts_list = [2*torch.rand(n_cls,77,512)-1 for i in range(cfg.DATASET.USERS)]

    prev_sigma = None
    prev_glo_adapter_w0 = None
    prev_glo_adapter_w2 = None

    for epoch in range(start_epoch, max_epoch):

        if args.trainer == 'CLIP':
            print("------------local test start-------------")
            results = []
            idxs_users = list(range(0,cfg.DATASET.USERS))
            # idxs_users.pop(0)
            m = max(int(args.frac * args.num_users), 1)
            # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for idx in idxs_users:
                local_trainer.model.load_state_dict(global_weights)
                results.append(local_trainer.test(idx=idx))
            global_test_acc = []
            global_test_error = []
            global_test_f1 = []
            for k in range(len(results)):
                global_test_acc.append(results[k][0])
                global_test_error.append(results[k][1])
                global_test_f1.append(results[k][2])
            global_time_list.append(time.time() - start)
            global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
            global_test_error_list.append(sum(global_test_error) / len(global_test_error))
            global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
            global_epoch_list.append(epoch)
            print("Global test acc:", sum(global_test_acc)/len(global_test_acc))
            print("Global test error:", sum(global_test_error) / len(global_test_error))
            print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
            print("------------local test finish-------------")
            print("Epoch on server :", epoch)
            break

        elif args.model == "fedavg":
            m = max(int(args.frac * args.num_users), 1)
            # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            idxs_users = list(range(0,cfg.DATASET.USERS))
            idxs_users.pop(5)
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                if epoch == 0:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                else:
                    local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                local_weight = local_trainer.model.state_dict()
                local_weights_0[idx] = copy.deepcopy(local_weight['prompt_learner.ctx'])
                local_weights_per[idx]['prompt_learner.ctx'] = local_weights_0[idx]
            print("------------local train finish epoch:", epoch, "-------------")

            global_weights = average_weights(local_weights_0,idxs_users, datanumber_client, islist=True)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0,cfg.DATASET.USERS))
            for idx in idxs_users:
                local_weights_per[idx]['prompt_learner.ctx'] = global_weights
                local_trainer.model.load_state_dict(local_weights_per[idx],strict=False)
                # results.append(local_trainer.test(idx=idx))
                acc_idx = local_trainer.test(idx=idx)
                client_acc[idx].append(acc_idx[0])
                results.append(acc_idx)
            global_test_acc = []
            global_test_error = []
            global_test_f1 = []
            for k in range(len(results)):
                global_test_acc.append(results[k][0])
                global_test_error.append(results[k][1])
                global_test_f1.append(results[k][2])
            global_time_list.append(time.time() - start)
            global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
            global_test_error_list.append(sum(global_test_error) / len(global_test_error))
            global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
            global_epoch_list.append(epoch)
            print("Global test acc:", sum(global_test_acc)/len(global_test_acc))
            print("Global test error:", sum(global_test_error) / len(global_test_error))
            print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
            print("------------local test finish-------------")
            for i in idxs_users:
                print('client', i, 'local acc', client_acc[i])
                print('client', i, 'max acc', max(client_acc[i]))
            print("Epoch on server :", epoch)
            if sum(global_test_acc) / len(global_test_acc) >= max(global_test_acc_list):
                torch.save(local_weights_per, args.output_dir + "/save.pt")


        elif args.model == "FedTV":
            if epoch == 0:
                idxs_users = list(range(0, cfg.DATASET.USERS))
            else:
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                if epoch == 0:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                else:
                    local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                local_weight = local_trainer.model.state_dict()
                # 保存本地适应器参数 - 修改为QR分解参数
                local_weights_0[idx] = copy.deepcopy(local_weight['prompt_learner.sigma'])
                local_weights_1[idx] = copy.deepcopy(local_weight['prompt_learner.B'])  # 原来是U，现在是B
                local_weights_2[idx] = copy.deepcopy(local_weight['prompt_learner.A'])
                local_weights_3[idx] = copy.deepcopy(local_weight['glo_adapter.layer.0.weight'])
                local_weights_4[idx] = copy.deepcopy(local_weight['glo_adapter.layer.2.weight'])
                local_weights_5[idx] = copy.deepcopy(local_weight['loc_adapter.layer.0.weight'])
                local_weights_6[idx] = copy.deepcopy(local_weight['loc_adapter.layer.2.weight'])
                local_weights_7[idx] = copy.deepcopy(local_weight['visual_fusion.w_query.weight'])
                local_weights_8[idx] = copy.deepcopy(local_weight['visual_fusion.w_key.weight'])
                local_weights_9[idx] = copy.deepcopy(local_weight['prompt_learner.S'])
                #local_weights_10[idx] = copy.deepcopy(local_weight['prompt_learner.ctx_local'])

            print("------------local train finish epoch:", epoch, "-------------")

            # Calculate and print pairwise similarity of global prompts between clients
            # Sort indices to ensure the matrix corresponds to client IDs in ascending order (e.g., 0, 1, 2...)
            sorted_idxs = sorted(idxs_users)
            sim_matrix = calculate_client_similarity_matrix(local_weights_0, sorted_idxs)
            print(f"Epoch {epoch} Client Similarity Matrix (Global Prompt) [Clients: {sorted_idxs}]:")
            print(sim_matrix)

            if epoch == 0 or prev_sigma is None:
              sigma = average_weights(local_weights_0, idxs_users, datanumber_client, islist=True)
              adapter_weight_0 = average_weights(local_weights_3, idxs_users, datanumber_client, islist=True)
              adapter_weight_1 = average_weights(local_weights_4, idxs_users, datanumber_client, islist=True)
            else:
                agg_temp = 1.0
                sigma = similarity_weighted_aggregation(
                    local_weights_0,
                    prev_sigma,
                    idxs_users,
                    datanumber_client,
                    temperature=agg_temp,
                    islist=True,
                )

                adapter_weight_0 = similarity_weighted_aggregation(
                    local_weights_3,
                    prev_glo_adapter_w0,
                    idxs_users,
                    datanumber_client,
                    temperature=agg_temp,
                    islist=True,
                )
                adapter_weight_1 = similarity_weighted_aggregation(
                    local_weights_4,
                    prev_glo_adapter_w2,
                    idxs_users,
                    datanumber_client,
                    temperature=agg_temp,
                    islist=True,
                )

            prev_sigma = sigma.detach().clone()
            prev_glo_adapter_w0 = adapter_weight_0.detach().clone()
            prev_glo_adapter_w2 = adapter_weight_1.detach().clone()

            print("------------local test start-------------")
            results = []
            all_users = list(range(0, cfg.DATASET.USERS))

            for idx in all_users:
                local_weights_per[idx]['prompt_learner.sigma'] = sigma
                local_weights_per[idx]['prompt_learner.B'] =local_weights_1[idx]
                local_weights_per[idx]['prompt_learner.A'] = local_weights_2[idx]
                local_weights_per[idx]['glo_adapter.layer.0.weight'] = adapter_weight_0
                local_weights_per[idx]['glo_adapter.layer.2.weight'] = adapter_weight_1
                local_weights_per[idx]['loc_adapter.layer.0.weight'] = local_weights_5[idx]
                local_weights_per[idx]['loc_adapter.layer.2.weight'] = local_weights_6[idx]
                local_weights_per[idx]['visual_fusion.w_query.weight'] = local_weights_7[idx]
                local_weights_per[idx]['visual_fusion.w_key.weight'] = local_weights_8[idx]
                local_weights_per[idx]['prompt_learner.S'] = local_weights_9[idx]
                #local_weights_per[idx]['prompt_learner.ctx_local'] = local_weights_10[idx]

                local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                results.append(local_trainer.test(idx=idx))
            global_test_acc = []
            global_test_error = []
            global_test_f1 = []
            for k in range(len(results)):
                global_test_acc.append(results[k][0])
                global_test_error.append(results[k][1])
                global_test_f1.append(results[k][2])
            global_time_list.append(time.time() - start)
            global_test_acc_list.append(sum(global_test_acc) / len(global_test_acc))
            global_test_error_list.append(sum(global_test_error) / len(global_test_error))
            global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
            global_epoch_list.append(epoch)
            print("Global test acc:", sum(global_test_acc) / len(global_test_acc))
            print("Global test error:", sum(global_test_error) / len(global_test_error))
            print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))

            print("------------local test finish-------------")
            print("Epoch on server :", epoch)
            if sum(global_test_acc) / len(global_test_acc) >= max(global_test_acc_list):
                torch.save(local_weights_per, args.output_dir + "/save.pt")

    for idx in idxs_users:
        local_trainer.fed_after_train()
    # global_trainer.fed_after_train()
    print("global_test_acc_list:",global_test_acc_list)
    print("maximum test acc:", max(global_test_acc_list))
    print("mean of acc:",np.mean(global_test_acc_list[-5:]))
    print("std of acc:",np.std(global_test_acc_list[-5:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="FedTV", help="model of aggregation, choose from:FedTV")
    parser.add_argument("--trainer", type=str, default="FedTV", help="name of trainer, choose from: CLIP, FecTV")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--n_ctx', type=int, default=16, help="number of text encoder of text prompts")
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--gamma', type=float, default=1, help='gamma of single_step')
    parser.add_argument('--iid', default=False, help="is iid")
    parser.add_argument('--subsample', type=str, default='base', help="all,base,new")
    parser.add_argument('--feature', default=False, help="is compute similarity between text feature and image feature map")
    parser.add_argument('--round', type=int, default=10,help="number of communication round")
    parser.add_argument('--partition', type=str, default='noniid-labeldir100',
                        help='the data partitioning strategy of cifar10 and cifar100, select from "homo, noniid-labeluni, noniid-labeldir,noniid-labeldir100"')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--mu', type=float, default=1, help='The parameter for fedprox')
    parser.add_argument('--temp', type=float, default=0.5, help='The tempuature')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--num_prompt', type=int, default=1, help="number of prompts")
    parser.add_argument('--avg_prompt', type=int, default=1, help="number of prompts to average")
    parser.add_argument('--thresh', type=float, default=1e-3, help='the thresh of sinkhorn distance')
    parser.add_argument('--eps', type=float, default=0.1, help='the lambada of sinkhorn distance')
    parser.add_argument('--top_percent', type=float, default=1, help='the top_percent of COT')
    parser.add_argument('--max_iter', type=int, default=100, help="max iteration of COT")
    parser.add_argument('--num_domain', type=int, default=4, help="number of domain")
    parser.add_argument('--ctx_init', default=False, help="is using the ctx init")

    parser.add_argument('--alpha', type=float, default=0.8, help="push loss")
    parser.add_argument('--num_shots', type=int, default=16, help="number of shots in few shot setting")
    parser.add_argument('--bottleneck', type=int, default=8,help="number of middle in reparameter")
    parser.add_argument('--local_epoch', type=int, default=2, help="number of local epoch")
    parser.add_argument('--useall', default=False, help="is useall, True for all training samples, False for few shot learning")

    parser.add_argument('--train_batch_size', type=int, default=32, help="number of trainer batch size")
    parser.add_argument('--test_batch_size', type=int, default=100, help="number of test batch size")
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--seed", type=int, default=7, help="only positive value enables a fixed seed")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default ="trainers/vit_b16.yaml", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/dtd.yaml", help="path to config file for dataset setup")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="autodl-tmp/FedPGP/outputtest/", help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")
    

    args = parser.parse_args()
    main(args)