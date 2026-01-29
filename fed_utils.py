import numpy as np
import torch
import torch.nn.functional as f
import copy
import os
from prettytable import PrettyTable
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

def save_acc_csv(para_dir,global_test_acc_dict,cfg):
    acc_path = os.path.join(para_dir, 'acc.csv')
    if os.path.exists(acc_path):
        with open(acc_path, 'a') as result_file:
            for key in global_test_acc_dict:
                method_result = global_test_acc_dict[key]
                result_file.write(str(key) + ',')
                for epoch in range(len(method_result)):
                    result_file.write(str(method_result[epoch]))
                    if epoch != len(method_result) - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
    else:
        with open(acc_path, 'w') as result_file:
            result_file.write('idx,')
            for epoch in range(cfg.OPTIM.ROUND):
                result_file.write('epoch_' + str(epoch))
                if epoch != cfg.OPTIM.ROUND - 1:
                    result_file.write(',')
                else:
                    result_file.write('\n')

            for key in global_test_acc_dict:
                method_result = global_test_acc_dict[key]
                result_file.write(str(key) + ',')
                for epoch in range(len(method_result)):
                    result_file.write(str(method_result[epoch]))
                    if epoch != len(method_result) - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')

def show_results(cfg, results, epoch,global_test_acc_dict):

    global_test_acc = []
    global_test_error = []
    global_test_f1 = []
    for k, result in enumerate(results):
        global_test_acc.append(results[k]['accuracy'])
        global_test_error.append(results[k]['error_rate'])
        global_test_f1.append(results[k]['macro_f1'])

        if k in global_test_acc_dict:
            global_test_acc_dict[k].append(results[k]['accuracy'])
        else:
            global_test_acc_dict[k] = [results[k]['accuracy']]

        print(k, "--Local test acc:", results[k]['accuracy'])

    print("--Global test acc:", sum(global_test_acc) / len(global_test_acc))

    print(f"Epoch:{epoch}")
    return global_test_acc,global_test_acc_dict

def average_weights(w, idxs_users, datanumber_client, islist=False):
    """
    Returns the average of the weights.
    """
    total_data_points = sum([datanumber_client[r] for r in idxs_users])

    w_avg = copy.deepcopy(w[idxs_users[0]])
    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points

        if islist:
            if idx == 0:
                w_avg = w_avg * fed_avg_freqs
            else:
                w_avg += w[idxs_users[idx]] * fed_avg_freqs
        else:
            if idx == 0:
                for key in w_avg:
                    w_avg[key] = w_avg[key] * fed_avg_freqs
            else:
                for key in w_avg:
                    w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs

    return w_avg


def calculate_client_similarity_matrix(w_list, idxs_users):
    """
    Calculates the pairwise cosine similarity matrix between clients' parameters.
    
    Args:
        w_list: List of parameters (tensors) for each client.
        idxs_users: List of client indices to consider.
        
    Returns:
        similarity_matrix: A numpy array of shape (len(idxs_users), len(idxs_users))
                           containing cosine similarities.
    """
    n_clients = len(idxs_users)
    similarity_matrix = np.zeros((n_clients, n_clients))
    
    # Pre-compute flattened vectors on CPU
    vectors = []
    for idx in idxs_users:
        vectors.append(w_list[idx].view(-1).cpu())
        
    for i in range(n_clients):
        for j in range(i, n_clients):
            vec_i = vectors[i]
            vec_j = vectors[j]
            
            sim = f.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0)).item()
            
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
            
    return similarity_matrix


def similarity_weighted_aggregation(w, global_w, idxs_users, datanumber_client, temperature=1.0, islist=False):
    """
    Aggregation based on Cosine Similarity between local weights and global weights.
    Helps mitigate client drift/overfitting by penalizing large deviations.
    """
    total_data_points = sum([datanumber_client[r] for r in idxs_users])

    # Calculate similarities
    similarities = []
    if islist:
        # w is a list of tensors
        # global_w is a single tensor
        # Ensure calculation is on CPU to avoid device mismatch
        global_vec = global_w.view(-1).cpu()

        for idx in idxs_users:
            local_vec = w[idx].view(-1).cpu()
            sim = f.cosine_similarity(local_vec.unsqueeze(0), global_vec.unsqueeze(0)).item()
            similarities.append(sim)
    else:
        # w is a list of state_dicts
        # Not implemented for full state_dict yet as we focus on specific params like sigma
        raise NotImplementedError("Similarity aggregation for full state_dict not implemented yet.")

    similarities = torch.tensor(similarities)
    #print(f"Raw similarities: {similarities}")
    # Softmax weighting based on similarity
    # Higher similarity -> Higher weight (trust clients that stayed close to global knowledge)
    # Lower similarity -> Lower weight (distrust clients that drifted/overfitted)
    weights = f.softmax(similarities / temperature, dim=0).numpy()
    #print(f"Softmax weights (temp={temperature}): {weights}")

    # Combine with data volume weights (optional, or just use similarity)
    # Here we blend them: alpha * data_weight + (1-alpha) * sim_weight
    # But for now, let's use the product and re-normalize

    final_weights = []
    data_weights_arr = np.array(
        [datanumber_client[idx] / total_data_points for idx in idxs_users],
        dtype=np.float64,
    )

    for i, idx in enumerate(idxs_users):
        data_weight = datanumber_client[idx] / total_data_points
        combined = data_weight * weights[i]
        final_weights.append(combined)

    #print(f"Data weights: {data_weights_arr.tolist()}")

    final_weights = np.array(final_weights, dtype=np.float64)
    final_weights = final_weights / final_weights.sum()
    #print(f"Final weights for users {idxs_users}: {final_weights}")

    max_diff = float(np.max(np.abs(final_weights - data_weights_arr)))
    #print(f"Max |final_weights - data_weights|: {max_diff:.8f}")

    # Aggregate
    if islist:
        # Move weights to CPU for aggregation to be safe, or keep on original device
        # We use w[idxs_users[0]] as base, so result will be on its device
        w_avg = copy.deepcopy(w[idxs_users[0]]) * final_weights[0]
        for i in range(1, len(idxs_users)):
            w_avg += w[idxs_users[i]] * final_weights[i]
    else:
         pass # Not needed for now

    return w_avg


def cluster_weights(w, datanumber):
    propmt_cluster = []
    for i in range(len(w)):
        prompt = w[i]['prompt_learner.ctx'].flatten(0).cpu()
        propmt_cluster.append(prompt.numpy())

    # cluster_matrix = linkage(propmt_cluster, 'average')
    # cluster_results = fcluster(cluster_matrix, 1, 'distance')
    cluster_model = AgglomerativeClustering(n_clusters=3, linkage="average", affinity="cosine")
    cluster_model = cluster_model.fit(propmt_cluster)
    cluster_results = cluster_model.labels_
    cluster_number = max(cluster_results) + 1
    cluster_group = [[] for i in range(cluster_number)]
    w_cluster = {cluster_i: None for cluster_i in range(cluster_number)}
    w_temp = copy.deepcopy(w[0])

    for idx in range(len(cluster_results)):
        cluster_group[cluster_results[idx]].append(idx)

    for num in range(cluster_number):
        client_list = cluster_group[num]
        total_data_points = sum([datanumber[r] for r in client_list])
        fed_avg_freqs = [datanumber[r] / total_data_points for r in client_list]
        for idx in range(len(client_list)):
            if idx == 0:
                prompt_avg = w[client_list[idx]]['prompt_learner.ctx'] * fed_avg_freqs[idx]
            else:
                prompt_avg += w[client_list[idx]]['prompt_learner.ctx'] * fed_avg_freqs[idx]
        w_temp['prompt_learner.ctx'] = prompt_avg
        w_cluster[num] = w_temp

    return w_cluster, cluster_group


def count_parameters(model, model_name):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if model_name in name:
            # if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params