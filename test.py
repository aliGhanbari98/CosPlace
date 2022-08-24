
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from math import sqrt


# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]

def measureDistance (x,y):
  return sqrt(((x[0]-y[0]) ** 2) + ((x[1] - y[1])** 2))


def in_list(item,L):
  for i in L:
      if item in i:
          return L.index(i)
  return -1


def re_ranking(utms, predictions, dist_threshold):
    result = []
    for i, pred_i in enumerate(utms):
        for j, pred_j in enumerate(utms):
            if i == j: continue
            if measureDistance(pred_i, pred_j) < dist_threshold:
                found_index = in_list(j, result)
                if found_index != -1:
                    if i not in result[found_index]:
                        result[found_index].append(i)
                    break
                else:
                    result.append([i, j])
                    break
            else:
                result.append([i])
                break
    sorted_result = sorted(result, key=lambda ele: len(ele), reverse=True)
    flatted_result = [item for sublist in sorted_result for item in sublist]
    new_predictions = []
    for pred_index in flatted_result:
        new_predictions.append(predictions[pred_index])
    return new_predictions


def test(args, eval_ds, model):
    """Compute descriptors of the given dataset and compute the recalls."""
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")
        for images, indices in tqdm(database_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
        
        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
    
    queries_descriptors = all_descriptors[eval_ds.database_num:]
    database_descriptors = all_descriptors[:eval_ds.database_num]
    
    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
    
    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))

    ### re-ranking is going to be added here

    predictions_utms = []
    for pred_index, pred in enumerate(predictions):
        predictions_utms.append(eval_ds.database_utms[pred_index])

    reranked_predictions = re_ranking(predictions_utms, predictions, args.reranking_minimum_distance)

    print('original : ', predictions[:100])
    print('re-ranked : ', reranked_predictions[:100])


    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(reranked_predictions):
        print(f"UTMS{query_index}", eval_ds.database_utms[query_index])
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
    return recalls, recalls_str

