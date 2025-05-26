import json
import numpy as np
import torch


def pad_tok_len(quote_embeddings, pad_value=0):
    lengths = [e.shape[0] for e in quote_embeddings]
    max_len = max(lengths)
    N, H = len(quote_embeddings), quote_embeddings[0].shape[1]
    padded_embeddings = np.full((N, max_len, H), pad_value, dtype=quote_embeddings[0].dtype)
    padded_masks = np.zeros((N, max_len), dtype=np.int64)
    for i, (emb, length) in enumerate(zip(quote_embeddings, lengths)):
        padded_embeddings[i, :length, :] = emb
        padded_masks[i, :length] = 1
    return padded_embeddings, padded_masks


def colbert_score_torch(query_embed, quote_embeddings, quote_masks, device='cuda'):
    # Convert to tensors if necessary and move to device
    if not torch.is_tensor(query_embed):
        query_embed = torch.from_numpy(query_embed)
    if not torch.is_tensor(quote_embeddings):
        quote_embeddings = torch.from_numpy(quote_embeddings)
    if not torch.is_tensor(quote_masks):
        quote_masks = torch.from_numpy(quote_masks)

    # Convert to float32, and move to device
    query_embed = query_embed.to(device=device, dtype=torch.float32)
    quote_embeddings = quote_embeddings.to(device=device, dtype=torch.float32)
    quote_masks = quote_masks.to(device=device)  # mask can remain as int/bool

    # [Q, H] @ [N, L, H].transpose(-1, -2) => [Q, N, L]
    # Efficient batched matrix multiplication via einsum
    sim = torch.einsum('qh,nlh->qnl', query_embed, quote_embeddings)  # [Q, N, L]
    # Mask padded tokens so they are not considered for max
    sim = sim.masked_fill(quote_masks.unsqueeze(0) == 0, -1e9)  # [Q, N, L]
    # MaxSim: max over L (quote token dimension)
    maxsim = sim.max(dim=2).values  # [Q, N]
    # Sum over query tokens
    scores = maxsim.sum(dim=0)  # [N]
    return scores


def colbert_score(query_embed, quote_embeddings, quote_masks, use_gpu=False):
    if use_gpu:
        return colbert_score_torch(query_embed, quote_embeddings, quote_masks)

    Q, H = query_embed.shape  # [Q, H]
    N, L, _ = quote_embeddings.shape  # [N, L, H]
    # 1. Compute [Q, N, L] (similarity btw every query token to every quote token)
    # Expand query to [Q, 1, 1, H], quote_embeddings to [1, N, L, H]
    query_expanded = query_embed[:, np.newaxis, np.newaxis, :]  # [Q, 1, 1, H]
    quote_expanded = quote_embeddings[np.newaxis, :, :, :]  # [1, N, L, H]
    sim = np.matmul(query_expanded, np.transpose(quote_expanded, (0, 1, 3, 2)))  # (Q, N, 1, L)
    # But let's use broadcasting for dot product:
    # sim[q, n, l] = np.dot(query_embed[q], quote_embeddings[n,l])
    sim = np.einsum('qh,nlh->qnl', query_embed, quote_embeddings)  # [Q, N, L]
    # 2. Mask invalid tokens
    sim = np.where(quote_masks[np.newaxis, :, :] == 1, sim, -1e9)  # [Q, N, L]
    # 3. MaxSim: For each query token, take max over quote tokens (L dimension)
    maxsim = sim.max(-1)  # [Q, N]
    # 4. Aggregate (sum over query tokens)
    scores = maxsim.sum(axis=0)  # [N]
    return scores


def load_jsonl(filename, debug_mode=False):
    data_list = []
    with open(filename, 'r', encoding="utf-8") as file:
        for line in file:
            if not debug_mode:
                data_list.append(json.loads(line.strip()))
            else:
                try:
                    data_list.append(json.loads(line.strip()))
                except:
                    print(line.strip())
    return data_list


def precision(retrieved, ground_truth):
    true_positives = len(set(retrieved) & set(ground_truth))
    return true_positives / len(retrieved) if len(retrieved) > 0 else 0


def recall(retrieved, ground_truth):
    true_positives = len(set(retrieved) & set(ground_truth))
    return true_positives / len(ground_truth) if ground_truth else 0


def ndcg(retrieved, ground_truth):
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(retrieved), len(ground_truth))))
    dcg = sum(1 / np.log2(i + 2) for i in range(len(retrieved)) if retrieved[i] in ground_truth)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0


def average_precision(retrieved, ground_truth):
    true_positives = 0
    avg_prec = 0.0

    for i, item in enumerate(retrieved):
        if item in ground_truth:
            true_positives += 1
            avg_prec += true_positives / (i + 1)

    return avg_prec / true_positives if true_positives else 0


def mean_reciprocal_rank(retrieved, ground_truth):
    for i, item in enumerate(retrieved):
        if item in ground_truth:
            return 1 / (i + 1)
    return 0


def top_k_indices(scores, k):
    # raise ValueError("k cannot be greater than the number of scores")
    # Create a list of indices and scores, sort by scores in descending order
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    if k <= len(scores):
        # Extract the indices of the top k scores
        top_indices = [index for index, score in indexed_scores[:k]]
        return top_indices
    else:
        return [index for index, score in indexed_scores]


def calculate_overlap_score(bbox1, bbox2):
    # Extract coordinates of each bounding box
    top1, left1, bottom1, right1 = bbox1
    top2, left2, bottom2, right2 = bbox2

    # Calculate the intersection rectangle
    inter_top = max(top1, top2)
    inter_left = max(left1, left2)
    inter_bottom = min(bottom1, bottom2)
    inter_right = min(right1, right2)

    # Check if there is an intersection
    if inter_top < inter_bottom and inter_left < inter_right:
        intersection_area = (inter_bottom - inter_top) * (inter_right - inter_left)  # Compute the intersection area
    else:
        intersection_area = 0.0  # No intersection
    return intersection_area


def recall_layout(layout_topk, layout_indices, layout_mapping):
    # for each retrieved layout, iterate the answers in layout mapping and get the recall.
    # sum the recall for each retrieved layout.
    recall_area = 0
    for layout_id in layout_topk:
        _, page_id, x1, y1, x2, y2 = layout_indices[layout_id]
        for gt_layout in layout_mapping:
            # if page matches, calculate bbox overlap, else we won't even consider.
            if (page_id == gt_layout["page"]):
                recall_area += calculate_overlap_score([x1, y1, x2, y2], gt_layout["bbox"])

    gt_area = 0
    # Calculate the area of the second bounding box
    for gt_layout in layout_mapping:
        top, left, bottom, right = gt_layout["bbox"]
        gt_area += (bottom - top) * (right - left)

    if gt_area == 0: return 0.0  # Avoid division by zero
    return recall_area / gt_area  # normalized overlap score


def evaluate_layout(data_json, model_name="", topk=1, metric=""):
    total_count = 0
    total_score = {"precision": 0, "recall": 0, "ndcg": 0, "map": 0, "mrr": 0}
    domain_list = ["Research report / Introduction", "Administration/Industry file",
                   "Tutorial/Workshop", "Academic paper", "Brochure",
                   "Financial report", "Guidebook", "Government", "Laws", "News"]
    total_score_by_domain = {item: total_score.copy() for item in domain_list}
    total_count_by_domain = {item: 0 for item in domain_list}

    for i, qa in enumerate(data_json):
        domain = qa["domain"]
        page_id = qa["page_id"]
        layout_topk = top_k_indices(qa["scores_layout"], topk)
        qa_recall = recall_layout(layout_topk, qa["layout_indices"], qa["layout_mapping"])
        total_score["recall"] += qa_recall
        total_count += 1
        total_score_by_domain[domain]["recall"] += qa_recall
        total_count_by_domain[domain] += 1

    results_list = []
    scores_list = []
    for domain, total_score_domain in total_score_by_domain.items():
        try:
            average_score = total_score_domain[metric] / total_count_by_domain[domain]
        except ZeroDivisionError:
            average_score = 0
        results_list.append((domain, average_score))
        scores_list.append(average_score)

    scores_list.append(sum(scores_list) / len(scores_list))
    scores_list.append(total_score[metric] / total_count)
    scores_list = [str(round(x * 100, 1)) for x in scores_list]
    out_str = " & ".join(scores_list)
    result = f"K={topk}, METRIC={metric}, Retriever={model_name}, Results= {out_str}"
    print(result)
    return result


def evaluate_page(data_json, model_name="", topk="", metric=""):
    total_count = 0
    total_score = {"precision": 0, "recall": 0, "ndcg": 0, "map": 0, "mrr": 0}
    domain_list = ["Research report / Introduction", "Administration/Industry file",
                   "Tutorial/Workshop", "Academic paper", "Brochure",
                   "Financial report", "Guidebook", "Government", "Laws", "News"]
    total_score_by_domain = {item: total_score.copy() for item in domain_list}
    total_count_by_domain = {item: 0 for item in domain_list}

    for i, qa in enumerate(data_json):
        domain = qa["domain"]
        page_id = qa["page_id"]

        scores = top_k_indices(qa["scores_page"], topk)

        total_score["precision"] += precision(scores, page_id)
        total_score["recall"] += recall(scores, page_id)
        total_score["ndcg"] += ndcg(scores, page_id)
        total_score["map"] += average_precision(scores, page_id)
        total_score["mrr"] += mean_reciprocal_rank(scores, page_id)
        total_count += 1

        total_score_by_domain[domain]["precision"] += precision(scores, page_id)
        total_score_by_domain[domain]["recall"] += recall(scores, page_id)
        total_score_by_domain[domain]["ndcg"] += ndcg(scores, page_id)
        total_score_by_domain[domain]["map"] += average_precision(scores, page_id)
        total_score_by_domain[domain]["mrr"] += mean_reciprocal_rank(scores, page_id)
        total_count_by_domain[domain] += 1

    results_list = []
    scores_list = []
    for domain, total_score_domain in total_score_by_domain.items():
        try:
            average_score = total_score_domain[metric] / total_count_by_domain[domain]
        except ZeroDivisionError:
            average_score = 0
        results_list.append((domain, average_score))
        scores_list.append(average_score)

    scores_list.append(sum(scores_list) / len(scores_list))
    scores_list.append(total_score[metric] / total_count)
    scores_list = [str(round(x * 100, 1)) for x in scores_list]
    out_str = " & ".join(scores_list)
    result = f"K={topk}, METRIC={metric}, Retriever={model_name}, Results= {out_str}"
    print(result)
    return result
