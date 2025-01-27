import math

def get_topk_results(predictions, scores, targets, k, all_items=None):
    # predictions: List[str] size = B(batch size)*k
    # scores: List[float] size = B*k
    # targets: List[str] size = B
    # k: int
    # all_items: List[str] or None
    results = []
    B = len(targets)
    # print(predictions[0].split("Response:")[0])
    predictions = [_.split(" POI index ")[-1].split(".")[0] for _ in predictions] # 取出最后一个Response:后的字符串，即预测的item
    predictions = [_.strip().replace(" ","") for _ in predictions] # 去掉空格
    # print(predictions[:k])
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000 # 如果预测的item不在all_items中，将其score设置为-1000

    for b in range(B): # 对于一个batch里的每个样本
        batch_seqs = predictions[b * k: (b + 1) * k] # k个预测的item
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True) # 按照score降序排列
        target_item = targets[b]
        one_results = [] # 长度为k的分数
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results

def get_top1_results(predictions, targets, all_items=None):
    # predictions: List[str] size = B(batch size)*k
    # targets: List[str] size = B
    # k: int
    # all_items: List[str] or None
    results = []
    B = len(targets)
    predictions = [_.split("user will visit POI index ")[-1].split(".")[0] for _ in predictions] # 取出最后一个Response:后的字符串，即预测的item
    predictions = [_.strip().replace(" ","") for _ in predictions] # 去掉空格
    # print(predictions[:1])
    if all_items is not None:
        predictions = [seq if seq in all_items else None for seq in predictions] # 如果预测的item不在all_items中，将其设置为None

    for b in range(B): # 对于一个batch里的每个样本
        batch_seqs = predictions[b: b + 1] # k个预测的item
        target_item = targets[b]
        one_results = [] # 长度为k的分数
        for pred in batch_seqs:
            if pred == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results

def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        elif m.lower().startswith("map"):
            k = int(m.split("@")[1])
            res[m] = map_k(topk_results, k)
        else:
            raise NotImplementedError

    return res


def ndcg_k(topk_results, k):
    # 归一化折损累计增益（NDCG）
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        ndcg += next((1 / math.log(i + 2, 2) for i in range(len(res)) if res[i] == 1), 0.0)
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit