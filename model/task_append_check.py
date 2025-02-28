import pickle
import torch
import torch.nn as nn
import torch.distributions as dist
import pdb

def task_append_check(output_dir, task_id):
    task_embeds = pickle.load(open(f"{output_dir}/til_{task_id}_embed_prototypes.pkl", "rb"))
    feature_vector = pickle.load(open(f"{output_dir}/{task_id}_x_query.pkl", "rb"))
    
    #### {t-1}번째 task의 task_embeds와 {t}번째 task의 feature_vector 간의 softmax(cos-similarity) 비교, 확률 간 entropy 계산하여 정보량이 없다고 판단되면 false 반환
    #### 이전의 task_feature_vector와 상관관계가 많다면, 특정 task와의 거리가 상대적으로 가까울 것이므로, entropy가 낮을 것이다.
    #### task가 늘어날수록, task_embed와 feature_vector 간의 거리가 짧아지기에, task order별 상대적 비교를 해야 하지 않을까?,
    #### Entropy가 작다면, 이전 task와의 상관관계가 높다고 판단하여 False 반환
    
    prob = nn.functional.softmax(
        torch.cosine_similarity(feature_vector.unsqueeze(1), torch.stack(task_embeds[:task_id]).unsqueeze(0).to(feature_vector.device), dim=-1).squeeze(1), 
        dim=-1)
    
    # entropy = dist.Categorical(prob).entropy()
    entropy=torch.tensor(1.0)

    num_classes = prob.shape[-1]
    normalized_entropy = entropy/torch.log(torch.tensor(num_classes).float())
    # pdb.set_trace()
    print("Prob: ", prob, flush=True)
    print(f"{task_id}_Entropy: ", torch.mean(normalized_entropy), flush=True)
    
    if torch.mean(entropy) < 1:
        return False, torch.mean(normalized_entropy)
    else:
        return True, torch.mean(normalized_entropy)