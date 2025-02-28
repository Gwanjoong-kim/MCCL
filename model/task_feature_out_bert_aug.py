import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import pickle
from tqdm import tqdm
import pdb


class Conti_Dataset(Dataset):
    
    def __init__(self, train_loader, task_id, output_dir):
        self.data = []
        self.labels = []

        if task_id >0:
                # pdb.set_trace()
                mean_of_input_query = pickle.load(open(f"{output_dir}/til_{task_id-1}_mean_of_input_query.pkl", "rb"))
                    
        std = 5
        
        # pdb.set_trace()
        
        self.data.extend(torch.tensor(train_loader[str(task_id)][0].dataset["input_ids"])[:200])
        self.labels.extend([task_id] * 200)
        # self.data.extend(torch.tensor(train_loader[str(task_id)][0].dataset["input_ids"]))
        # self.labels.extend([task_id] * len(train_loader[str(task_id)][0].dataset))
        
        for task in range(task_id):
            
            mean_tensor = torch.tensor(mean_of_input_query[str(task)], dtype=torch.float32)
            sampled_ids = (mean_tensor + torch.randn(200, 256) * std).int()
            sampled_ids = torch.clamp(sampled_ids, min=0, max=30521)
                       
            self.data.extend(torch.unbind(sampled_ids, dim=0))
            self.labels.extend([task] * sampled_ids.shape[0])
            
        self.length = len(self.data)
        
        # pdb.set_trace()

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return self.length


class SimpleEmbeddingNet(nn.Module):
    def __init__(self, lm, input_dim=768, embed_dim=768, num_hidden=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, embed_dim),
        )
        self.device = lm.device
        self.fc.to(self.device)
        self.lm = lm
        for param in self.lm.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.lm(x.to(self.device)).last_hidden_state
        x = x.mean(dim=1)  # (B, 1024)
        embed = self.fc(x.to(self.device))  # (B, embed_dim)
        return embed


def contrastive_loss_multi_class(outputs, labels, device, margin):
    """
    embeddings: (B, D)
    labels: (B,)  -> 0~(num_classes-1)
    margin: float
    """
    device = outputs.to(device)
    B = outputs.size(0)
    
    # (B, B) pairwise distance
    # cdist는 유클리디안 거리 계산 (p=2)
    distances = torch.cdist(outputs, outputs, p=2).to(device)  # (B, B)

    # (B, B) same_class_mask (같은 클래스면 1, 아니면 0)
    same_class_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float().to(device)  # (B, B)

    # positive_loss: same_class_mask * distance^2
    positive_loss = same_class_mask * (distances ** 2)

    # negative_loss: (1 - same_class_mask) * (max(margin - distance, 0))^2
    negative_loss = (1 - same_class_mask) * F.relu(margin - distances).pow(2)

    # 대각선(i == i)은 자기 자신 -> distance=0 -> 학습 의미 없으므로 제외
    diag_mask = torch.eye(B)
    diag_mask = diag_mask.to(device)
    positive_loss = positive_loss * (1 - diag_mask)
    negative_loss = negative_loss * (1 - diag_mask)

    # 전체 로스: 모든 원소 합 -> 평균
    loss = positive_loss + negative_loss
    loss_mean = loss.sum() / (B * (B - 1))  # (B*B 중 대각선 B개 빼서 B*(B-1))
    # print(f"Positive loss: {positive_loss.mean().item()}, Negative loss: {negative_loss.mean().item()}", flush=True)
    return loss_mean


def train_epoch(model, dataloader, margin=2.0):
    model.train()
    total_loss = 0
    optimizer = optim.Adam(model.parameters(), lr=3e-3)

    for x, y in dataloader:
        optimizer.zero_grad()

        outputs = model(x)  # (B, embed_dim)
        
        # Contrastive Loss
        loss = contrastive_loss_multi_class(outputs, y, device=x.device, margin=margin)
        # print("contrastive_loss: ", loss.item(), flush=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def get_task_feature_vector(train_loader, task_id, lm, output_dir, margin=2, epochs=5):
    
    model = SimpleEmbeddingNet(lm = lm)
    dataset = Conti_Dataset(train_loader, task_id, output_dir)        
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    
    task_feature_vector = {}
    mean_of_input_query = {}
    
    for task in range(task_id+1):
        mean_of_input_query[str(task)] = None
        task_feature_vector[str(task)] = []
        
    for task in range(task_id+1):
        
        for epoch in tqdm(range(1, epochs+1), desc = "Contrastive training"):
            train_epoch(model, dataloader, margin=2.0)
    
        model.eval()
        
        eval_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)
        
        count = 0
        
        for x, y in tqdm(eval_dataloader, desc = "Extracting features"):
            
            # pdb.set_trace()
            
            if task in y:
                feature = model(x)
                task_feature_vector[str(task)].append(feature.detach())
                
            x_vec = x.squeeze()
            count += 1
            if mean_of_input_query[str(y.item())] is None:
                # 첫번째 배치일 경우, 이동평균의 초기값으로 설정
                mean_of_input_query[str(y.item())] = x_vec.clone()
            else:
                # 이동평균 업데이트: new_avg = old_avg + (new_value - old_avg) / count
                mean_of_input_query[str(y.item())] = mean_of_input_query[str(y.item())] + (x_vec - mean_of_input_query[str(y.item())]) / count
                
        pickle.dump(mean_of_input_query, open(f"{output_dir}/til_{task_id}_mean_of_input_query.pkl", 'wb'))
                
    # pickle.dump(task_feature_vector, open(f'til_{task_id}_task_feature_vector.pkl', 'wb'))
    pickle.dump(task_feature_vector, open(f'{output_dir}/til_{task_id}_task_feature_vector.pkl', 'wb'))
    print(f"Task {task_id} feature vector saved")
    return task_feature_vector