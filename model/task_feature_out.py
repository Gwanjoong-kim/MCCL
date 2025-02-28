import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from model.modeling_t5 import T5ForConditionalGeneration

import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE  # scikit-learn t-SNE
import pickle
from tqdm import tqdm
import pdb
import gc


class Conti_Dataset(Dataset):
    
    def __init__(self, train_loader, task_id):
        self.data = []
        self.labels = []
        ### task order에 따라 dataset_dict 달라져야 됨
        dataset_dict = {'0': "dbpedia", '1': "yahoo", '2': "amazon", '3': "yelp", '4': "agnews"}
        self.length = len(train_loader[dataset_dict[str(task_id)]][0].dataset)
        
        for task in range(task_id+1):
            self.data.extend(torch.unbind(torch.tensor(train_loader[dataset_dict[str(task)]][0].dataset["input_ids"]), dim=0))
            self.labels.extend([task_id] * len(train_loader[dataset_dict[str(task)]][0].dataset))

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return self.length


class SimpleEmbeddingNet(nn.Module):
    def __init__(self, lm, input_dim=1024, embed_dim=1024, num_hidden=512):
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
        # (B, 512, 1024) -> (B, 1024)
        # print("input_shape of x: ", x.size())
        # pdb.set_trace()
        x = self.lm(x.to(self.device)).last_hidden_state
        x = x.mean(dim=1)  # (B, 1024)
        embed = self.fc(x.to(self.device))  # (B, embed_dim)
        return embed


def contrastive_loss_multi_class(outputs, labels, device, margin=2.0):
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
    return loss_mean


def train_epoch(model, dataloader, margin=2.0):
    model.train()
    total_loss = 0
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
    
    task_feature_vector = {}
    
    # real_embeds = {}
    
    # for key in list(embeds.keys()):
    #     if len(embeds[key]) != 0:
    #         real_embeds[key] = embeds[key]

    # embeds = real_embeds
    
    # embeds에 포함되어있는 Task의 수에 맞게 train_epoch를 돌림
    # Task가 늘어날 수록 embeds dictionary에 담겨있는 Task의 수가 늘어남, 이에 따라 Dataloader도 늘어남
        
    for task in range(task_id+1):
        
        task_feature_vector[str(task)] = []
        dataset = Conti_Dataset(train_loader, task)
        subset_indices = list(range(int(max(len(dataset) * 0.1, 2)))) 
        subset = Subset(dataset, subset_indices)
        dataloader = DataLoader(subset, batch_size=8, shuffle=True, drop_last=True)
        
        # pdb.set_trace()
        
        for epoch in tqdm(range(1, epochs+1), desc = "Contrastive training"):
            train_epoch(model, dataloader, margin=2.0)
            # print(f"Task {task} Epoch {epoch} Done",flush=True)
    
        model.eval()
        
        eval_dataloader = DataLoader(subset, batch_size=1, shuffle=False, drop_last=True)
                
        for x, y in tqdm(eval_dataloader, desc = "Extracting features"):
            if task in y:
                feature = model(x)
                task_feature_vector[str(task)].append(feature.detach())
                

    # pickle.dump(task_feature_vector, open(f'til_{task_id}_task_feature_vector.pkl', 'wb'))
    pickle.dump(task_feature_vector, open(f'{output_dir}/til_{task_id}_task_feature_vector.pkl', 'wb'))
    print(f"Task {task_id} feature vector saved")
    return task_feature_vector