import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn import Transformer

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()
class MotionDataset(Dataset):
    def __init__(self, data, labels):
        """
        Initialize the dataset with data and labels.
        Args:
        data (Tensor): The data points of the dataset.
        labels (Tensor): The labels for each data point.
        """
        self.data = data
        self.labels = labels
    
    def __len__(self):
        """Return the number of samples."""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Fetch the item (data, label) by index.
        Args:
        idx (int): The index of the item.
        Returns:
        Tuple[Tensor, Tensor]: The data and its corresponding label.
        """
        return self.data[idx], self.labels[idx]

# 定义模型
class ShapeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=2, num_layers=1, dim_feedforward=256):
        super(ShapeClassifier, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)  # 将输入转换到适合Transformer的维度
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                       num_decoder_layers=num_layers, dim_feedforward=dim_feedforward)
        self.fc_out = nn.Linear(d_model, num_classes)  # 分类头

    def forward(self, src):
        src = self.input_fc(src)
        src = self.pos_encoder(src)
        output = self.transformer(src, src)
        output = self.fc_out(output.mean(dim=1))  # 使用平均池化来合并时间维度的信息
        return output

# 训练函数
def train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()  # 清除GPU缓存

# 主函数
def main():
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load('data/shape_paths_data.pt')
    labels = torch.load('data/shape_labels.pt')
    dataset = MotionDataset(data, labels)
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # 检查数据集长度
    print(f"Total number of samples in dataset: {len(dataset)}")

    # 如果使用 DataLoader，检查每个批次的索引
    for i, (data, labels) in enumerate(loader):
        print(f"Batch {i}: Data shape {data.shape}, Labels shape {labels.shape}")
        if data.shape[0] != labels.shape[0]:
            print(f"Warning: Mismatch in batch {i} sizes. Data size {data.shape[0]}, Labels size {labels.shape[0]}")

    # loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    
    # model = ShapeClassifier(input_dim=30, num_classes=3).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # loss_fn = nn.CrossEntropyLoss()

    # for epoch in range(10):
    #     train(model, loader, loss_fn, optimizer, device)
    #     print(f"Epoch {epoch+1} complete.")

if __name__ == "__main__":
    main()
