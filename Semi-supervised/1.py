from datasets import FMRIDataset
import torch

# 指定你的数据集路径
data_root = 'C:/Users/Lenovo/Desktop/datas'

# 加载数据集
dataset = FMRIDataset(root=data_root)

# 打印整体数据集信息
print(f"总样本数: {len(dataset)}")

# 查看第一个样本
data = dataset[0]
print(f"\n第一个样本信息:")
print(f"节点特征 x 的形状: {data.x.shape}")          # (num_nodes, num_node_features)
print(f"边索引 edge_index 的形状: {data.edge_index.shape}")  # (2, num_edges)
print(f"标签 y: {data.y}")
print(f"是否有自环: {torch.any(data.edge_index[0] == data.edge_index[1])}")
print(f"是否为无向图: {'对称' if torch.equal(data.edge_index, data.edge_index.flip(0)) else '非对称'}")

# 检查所有样本的维度一致性
node_counts = set()
feature_dims = set()
for i, g in enumerate(dataset):
    node_counts.add(g.x.shape[0])
    feature_dims.add(g.x.shape[1])
    if g.edge_index.shape[1] == 0:
        print(f"警告：样本 {i} 没有边！")

print(f"\n所有图的节点数: {node_counts}")
print(f"所有节点特征维度: {feature_dims}")
