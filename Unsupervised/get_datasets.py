import os
import numpy as np
import scipy.io as sio
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree
import torch_geometric.transforms as T

class FMRI_Dataset(InMemoryDataset):
    def __init__(self, root, threshold=0.3, feature_type='timeseries', 
                 transform=None, pre_transform=None):
        self.threshold = threshold
        self.feature_type = feature_type
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # 请确保你的 .mat 文件放在 root/raw/ 目录下
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.mat')]

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def process(self):
        data_list = []
        error_files = []

        for mat_path in self.raw_paths:
            filename = os.path.basename(mat_path)
            try:
                # === 1. 加载数据 ===
                mat_data = sio.loadmat(mat_path)
                if 'ROISignals' not in mat_data:
                    raise ValueError(f"文件 {filename} 缺少 ROISignals 键")
                
                # === 2. 验证或补全形状 ===
                roi_signals = mat_data['ROISignals'].astype(np.float32)
                if roi_signals.shape[1] != 116:
                    raise ValueError(f"文件 {filename} 形状错误，应为 (*, 116)，实际为 {roi_signals.shape}")
                if roi_signals.shape[0] < 200:
                    pad_len = 200 - roi_signals.shape[0]
                    roi_signals = np.pad(roi_signals, ((0, pad_len), (0, 0)), mode='constant')
                elif roi_signals.shape[0] > 200:
                    roi_signals = roi_signals[:200, :]  # 截断多余部分

                # === 3. 构建邻接矩阵 ===
                corr_matrix = np.corrcoef(roi_signals.T)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                adj_matrix = (np.abs(corr_matrix) > self.threshold).astype(np.float32)

                if adj_matrix.sum() == 0:
                    raise ValueError(f"文件 {filename} 邻接矩阵为空（阈值 {self.threshold} 过高）")

                edge_index, _ = dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float32))
                edge_index = edge_index.contiguous().long()  # 确保 edge_index 连续且为 LongTensor

                if edge_index.dim() != 2 or edge_index.shape[0] != 2:
                    raise ValueError(f"edge_index 格式错误，当前为 {edge_index.shape}")

                # === 4. 节点特征 ===
                if self.feature_type == 'timeseries':
                    x = torch.tensor(roi_signals.mean(axis=0), dtype=torch.float32).view(-1, 1)
                else:
                    x = torch.ones(116, 1)

                # === 5. 标签解析 ===
                if "-1-" in filename:
                    y = 0
                elif "-2-" in filename:
                    y = 1
                else:
                    raise ValueError(f"文件 {filename} 缺少有效标签标识符 (-1- 或 -2-)")
                
                data = Data(x=x, edge_index=edge_index, y=torch.tensor([y]))
                data_list.append(data)
                print(f"✅ 成功处理 {filename}")

            except Exception as e:
                error_files.append(f"{filename}: {str(e)}")
                print(f"❌ 处理失败 {filename}: {str(e)}")

        # === 6. 空数据检查 ===
        if not data_list:
            raise RuntimeError("没有有效数据被处理，请检查错误日志！")

        # === 7. 特征工程 ===
        if self.feature_type == 'degree':
            self._add_degree_features(data_list)

        # === 8. 数据转换 ===
        self._auto_configure_transforms(data_list)

        # === 9. 保存数据 ===
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _add_degree_features(self, data_list):
        """添加节点度数特征"""
        all_degrees = []
        for data in data_list:
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
            all_degrees.append(deg)
            if deg.numel() == 0:
                raise ValueError("检测到空度数张量，请检查邻接矩阵！")
        
        degrees = torch.cat(all_degrees, dim=0).float()
        mean, std = degrees.mean().item(), degrees.std().item()
        
        for data in data_list:
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes).float()
            data.x = ((deg - mean) / std).view(-1, 1)

    def _auto_configure_transforms(self, data_list):
        """安全配置数据转换"""
        try:
            max_degree = max(
                degree(data.edge_index[0]).max().item()
                for data in data_list
                if data.edge_index.numel() > 0  # 过滤空边
            )
        except ValueError:
            raise RuntimeError("无法计算最大度数，请检查边是否存在！")
        
        # 对于 timeseries 特征，我们保留稀疏格式，不调用 ToDense()
        if self.feature_type == 'timeseries':
            self.transform = T.NormalizeFeatures()
        else:
            if max_degree < 1000:
                self.transform = T.OneHotDegree(max_degree)
            else:
                self.transform = T.Compose([
                    T.NormalizeFeatures(),
                    T.ToDense()
                ])

def get_dataset(data_dir, threshold=0.3, feature_type='timeseries'):
    """获取数据集实例"""
    return FMRI_Dataset(
        root=data_dir,
        threshold=threshold,
        feature_type=feature_type
    )
