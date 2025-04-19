import os
import scipy.io as sio
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T

class FMRIDataset(InMemoryDataset):
    def __init__(self, root, fixed_steps=200, threshold=0.7, transform=None, pre_transform=None):
        self.fixed_steps = fixed_steps
        self.threshold = threshold
        self.eps = 1e-8

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.mat')]

    @property
    def processed_file_names(self):
        return ['fmri_processed_data.pt']

    def download(self):
        # 数据集是本地的，无需下载
        pass

    def process(self):
        data_list = []
        for filename in self.raw_file_names:
            try:
                mat_path = os.path.join(self.raw_dir, filename)
                mat_data = sio.loadmat(mat_path)
                signals = mat_data['ROISignals'].astype(np.float32)
                print(f"Processing {filename}, shape: {signals.shape}")

                # 补齐或截断时间维度
                signals = self._align_time_dimension(signals)

                # 构建图数据
                data = Data(
                    x=self._extract_node_features(signals),
                    edge_index=self._build_functional_connectivity(signals),
                    y=self._get_label(filename)
                )
                data_list.append(data)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        if not data_list:
            raise RuntimeError("No valid samples processed.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("Done!")

    def _align_time_dimension(self, signals):
        if signals.shape[0] > self.fixed_steps:
            return signals[:self.fixed_steps]
        elif signals.shape[0] < self.fixed_steps:
            return np.pad(signals, ((0, self.fixed_steps - signals.shape[0]), (0, 0)))
        return signals

    def _extract_node_features(self, signals):
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_signals = (signals - np.nanmean(signals, axis=0)) / \
                           (np.nanstd(signals, axis=0) + self.eps)
        norm_signals = np.nan_to_num(norm_signals)

        features = np.column_stack([
            np.mean(norm_signals, axis=0),
            np.std(norm_signals, axis=0),
            np.median(norm_signals, axis=0),
            np.ptp(norm_signals, axis=0)
        ])
        return torch.tensor(features, dtype=torch.float32)

    def _build_functional_connectivity(self, signals):
        with np.errstate(invalid='ignore'):
            corr = np.corrcoef(signals.T)
        corr = np.nan_to_num(corr)

        row, col = np.where(np.triu(np.abs(corr), k=1) > self.threshold)
        edge_index = np.vstack([row, col])
        return torch.tensor(edge_index, dtype=torch.long)

    def _get_label(self, filename):
        if '-1-' in filename:
            return torch.tensor(0, dtype=torch.long)
        elif '-2-' in filename:
            return torch.tensor(1, dtype=torch.long)
        else:
            raise ValueError(f"Invalid label in filename: {filename}")

# 外部调用接口
def get_dataset(data_root):
    dataset = FMRIDataset(root=data_root)
    dataset.transform = T.Compose([
        T.NormalizeFeatures(),
        T.AddSelfLoops(),
        T.ToUndirected()
    ])
    return dataset
