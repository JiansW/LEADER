from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler

class HalfShuffledSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.indices = list(range(len(data_source)))
        self.roop_M = 2
        self.rem = -1

    def half_shuffle(self):
        # epoch为奇数时，将indices的奇数部分进行随机打乱，偶数部分不变
        # epoch为偶数时，将indices的偶数部分进行随机打乱，奇数部分不变
        self.rem += 1
        if self.rem == self.roop_M:
            self.rem = 0
            self.roop_M += 1
        indices = deepcopy(self.indices)
        len_indices = len(indices)
        pad = self.roop_M - len_indices % self.roop_M
        paded_indices = np.concatenate([indices, np.zeros(pad, dtype=np.int32)])
        paded_indices = paded_indices.reshape((-1, self.roop_M))
        odd_indices = [paded_indices[:, :self.rem], paded_indices[:, self.rem+1:]]
        odd_indices = np.concatenate(odd_indices, axis=1)
        odd_indices = odd_indices.reshape(-1)
        np.random.shuffle(odd_indices[:len_indices])
        odd_indices = odd_indices.reshape((-1, self.roop_M-1))
        paded_indices[:, :self.rem] = odd_indices[:, :self.rem]
        paded_indices[:, self.rem+1:] = odd_indices[:, self.rem:]
        indices = paded_indices.reshape(-1)[:len_indices]
        return indices     

    def __iter__(self):
        indices = self.half_shuffle()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

class CollateFunction:
    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        coords   = [item[0] for item in batch]
        coords_w = [item[1] for item in batch]
        feats    = [item[2] for item in batch]
        poses    = [item[3] for item in batch]
        
        stacked_feats = np.concatenate(feats, axis=0)
        stacked_coords = np.concatenate(coords, axis=0)
        stacked_coords_w = np.concatenate(coords_w, axis=0)
        stacked_poses = np.vstack(poses)

        # offset = [sum(len(item[0]) for item in batch[:i+1]) for i in range(len(batch))]
        # offset = np.array(offset, dtype=np.int32)
        
        offset = [len(item[0]) for item in batch]
        offset = np.cumsum(offset, dtype=np.int32)

        return {
            'feat': stacked_feats,
            'coord': stacked_coords,
            'coord_w': stacked_coords_w,
            'offset': offset,
            'pose': stacked_poses
        }