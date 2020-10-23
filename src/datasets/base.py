import numpy as np
from torch.utils.data import Dataset


class FewShotDataset(Dataset): # Not used Anymore
    def __init__(self, datasets, iterations, N_shot, N_query, transforms=None):
        super().__init__()
        
        self.iterations = iterations
        self.transforms = transforms
        self.N_query = N_query
        self.N_shot = N_shot
        self.episodes = []
        self.datasets = datasets
        
        for it in range(self.iterations):
            dataset_idx = np.random.randint(len(self.datasets))
            random_idxs = np.random.permutation(len(self.datasets[dataset_idx]))
            self.episodes.append([(dataset_idx, img_idx) for img_idx in random_idxs[:self.N_query + self.N_shot]])
        
    def __len__(self):
        return self.iterations
    
    def __getitem__(self, idx):
        sample = [self.datasets[dataset_idx][data_idx]
                  for dataset_idx, data_idx in self.episodes[idx]]
        support = sample[:self.N_shot]
        query = sample[-self.N_query:]
        assert len(support) + len(query) == len(sample)
        return support, query

class MedicalFewshotDataset(Dataset):
    def __init__(self, datasets_list: list, shots=5):
        super().__init__()

        self.shots = shots
        self.datasets_list = []
        self.idxs = []
        for dataset_id, datasets in enumerate(datasets_list): # Organ
            if len(datasets) > 1: # An organ with more than one patient!
                self.datasets_list.append(datasets)
                for i in range(len(datasets)): # Patient support
                    for j in range(i+1, len(datasets)): # Patient query
                        self.idxs.append((dataset_id, i, j))
                        self.idxs.append((dataset_id, j, i))
    
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self,idx):
        dataset_id, vol_support, vol_query = self.idxs[idx]
        support_vol = self.datasets_list[dataset_id][vol_support]
        query_vol = self.datasets_list[dataset_id][vol_query]

        support_chunk_len = int(len(support_vol) / self.shots)
        
        support = [support_vol[i] for i in range(int(support_chunk_len / 2),len(support_vol),support_chunk_len)]
        query = [query_vol[i] for i in range(len(query_vol))]

        return support, query

class MedicalNormalDataset(Dataset):
    def __init__(self, dataset_list: list):
        
        self.datasets = []
        self.len_all = 0
        for dataset in dataset_list:
            self.datasets.append((len(dataset), dataset))
            self.len_all += len(dataset)

    def __len__(self):
        return self.len_all
    
    def __getitem__(self, idx):
        curser = idx
        for dataset_len, dataset in self.datasets:
            if curser < dataset_len:
                return dataset[curser]
            else:
                curser -= dataset_len
        raise Exception("IDX Not found!")
