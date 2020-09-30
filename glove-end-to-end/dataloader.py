import pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader

class UtteranceDataset(Dataset):

    def __init__(self, filename1, filename2, filename3, filename4):
        
        utterances, labels, loss_mask, speakers = [], [], [], []
        
        with open(filename1) as f:
            for line in f:
                content = line.strip().split('\t')[1:]
                utterances.append(content)
        
        with open(filename2) as f:
            for line in f:
                content = line.strip().split('\t')[1:]
                labels.append([int(l) for l in content])
                
        with open(filename3) as f:
            for line in f:
                content = line.strip().split('\t')[1:]
                loss_mask.append([int(l) for l in content])
                
        with open(filename4) as f:
            for line in f:
                content = line.strip().split('\t')[1:]
                speakers.append([int(l) for l in content])

        self.utterances = utterances
        self.labels = labels
        self.loss_mask = loss_mask
        self.speakers = speakers
        
    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index): 
        s = self.utterances[index]
        l = self.labels[index]
        m = self.loss_mask[index]
        sp = self.speakers[index]
        return s, l, m, sp, []
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
    
    
def DialogLoader(filename1, filename2, filename3, filename4, batch_size, shuffle):
    dataset = UtteranceDataset(filename1, filename2, filename3, filename4)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return loader


class ShuffledUtteranceDataset(Dataset):
    
    def __init__(self, filename1, filename2, filename3, filename4):
        
        utterances, labels, loss_mask, speakers, r_indices = [], [], [], [], []
        f1 = open(filename1)
        f2 = open(filename2)
        f3 = open(filename3)
        f4 = open(filename4)
        
        for line1, line2, line3, line4 in zip(f1, f2, f3, f4):
                        
            content1 = line1.strip().split('\t')[1:]
            content2 = line2.strip().split('\t')[1:]
            content3 = line3.strip().split('\t')[1:]
            content4 = line4.strip().split('\t')[1:]
            
            indices = list(range(len(content1)))
            np.random.shuffle(indices)
            
            content1 = [content1[j] for j in indices]
            content2 = [content2[j] for j in indices]
            content3 = [content3[j] for j in indices]
            content4 = [content4[j] for j in indices]
            
            utterances.append(content1)
            labels.append([int(l) for l in content2])
            loss_mask.append([int(l) for l in content3])
            speakers.append([int(l) for l in content4])
            r_indices.append(indices)

        self.utterances = utterances
        self.labels = labels
        self.loss_mask = loss_mask
        self.speakers = speakers
        self.r_indices = r_indices
        
    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index): 
        s = self.utterances[index]
        l = self.labels[index]
        m = self.loss_mask[index]
        sp = self.speakers[index]
        i = self.r_indices[index]
        return s, l, m, sp, i
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]


def ShuffledDialogLoader(filename1, filename2, filename3, filename4, batch_size, shuffle):
    dataset = ShuffledUtteranceDataset(filename1, filename2, filename3, filename4)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return loader
    