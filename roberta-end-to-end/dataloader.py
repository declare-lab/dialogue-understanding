import pandas as pd
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
        return s, l, m, sp
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
    
    
def DialogLoader(filename1, filename2, filename3, filename4, batch_size, shuffle):
    dataset = UtteranceDataset(filename1, filename2, filename3, filename4)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return loader