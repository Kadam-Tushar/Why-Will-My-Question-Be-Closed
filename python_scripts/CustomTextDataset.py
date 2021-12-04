from torch.utils.data import Dataset, DataLoader

class CustomTextDataset(Dataset):
    def __init__(self, txt,pos_tags,labels):
        self.labels = labels
        self.text = txt
        self.pos = pos_tags
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        pos_tags = self.pos[idx]
        text = self.text[idx]
        
#         sample = {"Text": text, "Class": label}
        return (text,pos_tags,label)
