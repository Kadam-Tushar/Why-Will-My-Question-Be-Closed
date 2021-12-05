from torch.utils.data import Dataset, DataLoader

class CustomTextDataset(Dataset):
    def __init__(self, txt,labels):
        self.labels = labels
        self.text = txt
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.text[idx]
        
#         sample = {"Text": text, "Class": label}
        return (text,label)
