#!/usr/bin/env python
# coding: utf-8
import modules 
import CustomeTextDataset

# In[1]:





# In[2]:

dataset_path = '../Dataset/title_body.pt'
df = pd.read_csv(')


# In[3]:


# Encode the sentence
tz = BertTokenizer.from_pretrained("bert-base-cased")
def tokenize(df,max_length):
    input_ids = []
    for sent in df:
        encoded = tz.encode_plus(
            text=sent,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = max_length,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            truncation=True
        )
        # Get the input IDs and attention mask in tensor format
        input_ids.append(encoded['input_ids'])
        
        
    return torch.tensor(input_ids)
        
    


# In[4]:


title_max_length = 150
body_max_length = 1500 
tags_max_length = 150 
df.replace(np.nan, '', inplace=True)
title_tensor = tokenize(df['Title'],title_max_length)
body_tensor = tokenize(df['Body'],body_max_length)
tags_tensor = tokenize(df['Tags'],tags_max_length)


# In[5]:


torch.save(title_tensor, 'title_tensor.pt')
torch.save(body_tensor, 'body_tensor.pt')
torch.save(tags_tensor, 'tags_tensor.pt')
torch.save(tz.vocab,'vocab.v')


# In[6]:



# define data set object
dataset = CustomTextDataset(title_tensor,df['closed'].to_numpy())
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed_val))


# In[7]:



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 256
hidden_size = 256
num_layers = 2
num_classes = 2
sequence_length = title_tensor.size(1)
learning_rate = 0.005
batch_size = 2048
num_epochs = 3

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,num_embeddings,embedding_size = 256):
        super(RNN, self).__init__()
        self.emb = nn.Embedding(num_embeddings,embedding_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        
        x = self.emb(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


# In[8]:


# Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
model = RNN(input_size, hidden_size, num_layers, num_classes,len(tz.vocab)).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent update step/adam step
        optimizer.step()

        

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")


# In[9]:


torch.save(model,"RNN_title_0.model")


# In[ ]:




