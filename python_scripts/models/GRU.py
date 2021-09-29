from modules import *
import torch.nn as nn

#  Recurrent neural network (many-to-one)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,num_embeddings,embedding_size = 256):
        super(GRU, self).__init__()
        self.emb = nn.Embedding(num_embeddings,embedding_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        
        x = self.emb(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

