from modules import *
import torch.nn as nn


#  Recurrent neural network (many-to-one)

class BERT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,bert):
        super(BERT, self).__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = bert.config.to_dict()['hidden_size']
        self.gru = nn.GRU(self.embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        with torch.no_grad():
            x = self.bert(x)[0]
         
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate 
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

