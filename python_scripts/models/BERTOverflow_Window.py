from modules import *
import torch.nn as nn


#  Recurrent neural network (many-to-one)

class BERTOverflow_Window(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,bert):
        super(BERTOverflow_Window, self).__init__()
        self.bert = bert.to(device)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size =  bert.config.to_dict()['hidden_size']
        self.gru = nn.GRU(self.embedding_size, hidden_size, num_layers, batch_first=True)
        self.stride = 400 
        start_ind = 0 
        actual_seq_len = 0 
        while(start_ind + 512 <= sequence_length):
            start_ind += self.stride
            actual_seq_len += 512 
        

        self.fc = nn.Linear(hidden_size * actual_seq_len , num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        with torch.no_grad():
            start_ind = 0 
            emb = []
            while(start_ind + 512 <= sequence_length):
                emb.append(self.bert(x[:,start_ind:start_ind + 512],output_hidden_states=True)[1][-2])
                start_ind += self.stride
            x = torch.cat(tuple(emb),axis=1)
        x = x.to(device)


         
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate 
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

