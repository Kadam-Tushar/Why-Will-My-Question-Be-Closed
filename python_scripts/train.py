from modules import * 
from CustomTextDataset import *
from models.GRU import *
from models.BERT import *
from models.BERTOverflow import *
import modules
from transformers import AutoModelForTokenClassification
from transformers import BertTokenizer, BertModel

title_body_tags_list_path = '..' + path_sep + 'Dataset' + path_sep + prob+ prefix+ "_title_body_tags.pt"
df = pd.read_csv(dataset_path)

logging.info("Done reading dataset csv file : {}".format(dataset_path))

title_body_tags = torch.load(title_body_tags_list_path)
logging.info("Done reading list of title_body_tags file : {}".format(title_body_tags_list_path))

vocab = torch.load(export_path+prob+prefix+"_vocab.v")
logging.info("Done reading list of vocab file : {}".format(export_path + prob+prefix+"_vocab.v"))
logging.info("sequnce length:{}".format(title_body_tags.size(1)))



# define data set object
dataset = CustomTextDataset(title_body_tags,df[col].to_numpy())
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed_val))
modules.sequence_length = title_body_tags.size(1)

# Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
model = 0
if model_type == "GRU":
    model = GRU(input_size, hidden_size, num_layers, num_classes,len(vocab)).to(device)

if model_type == "BERT":
    # Bert constants 
    bert_model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id
    max_input_length = tokenizer.max_model_input_sizes[bert_model_name]
    bert = BertModel.from_pretrained(bert_model_name, output_attentions=True, output_hidden_states=True)
    model = BERT(input_size, hidden_size, num_layers, num_classes,bert).to(device)

if model_type == "BERTOverflow":
    bert = AutoModelForTokenClassification.from_pretrained("jeniya/BERTOverflow").to(device)
    model = BERTOverflow(input_size, hidden_size, num_layers, num_classes,bert).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
update_interval = good_update_interval(total_iters=len(train_loader), num_desired_updates=10)
logging.info("Number of batches: {} and update interval : {}".format(len(train_loader),update_interval))
       
# Train Network
for epoch in range(num_epochs):
    running_loss = 0.0 
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        if model_type == "BERT":
            data = data[:,:512]
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        #print(scores.size())
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        running_loss += loss.item()* data.size(0)
        if batch_idx % update_interval == 0:
            logger.info("[{},{}] loss: {}".format(epoch + 1,batch_idx, running_loss/(1 + update_interval*data.size(0))))
            running_loss = 0.0 
        
        # gradient descent update step/adam step
        optimizer.step()
    prefix = "Epoch_"+str(epoch)+"_"
    torch.save(model,model_path+prefix+model_name)
    logging.info("model saved!: {}".format(prefix+model_name))


        
