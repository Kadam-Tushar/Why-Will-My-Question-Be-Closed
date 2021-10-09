from modules import * 
from CustomTextDataset import *
from models.GRU import *
import modules

dataset_name = 'title_body.csv'
dataset_path = '..' + path_sep + 'Dataset' + path_sep + dataset_name
export_path = '..' + path_sep + 'Dataset' + path_sep 
title_body_list_path = '..' + path_sep + 'Dataset' + path_sep + "title_body.pt"
df = pd.read_csv(dataset_path)
logging.info("Done reading dataset csv file : {}".format(dataset_path))

title_body = torch.load(title_body_list_path)
logging.info("Done reading list of title_body file : {}".format(title_body_list_path))

vocab = torch.load(export_path+"vocab.v")
logging.info("Done reading list of vocab file : {}".format(export_path + "vocab.v"))
logging.info("sequnce length:{}".format(title_body.size(1)))


# define data set object
dataset = CustomTextDataset(title_body,df['closed'].to_numpy())
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed_val))
modules.sequence_length = title_body.size(1)

# Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
model = GRU(input_size, hidden_size, num_layers, num_classes,len(vocab)).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
update_interval = good_update_interval(total_iters=len(train_loader), num_desired_updates=10)
logging.info("Number of batches: {} and update interval : {}".format(len(train_loader),update_interval))
eps =torch.ones(batch_size,num_classes)* (1e-8) 
eps = eps.to(device)
       
# Train Network
for epoch in range(num_epochs):
    running_loss =0.0 
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        scores += eps
        #print(scores.size())
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        running_loss += loss.item()* data.size(0)
        print(loss.item())
        if batch_idx % update_interval == 0:
            logger.info("[{},{}] loss: {}".format(epoch + 1,batch_idx, running_loss/(1 + update_interval*data.size(0))))
            running_loss = 0.0 
        
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


logger.info(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
logger.info(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
torch.save(model,export_path+"GRU_title_body.model")

