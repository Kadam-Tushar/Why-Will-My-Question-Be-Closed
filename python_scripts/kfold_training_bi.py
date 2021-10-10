from modules import * 
from CustomTextDataset import *
from models.GRU import *
from models.Bi_GRU import *

import modules
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv(dataset_path)

logging.info("Done reading dataset csv file : {}".format(dataset_path))

title_body = torch.load(title_body_list_path)
logging.info("Done reading list of title_body file : {}".format(title_body_list_path))

vocab = torch.load(export_path+"vocab.v")
logging.info("Done reading list of vocab file : {}".format(export_path + "vocab.v"))
logging.info("sequnce length:{}".format(title_body.size(1)))

# K fold 
skf = StratifiedKFold(n_splits=5,shuffle = True, random_state = seed_val)
fold = 0 
for train_index, test_index in skf.split(title_body,df[col]):
    
    train_dataset =  CustomTextDataset(title_body[train_index],df.loc[train_index][col].to_numpy())
    test_dataset =  CustomTextDataset(title_body[test_index],df.loc[test_index][col].to_numpy())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    # Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
    model = Bi_GRU(input_size, hidden_size, num_layers, num_classes,len(vocab)).to(device)

    model_path = ".." + path_sep + 'trained_models' + path_sep + 'kfold' + path_sep
    m_name = "{}_{}.model".format(model_name,fold)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    update_interval = good_update_interval(total_iters=len(train_loader), num_desired_updates=10)
    logging.info("Training on Fold: {}".format(fold))
    logging.info("Number of batches: {} and update interval : {}".format(len(train_loader),update_interval))

        # Train Network
    for epoch in range(num_epochs):
        running_loss = 0.0 
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
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
        

            
    torch.save(model,model_path+m_name)
    logging.info("model saved!: {}".format(m_name))
    fold+=1



