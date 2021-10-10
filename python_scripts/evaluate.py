from modules import * 
from CustomTextDataset import *
from models.GRU import *
import modules


title_body_list_path = '..' + path_sep + 'Dataset' + path_sep + "title_body.pt"
df = pd.read_csv(dataset_path)
logging.info("Done reading dataset csv file : {}".format(dataset_path))

title_body = torch.load(title_body_list_path)
logging.info("Done reading list of title_body file : {}".format(title_body_list_path))

vocab = torch.load(export_path+"vocab.v")
logging.info("Done reading list of vocab file : {}".format(export_path + "vocab.v"))
logging.info("sequnce length:{}".format(title_body.size(1)))


# define data set object
dataset = CustomTextDataset(title_body,df[col].to_numpy())
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed_val))


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = torch.load(model_path+model_name,map_location=device)

logging.info("Done loading model")

def predictions(loader,model):
    # Set model to eval
    model.eval()
    preds = torch.empty((0), dtype=torch.int32, device = device)
    target = torch.empty((0), dtype=torch.int32, device = device)
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            preds = torch.cat((preds,predictions),0)
            target = torch.cat((target,y),0)

    # Toggle model back to train
    model.train()
    return preds,target



def check_accuracy(loader, model):
    # gradient descent update step/adam step
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    model.train()
    return num_correct / num_samples

preds,target = predictions(test_loader,model)

torch.save(preds,export_path + "multi_preds.pt")
torch.save(target,export_path + "multi_target.pt")

logging.info("Saved predictions and targets")






