from modules import * 
from CustomTextDataset import *
from models.GRU import *
import modules
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


title_body_tags_list_path = '../Dataset' + path_sep + prob+ prefix+ "_title_body_tags.pt"
df = pd.read_csv(dataset_path)

logging.info("Done reading dataset csv file : {}".format(dataset_path))

title_body_tags = torch.load(title_body_tags_list_path)
logging.info("Done reading list of title_body_tags file : {}".format(title_body_tags_list_path))

vocab = torch.load(export_path+prob+prefix+"_vocab.v")
logging.info("Done reading list of vocab file : {}".format(export_path + prob+prefix+"_vocab.v"))
logging.info("sequnce length:{}".format(title_body_tags.size(1)))


# define data set object
dataset = CustomTextDataset(title_body_tags,df[col].to_numpy())
train_size = int(0.80 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed_val))

# Sampling only subset around - 80% 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = torch.load(model_path+model_name,map_location=device)

logging.info("Done loading model: {}".format(model_name))

def class_embedding_GRU(model,x):
    with torch.no_grad():
        x = model.emb(x)
        x = x.to(device)
    h0 = torch.zeros(model.num_layers, x.size(0), model.hidden_size).to(device)
    out, _ = model.gru(x, h0)
    return out[:,-1,:]


def class_embedding(model,x):
    with torch.no_grad():
        x= model.bert(x[:,:512],output_hidden_states=True)[1][-2]
    x = x.to(device)
    h0 = torch.zeros(model.num_layers, x.size(0), model.hidden_size).to(device)
    out, _ = model.gru(x, h0)
    return out[:,-1,:]



def predictions(loader,model):
    # Set model to eval
    model.eval()
    preds = torch.empty((0), dtype=torch.int32, device = device)
    target = torch.empty((0), dtype=torch.int32, device = device)
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
            last = class_embedding_GRU(model,x)
            
            preds = torch.cat((preds,last),0)
            target = torch.cat((target,y),0)

    # Toggle model back to train
    model.train()
    return preds,target

def low_emb(dataset):
    if dataset == "train":
        preds,target = predictions(train_loader,model)
    else:
        preds,target = predictions(test_loader,model)
    torch.save(preds,export_path + dataset + "_" + prob + prefix +  "class_emb.pt")
    torch.save(target,export_path +dataset + "_" +  prob + prefix +  "class_emb_target.pt")
    logging.info("Saved class emb and their targets")

def model_outputs(loader, model):
    

    # Set model to eval
    model.eval()
    preds = torch.empty((0), dtype=torch.int32, device = device)
    target = torch.empty((0), dtype=torch.int32, device = device)

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, out = scores.max(1)
            

            preds = torch.cat((preds,out),0)
            target = torch.cat((target,y),0)
           

    # Toggle model back to train
    model.train()
    return preds,target




def cf_mat():
    preds,target = model_outputs(test_loader,model)
    #Get the confusion matrix
    cf_matrix = confusion_matrix(target.cpu(),preds.cpu())
    labels  = ['Open', 'Off-topic','Unclear','Broad','Opinion']
    sns.heatmap(cf_matrix/np.sum(cf_matrix), xticklabels=labels, yticklabels=labels, fmt='.2%',  annot=True).set_title('Confusion Matrix ')
    
    plt.savefig(model_name+"_confusion.png")



low_emb("train")
low_emb("test")
# cf_mat()






