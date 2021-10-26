from modules import * 
from CustomTextDataset import *
from models.GRU import *
import modules
from sklearn.model_selection import StratifiedKFold

title_body_tags_list_path = '..' + path_sep + 'Dataset' + path_sep + prob+ "_title_body_tags.pt"
df = pd.read_csv(dataset_path)

logging.info("Done reading dataset csv file : {}".format(dataset_path))

title_body_tags = torch.load(title_body_tags_list_path)
logging.info("Done reading list of title_body_tags file : {}".format(title_body_tags_list_path))

vocab = torch.load(export_path+prob+"_vocab.v")
logging.info("Done reading list of vocab file : {}".format(export_path + prob+"_vocab.v"))
logging.info("sequnce length:{}".format(title_body_tags.size(1)))


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

# K fold 
skf = StratifiedKFold(n_splits=30,shuffle = True, random_state = seed_val)
fold = 0 
for train_index, test_index in skf.split(title_body_tags,df[col]):
    
    train_dataset =  CustomTextDataset(title_body_tags[train_index],df.loc[train_index][col].to_numpy())
    test_dataset =  CustomTextDataset(title_body_tags[test_index],df.loc[test_index][col].to_numpy())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    model_path = ".." + path_sep + 'trained_models' + path_sep + 'kfold' + path_sep
    m_name = "{}_{}.model".format(model_name.replace(".model",""),fold)

    model = torch.load(model_path + m_name) 
    train_preds,train_target = predictions(train_loader,model)
    test_preds,test_target = predictions(test_loader,model)
    torch.save(train_preds,export_path + path_sep + "kfold" + path_sep +  "train_"+ m_name +"_preds.pt")
    torch.save(train_target,export_path + path_sep + "kfold" + path_sep +  "train_"+m_name+"_target.pt")
    torch.save(test_preds,export_path + path_sep + "kfold" + path_sep +  "test_"+m_name+"_preds.pt")
    torch.save(test_target,export_path + path_sep + "kfold" + path_sep +  "test_"+m_name+"_target.pt")

    logging.info("model: {}  saved train / test preds,targets ".format(m_name))
    fold+=1

