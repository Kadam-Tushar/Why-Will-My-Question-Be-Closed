from modules import * 
from CustomTextDataset import *
from models.GRU import *
import modules
from sklearn.manifold import TSNE


preds = torch.load(export_path + "class_emb.pt")
target = torch.load(export_path + "class_emb_target.pt")

preds_embedded = TSNE(n_components=num_classes, learning_rate='auto',init='random').fit_transform(preds)

