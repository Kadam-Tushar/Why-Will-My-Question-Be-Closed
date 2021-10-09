from modules import * 
from CustomTextDataset import *
from models.GRU import *
import modules
from torchmetrics import Accuracy
from torchmetrics import Precision
from torchmetrics import F1
from torchmetrics import Recall

export_path = '..' + path_sep + 'Dataset' + path_sep 

preds = torch.load(export_path+"preds.pt")
target = torch.load(export_path+"target.pt")

accuracy = Accuracy().to(device)
precison_macro = Precision(average = 'macro',num_classes = num_classes).to(device)
precison_micro = Precision(average = 'micro',num_classes = num_classes).to(device)
f1_micro = F1(num_classes=num_classes,average = 'micro').to(device)
f1_macro = F1(num_classes=num_classes,average = 'macro').to(device)
recall_micro = Recall(average = 'micro' , num_classes = num_classes).to(device)
recall_macro = Recall(average = 'macro' , num_classes = num_classes).to(device)

logging.info(f"Accuracy on test set: {accuracy(preds, target)*100:2f}")
logging.info(f"Precision macro : {precison_macro(preds, target)*100:2f}")
logging.info(f"Precision micro : {precison_micro(preds, target)*100:2f}")
logging.info(f"Recall macro : {recall_macro(preds, target)*100:2f}")
logging.info(f"Recall micro : {recall_micro(preds, target)*100:2f}")