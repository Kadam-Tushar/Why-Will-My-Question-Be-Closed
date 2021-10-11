from modules import * 
from CustomTextDataset import *
from models.GRU import *
import modules
from torchmetrics import Accuracy
from torchmetrics import Precision
from torchmetrics import F1
from torchmetrics import Recall

export_path = '..' + path_sep + 'Dataset' + path_sep + "kfold" + path_sep 
kfold = 3 


def print_metrics(pred_full_path,target_full_path):
    preds = torch.load(pred_full_path)
    target = torch.load(target_full_path)
    accuracy = Accuracy().to(device)
    precison_macro = Precision(average = 'macro',num_classes = num_classes).to(device)
    precison_micro = Precision(average = 'micro',num_classes = num_classes).to(device)
    f1_micro = F1(num_classes=num_classes,average = 'micro').to(device)
    f1_macro = F1(num_classes=num_classes,average = 'macro').to(device)
    recall_micro = Recall(average = 'micro' , num_classes = num_classes).to(device)
    recall_macro = Recall(average = 'macro' , num_classes = num_classes).to(device)

    logging.info(f"Accuracy : {accuracy(preds, target)*100:2f}")
    logging.info(f"Precision macro : {precison_macro(preds, target)*100:2f}")
    logging.info(f"Precision micro : {precison_micro(preds, target)*100:2f}")
    logging.info(f"Recall macro : {recall_macro(preds, target)*100:2f}")
    logging.info(f"Recall micro : {recall_micro(preds, target)*100:2f}")


for fold in range(kfold):
    m_name = "{}_{}.model".format(model_name.replace(".model",""),fold)
    logging.info("Generating metrics for  model : {}".format(m_name))
    for s in ['train','test']:
        preds_path = s + "_" + m_name + "_" + "preds.pt"
        target_path = s + "_" + m_name + "_" + "target.pt"
        logging.info(s+" metrics")
        print_metrics(export_path + preds_path, export_path + target_path)
        logging.info("\n")



