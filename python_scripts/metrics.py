from modules import * 
from CustomTextDataset import *
from models.GRU import *
import modules
from torchmetrics import Accuracy
from torchmetrics import Precision
from torchmetrics import F1
from torchmetrics import Recall

kfold = 1
export_path = (export_path + "kfold" + path_sep) if kfold > 1 else export_path


rec = [0,0]
acc = 0
pre = [0,0]
f1  = [0,0] 

def print_metrics(pred_full_path,target_full_path):
    global rec,acc,pre,f1
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
    logging.info(f"f1 micro : {f1_micro(preds, target)*100:2f}")
    logging.info(f"f1 macro : {f1_macro(preds, target)*100:2f}")

    rec[0]+= recall_micro(preds, target)*100
    rec[1]+= recall_macro(preds, target)*100
    pre[0]+= precison_micro(preds, target)*100
    pre[1]+= precison_macro(preds, target)*100
    f1[0]+= f1_micro(preds, target)*100
    f1[1]+= f1_macro(preds, target)*100
    acc +=  accuracy(preds, target)*100

if kfold > 1 :
    for fold in range(kfold):
        m_name = "{}_{}.model".format(model_name.replace(".model",""),fold)
        logging.info("Generating metrics for  model : {}".format(m_name))
        for s in ['test']:
            preds_path = s + "_" + m_name + "_" + "preds.pt"
            target_path = s + "_" + m_name + "_" + "target.pt"
            logging.info(s+" metrics")
            print_metrics(export_path + preds_path, export_path + target_path)
            logging.info("\n")
else:
    m_name = model_name.replace(".model","")
    logging.info("Generating metrics for  model : {}".format(m_name))
    for s in ['test']:
        preds_path = s + "_" + m_name + "_" + "preds.pt"
        target_path = s + "_" + m_name + "_" + "target.pt"
        logging.info(s+" metrics")
        print_metrics(export_path + preds_path, export_path + target_path)
        logging.info("\n")



logging.info("acc: {} \n rec_mic: {} \n rec_mac: {} \n pre_mic: {} \n  pre_mac: {} \n f1_mic: {} f1_mac: {}"
                                    .format(acc/kfold,rec[0]/kfold,rec[1]/kfold,pre[0]/kfold,pre[1]/kfold,f1[0]/kfold,f1[1]/kfold))


