from modules import * 
import CustomTextDataset
from transformers import AutoTokenizer
# import spacy 

# nlp = spacy.load("en_core_web_sm")

dataset_name = 'balanced_'+prob +'.csv'
dataset_path = '/scratch/tusharpk/Dataset' + path_sep + dataset_name
export_path = '/scratch/tusharpk/Dataset' + path_sep 
df = pd.read_csv(dataset_path)

logging.info("Done reading dataset csv file : {}".format(dataset_path))

# Encode the sentence
tz = BertTokenizer.from_pretrained("bert-base-cased") if model_type != "BERTOverflow" else  AutoTokenizer.from_pretrained("jeniya/BERTOverflow")
def tokenize(df,max_length):
    input_ids = []
    df_sen = df['title_body_tags']
   
    for i in tqdm(range(len(df))):
        
        encoded_sent = tz.encode_plus(
            text=df_sen[i],  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = max_length,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            truncation=True
        )
        # Get the input IDs and attention mask in tensor format
        input_ids.append(encoded_sent['input_ids'])
        
        
    return torch.tensor(input_ids)
        
    


logging.info("Replacing NaNs with empty strings")
df.replace(np.nan, '', inplace=True)

# df['pos'] = df['title_body_tags'].apply(nlp).apply(lambda doc :  " ". join([token.pos_ for token in doc]))

# df = pd.read_csv("modified.csv")
title_body_tags_max_length = 512 if "BERT" in model_type else 1700

if model_type == "BERTOverflow_Window":
    title_body_tags_max_length = 1700

title_body_tags_list = tokenize(df,title_body_tags_max_length)

logging.info("Done tokenizing dataset : {}".format(dataset_path))

torch.save(tz.vocab,export_path + prob + prefix+ '_vocab.v')
torch.save(title_body_tags_list,export_path + prob + prefix+'_title_body_tags.pt')

logging.info("Done saving vocab and dataset tensors")




