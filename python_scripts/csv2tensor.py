from modules import * 
import CustomeTextDataset

dataset_name = 'fixed_final_df.csv'
dataset_path = '..' + path_sep + 'Dataset' + path_sep + dataset_name
export_path = '..' + path_sep + 'Dataset' + path_sep 
df = pd.read_csv(dataset_path)

logging.info("Done reading dataset csv file : {}".format(dataset_path))

# Encode the sentence
tz = BertTokenizer.from_pretrained("bert-base-cased")
def tokenize(df,max_length):
    input_ids = []
    for sent in tqdm(df):
        encoded = tz.encode(
            text=sent,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = max_length,  # maximum length of a sentence
            padding=False,  #  Do not Add [PAD]s for dynamic batching
            truncation=True
        )
        # Get the input IDs and attention mask in tensor format
        input_ids.append(encoded)
        
        
    return input_ids
        
logging.info("Replacing NaNs with empty strings")
df.replace(np.nan, '', inplace=True)

title_body_max_length = 1700
title_body_list = tokenize(df['title_body'],title_body_max_length)

logging.info("Done tokenizing dataset : {}".format(dataset_path))

torch.save(tz.vocab,export_path + 'vocab.v')
torch.save(title_body_list,export_path + 'title_body.pt')

logging.info("Done saving vocab and dataset tensors")




