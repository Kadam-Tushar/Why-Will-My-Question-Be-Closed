import pandas as pd 
import numpy as np
import pandas as pd 
import numpy as np
import torch
import csv
import numpy as np
import pandas as pd
import re
import nltk
import spacy
import string
import re
from bs4 import BeautifulSoup
import contractions
import re
import unicodedata
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from transformers import BertTokenizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # For a nice progress bar!
from torch.utils.data import Dataset, DataLoader

from os import listdir
from os.path import isfile, join


sampled_dataset = 'sample_0.02_33.csv'
path = '/Users/tushar/iisc/sem3/TSE/project/Why-Will-My-Question-Be-Closed/Dataset/' 
df = pd.read_csv(path+sampled_dataset,dtype={'OwnerUserId': 'float64'})

print("Read sampled dataset ", sampled_dataset)

# Keeping only Questions i.e removing Answers, wiki, comments etc
df = df[ df['PostTypeId'] == 1]  

# Making new column to represent if question is closed or not 
df['closed'] = df['ClosedDate'].notnull().astype(int)

# converting to date format to datetime
df['ClosedDate'] = pd.to_datetime( df['ClosedDate'])
df['CreationDate'] = pd.to_datetime( df['CreationDate'])

# Filtering datasets
startDate = '07-01-2013'
df = df[df['CreationDate'] >= startDate ] 


# Getting reasons of closed questions, separating closed and open questions
closed_reasons = pd.read_csv('/Users/tushar/iisc/sem3/TSE/project/ClosedPosts-Type/closed_reasons.csv')
closed_questions =df[ df['closed'] == 1 ]
open_questions = df[ df['closed'] == 0 ]

print("separated dataset of open closed questions")

# Adding new column of comment in open questinos
open_questions['comment'] = 0 




# Inner joining joining dataset with closed question dataset to obtain reason for closing 
merged_closed_questions = pd.merge(closed_questions,closed_reasons,left_on = 'Id',right_on = 'id')


# Removing duplicate closed questions 
merged_closed_questions = merged_closed_questions[~(merged_closed_questions['comment'] == 101.0)]


# Replaing values with proper class labels 
merged_closed_questions['comment'] = merged_closed_questions['comment'].replace(np.nan,0)
merged_closed_questions['comment'] = merged_closed_questions['comment'].replace(102.0,1)
merged_closed_questions['comment'] = merged_closed_questions['comment'].replace(103.0,2)
merged_closed_questions['comment'] = merged_closed_questions['comment'].replace(104.0,3)
merged_closed_questions['comment'] = merged_closed_questions['comment'].replace(105.0,4)


# converting to ints
merged_closed_questions['comment'] = merged_closed_questions['comment'].astype(int)


# Removing nans 
merged_closed_questions = merged_closed_questions.replace(np.nan,"") 
open_questions = open_questions.replace(np.nan,"")

# Preparing for same columns for to concat open and closed questions
closed_questions = merged_closed_questions[open_questions.columns]

# concating open and closed questions 

final_df = pd.concat([open_questions, closed_questions])

print("Combined open and closed questions ")
print("Length of final_df",len(final_df))

print("Distibution of classes:")
print(final_df['comment'].value_counts(normalize=True))


print("Distibution of binary classes:")
print(final_df['closed'].value_counts())





final_df['title_body'] = final_df['Title'] + final_df['Body']
final_df.to_csv(path + 'fixed_final_df.csv',index=False)
print("Saved final dataset!")
title_body_fixed = final_df[['Id','Title','Body','Tags','closed','comment','title_body']]
title_body_fixed.to_csv(path+"fixed_title_body.csv",index= False)
print("Saved fixed!")






