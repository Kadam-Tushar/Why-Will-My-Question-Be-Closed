import pandas as pd 
import numpy as np
import torch
import csv

import re
import string
import re

from bs4 import BeautifulSoup
import contractions
import re
import unicodedata


from transformers import BertTokenizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # For a nice progress bar!
from torch.utils.data import Dataset, DataLoader


# Os related dependencies
import os
from os import listdir
from os.path import isfile, join

#Logging 
import logging

# Date and time related modules
import time
import datetime
import random

import pickle

# Global variables
path_sep = os.path.sep 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Hyperparameters
input_size = 256
hidden_size = 256
num_layers = 1

num_classes = 2
col = "closed" if num_classes == 2 else "comment"
prob = "bin" if num_classes == 2 else "multi"
model_type = "GRU"
model_name = "_UNI_Bin.model" if num_classes == 2 else "_UNI_Multi.model"
model_name = model_type + model_name
sequence_length = 512 if "BERT" in model_type  else 1700
if model_type == "BERTOverflow_Window":
    sequence_length = 1700

learning_rate = 0.001
batch_size = 64
num_epochs = 15
prefix = ""

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Set the seed value all over the place to make this reproducible.
seed_val = 321
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


#model params
model_path = '/scratch/tusharpk/trained_models' + path_sep

#paths 
dataset_name = 'balanced_'+prob +'.csv'
dataset_path = '/scratch/tusharpk/Dataset' + path_sep + dataset_name
export_path = '/scratch/tusharpk/Dataset' + path_sep 
title_body_tags_list_path = '/scratch/tusharpk/Dataset' + path_sep + prob + "_title_body_tags.pt"



# Global functions 
def good_update_interval(total_iters, num_desired_updates):
    '''
    This function will try to pick an intelligent progress update interval 
    based on the magnitude of the total iterations.

    Parameters:
      `total_iters` - The number of iterations in the for-loop.
      `num_desired_updates` - How many times we want to see an update over the 
                              course of the for-loop.
    '''
    # Divide the total iterations by the desired number of updates. Most likely
    # this will be some ugly number.
    exact_interval = total_iters / num_desired_updates

    # The `round` function has the ability to round down a number to, e.g., the
    # nearest thousandth: round(exact_interval, -3)
    #
    # To determine the magnitude to round to, find the magnitude of the total,
    # and then go one magnitude below that.

    # Get the order of magnitude of the total.
    order_of_mag = len(str(total_iters)) - 1

    # Our update interval should be rounded to an order of magnitude smaller. 
    round_mag = order_of_mag - 1

    # Round down and cast to an int.
    update_interval = int(round(exact_interval, -round_mag))

    # Don't allow the interval to be zero!
    if update_interval == 0:
        update_interval = 1

    return update_interval

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
