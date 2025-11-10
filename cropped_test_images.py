# IMPORTS 
# Libraries
import os
import pandas as pd

# Modules
from utils.utils import *

# VARIABLES
# Configuration
# filepaths for csv data
train_path = os.path.relpath('./dataset_v2/train.csv')
test_path = os.path.relpath('./dataset_v2/test.csv')

# csv dataset
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Global Variables

# FUNCTIONS
# Remove unwanted characters within labels
def edit_labels():
    train.replace(' ', '_', regex=True,inplace=True)
    train.replace('/', '_', regex=True,inplace=True)
    test.replace(' ', '_', regex=True,inplace=True)
    test.replace('/', '_', regex=True,inplace=True)

def main():
    ...

if __name__ == '__main__':
    main()