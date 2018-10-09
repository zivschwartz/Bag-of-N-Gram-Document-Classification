#Loading the data
import pandas as pd
import numpy as np
import random
from os import listdir

#Load single .txt file
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
 
#Load all .txt files in the directory
def process_docs(directory, data):
    i = 0
    targetdata = []
    for target in ('pos', 'neg'):
        for filename in listdir(directory + target):
            #Attributing the correct sentiment labels
            if target == 'pos':
                target_int = 1
            else:
                target_int = 0
            #Including only relevant .txt files
            if not filename.endswith(".txt"):
                continue
            #Build the full path of the file to open
            path = directory + target + '/' + filename
            #Load document and label target
            data.append(load_doc(path))
            i += 1
            targetdata.append(target_int)
    return data, targetdata

#Paths to training and testing directories
train_dir = '/Users/zivschwartz1/IMDB/train/'
test_dir = '/Users/zivschwartz1/IMDB/test/'

train = []
test = []

#Create full train/test lists and their target lists
full_train, full_train_target = process_docs(train_dir, train)
test, test_target = process_docs(test_dir, test)
