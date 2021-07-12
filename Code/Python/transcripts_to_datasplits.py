'''
This script iterates over a directory containing CHILDES transcripts in .capp format and creates a
training set of child-directed utterances, a test set of child utterances, and
a validation set of child-directed utterances.
'''
import os
import re 
import random

#The ISO 639-3 code for the desired language or the name of a custom language directory
LANG_NAME = "eng" 

#Path to directory containing all the transcript data to be included
TRANSCRIPT_DIR = "../../Data/transcripts/" + LANG_NAME

#Path to directory where the final test, train, and validation sets will be stored
MODEL_SETS_DIR = "../../Data/model-sets"

#Proportion of child-directed utterances in the training set
TRAIN_CDU_PROP = .80 

#Proportion of child-directed utterances in the validation set
VALIDATION_CDU_PROP = 1 - TRAIN_CDU_PROP 


'''
    Uses a biased (VALIDATION_CDU_PROP) coin flip to determine whether a child-directed utterance will
    be part of the training or validation set.
    
    Parameters: NA

    Returns:
        True if a generated random float in the range [0, 1.0] is less than
        VALIDATION_CDU_PROP and False otherwise
'''

def is_validation_sent():
    return True if random.random() < VALIDATION_CDU_PROP else False


'''
    Iterates over all of the transcripts in TRANSCRIPT_DIR and assigns and
    adds each line to either train, validation, or test set lists. Removes the speaker code (e.g. "*CHI:") 
    when the line is added to appropriate list. 

    Parameters: NA

    Returns:
        splits: a 3-tuple containing the model datasets in the form (test, validation, validation)
'''
def create_splits_from_transcripts():
    test = []
    validation = [] 
    train = []
    for subdir, dirs, files in os.walk(TRANSCRIPT_DIR):
        for file in files:
            if ('.capp' in file):
                textfile = subdir+'/'+file
                with open(textfile,'r') as f :
                    lines = f.readlines()
                for sent in lines :
                    if '*CHI:' in sent :
                        sent = re.sub('\*[A-Z]+: ', '', sent)
                        test.append(sent)
                    else :
                        sent = re.sub('\*[A-Z]+: ', '', sent)
                        if is_validation_sent():
                            validation.append(sent)
                        else:
                            train.append(sent)
    return {"test": test, "validation":validation,
                                    "train":train} 

'''
    Writes each line in the train, validation, and test lists sequentially to a
    .txt file to get the final model splits.

    Parameters: 
        splits: a 3-tuple containing the model datasets in the form (test, validation, validation)

    Returns: NA
'''
def write_splits_to_disk(splits):
    #save train, validation, and test split in case we need to rerun model
    with open(MODEL_SETS_DIR + '/test.txt','w') as f :
        for line in splits["test"]:
            f.write(line)
    with open(MODEL_SETS_DIR + '/validation.txt','w') as f :
        for line in splits["validation"]:
            f.write(line)
    with open(MODEL_SETS_DIR + '/train.txt','w') as f :
        for line in splits["train"]:
            f.write(line)

if __name__ == "__main__":
    splits = create_splits_from_transcripts()
    write_splits_to_disk(splits)
