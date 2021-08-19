'''
This script iterates over a directory containing CHILDES transcripts in .capp format and creates a
training set of child-directed utterances, a test set of child utterances, and
a validation set of child-directed utterances.
'''
import os
import re
import random
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_name', default='eng',
                        help='The ISO 639-3 code for the desired language or the name of a custom language directory.')
    parser.add_argument('--transcript_dir', default="../../Data/transcripts/",
                        help='Path to directory containing all the transcript data to be included')
    parser.add_argument('--result_dir', default='../../Data/model_datasets/',
                        help='Path to directory where the final test, train, and validation sets will be stored')
    parser.add_argument('--train_prop', type=float, default=0.8, help='Proportion of child-directed utterances in the training set')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    return args


'''
    Uses a biased (prop) coin flip to determine whether a child-directed utterance will
    be part of the training or validation set.

    Parameters: NA

    Returns:
        True if a generated random float in the range [0, 1.0] is more than
        train prop and False otherwise
'''

def is_validation_sent(prop):
    return True if random.random() > prop else False


'''
    Iterates over all of the transcripts in transcript_dir and assigns and
    adds each line to either train, validation, or test set lists. Removes the speaker code (e.g. "*CHI:")
    when the line is added to appropriate list.

    Parameters: NA

    Returns:
        splits: a 3-tuple containing the model datasets in the form (test, validation, train)
'''
def create_splits_from_transcripts(transcript_dir):
    test = []
    validation = []
    train = []
    for subdir, dirs, files in os.walk(transcript_dir):
        for file in files:
            if ('.capp' in file):
                textfile = subdir + '/' + file
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
def write_splits_to_disk(splits, result_dir):
    with open(os.join.path(result_dir, 'test_child_data.txt'),'w') as f :
        for line in splits["test"]:
            f.write(line)
    with open(os.join.path(result_dir, 'validation.txt'),'w') as f :
        for line in splits["validation"]:
            f.write(line)
    with open(os.join.path(result_dir, 'train.txt'),'w') as f :
        for line in splits["train"]:
            f.write(line)
    with open(os.join.path(result_dir, 'all_child_directed_data.txt'),'w') as f :
        for line in splits["validation"] + splits["train"]:
            f.write(line)

if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)
    
    transcript_dir = os.path.join(args.transcript_dir, args.lang_name)
    splits = create_splits_from_transcripts(transcript_dir)
    result_dir = os.path.join(args.results_dir, args.lang_name)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    write_splits_to_disk(splits, result_dir)
