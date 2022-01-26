'''
Calculates the surprisal values for words in a list of words.
'''
import os
import torch
import torch.nn.functional as F
import sys
import argparse
import math
from data_loader import Dataset
from torch.utils.data import DataLoader
from collections import defaultdict
import utils
import os
import csv

'''
Gets arguments from the command-line.

Returns:
    params: a dictionary of command-line arguments
'''
def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="all_child_directed_data_vocab_size_5000.pkl")
    parser.add_argument("--language", default="eng")
    parser.add_argument("--gpu_run", action="store_true")
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--aoa_word_list", default="word_list_english_(american)_clean.csv")
    parser.add_argument("--experiment_dir", default="../../Results/experiments/")
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--split", default="all_child_directed_data")
    params = parser.parse_args()
    return params

def get_batched_surprisal_perplexity(model, dataloader, device):
    model.eval()
    surprisals_perplexity_n = defaultdict(lambda : [sys.float_info.epsilon, sys.float_info.epsilon, 0])
    for n, batch in enumerate(dataloader):
        if n % 100 == 0:
            print(n)
        batch = batch.to(device)
        outputs = model(batch)
        surprisals = -F.log_softmax(outputs, dim=2)
        indexes = batch.unsqueeze(-1)
        results = torch.gather(surprisals, -1, indexes)
        results = torch.reshape(results, (-1,))
        labels = torch.reshape(batch, (-1,))
        for word_id, surprisal in zip(labels, results):
            id = word_id.item()
            surp = surprisal.item()
            if id != 0:
                perplexity = 1/math.exp(-surp)
                surp_id = surprisals_perplexity_n[id]
                surp_id[0] += surp
                surp_id[1] += perplexity
                surp_id[2] += 1
    return(surprisals_perplexity_n)

def main():
    params = get_parameters()
    device = torch.device('cuda') if params.gpu_run == True else torch.device('cpu')
    vocabulary = utils.open_pkl(os.path.join("../../Data/model_datasets",params.language, "encoding_dictionary_vocab_size_5000.pkl"))
    model = torch.load(os.path.join(params.experiment_dir, params.model))
    model = model.to(device)
    data = Dataset(os.path.join("../../Data/model_datasets",params.language,params.data_path))
    dataloader = DataLoader(data, batch_size=params.batch_size)

    word_list = utils.open_word_list_csv(os.path.join("../../Data/word-lists/",params.language, params.aoa_word_list)
    surprisals_perplexity_n = get_batched_surprisal_perplexity(model, dataloader, device)
    file_name = params.aoa_word_list + "_" + params.split + "_singleword_average_surprisal_perplexity.csv"
    with open(os.path.join(params.experiment_dir, file_name), mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["word_clean", "n_tokens", "avg_surprisal", "avg_perplexity", "n_instances", "language"])
        for word in word_list:
            language = word['language']
            w = word["word_clean"]
            if w in vocabulary:
                id = vocabulary[w]
                if id != 0:
                    sum_surp, sum_perp, n = surprisals_perplexity_n[id]
                    if n != 0 :
                        avg_surp = sum_surp/n
                        avg_perp = sum_perp/n
                        writer.writerow([w, "1", f"{avg_surp:.16f}" , f"{avg_perp:.16f}", str(n), language])
                    else :
                        writer.writerow([w, 1, 'NA', 'NA', 0, language])

if __name__=="__main__":
    main()
