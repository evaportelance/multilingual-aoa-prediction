import argparse
import os
import torch
import torch.nn.functional as F
from bert_custom_dataset import CHILDESDataset
import csv

def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_child_directed_data_path", default="../../data/model-sets/toy_datasets/toy_all.pkl")
    parser.add_argument("--aoa_word_list", default="../../data/model-sets/aoa_word_list.csv")
    parser.add_argument("--experiment_dir", default="../../results/experiments/2021-12-08T22-59-15/")
    parser.add_argument("--model", default="model.pt")
    params = parser.parse_args()
    return params


def make_word_dict(aoa_word_list, dataset):
    tokenizer = dataset.tokenizer
    word_dict = {}
    for word in aoa_word_list:
        token = tokenizer(word)[0]
        word_dict[token] = word
    return word_dict

## Evaluation function for getting accuracy using argmax of token predictions
def get_surprisals(model, dataset, word_dict, device):
    model.eval()
    word_surprisals = {}
    for index in word_dict.keys():
        surprisals[word_dict[index]] = [0.0, 0]
    for i in range(0, len(dataset)):
        item = dataset[i]
        for key in item:
            item[key] = item[key].to(device)
        labels = item['labels']
        outputs = model(**item)
        surprisals = -F.log_softmax(outputs.logits, -1)
        for word_index in word_dict:
            index_matches = (labels == word_index).nonzero(as_tuple=False)
            if len(index_matches) > 0:
                for i in index_matches:
                    match = i.item()
                    surprisal = (surprisals[0][match][word_index] + sys.float_info.epsilon).item()
                    word = word_dict[word_index]
                    word_surprisals[word][0] += surprisal
                    word_surprisals[word][1] += 1

    return word_surprisals

def main():
    params = get_parameters()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(os.join.path(params.experiment_dir, params.model))
    model = model.to(device)
    dataset = CHILDESDataset(params.all_child_directed_data_path)
    aoa_word_list = utils.open_word_list_csv(params.aoa_word_list)
    word_dict = make_word_dict(aoa_word_list, dataset)
    word_surprisals = get_surprisals(model, dataset, word_dict, device)
    with open(os.join.path(params.experiment_dir, "aoa_surprisals.csv"), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow("word", "avg_surprisal", "count")
        for word in word_surprisals:
            sum = word_surprisals[word][0]
            n = word_surprisals[word][0]
            avg = sum/n
            writer.writerow(word, avg, n)

if __name__=="__main__":
    main()
