import argparse
import os
import torch
import torch.nn.functional as F
from bert_custom_dataset import CHILDESDataset
import operator
import functools
from torch.utils.data import DataLoader

def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../../Data/model_datasets/eng/validation.txt")
    parser.add_argument("--gpu_run", action="store_true")
    parser.add_argument("--batch_size", default=500, type=int)
    parser.add_argument("--aoa_word_list", default="../../Data/model_datasets/eng/aoa_words.csv")
    parser.add_argument("--experiment_dir", default="../../Results/experiments/")
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--split", default="validation")
    params = parser.parse_args()
    return params


def make_token_word_pairs(aoa_word_list, dataset):
    tokenizer = dataset.tokenizer
    word_pairs = []
    for word in aoa_word_list:
        seq = tokenizer(word)['input_ids']
        token = torch.Tensor(seq[1:-1]).long()
        word_pairs.append((token, word))
    return word_pairs

def indexes_in_sequence(query, base):
    id_, label = base[0], base[1]
    label = label.squeeze()
    l = len(query)
    locations = []
    for i in range((len(label)-l)):
        if torch.all(label[i:i+l] == query):
            locations.append([id_, i])            
    return locations

def get_batched_surprisals(model, dataloader, word_pairs, device):
    model.eval()
    word_surprisals = {}
    for index, word in word_pairs:
        word_surprisals[word] = [0.0, 0]
    batch_size = dataloader.batch_size
    for n, batch in enumerate(dataloader):
        print(n)
        for key in batch:
            batch[key] = batch[key].to(device)
        outputs = model(**batch)
        surprisals = -F.log_softmax(outputs.logits, -1)
        labels = batch['labels']
        labels_split = torch.tensor_split(labels, batch_size)
        for indexes, word in word_pairs:
            indexes = indexes.to(device)
            print(indexes)
            match_list = list(map(lambda x: indexes_in_sequence(indexes, x), enumerate(labels_split)))
            index_matches = functools.reduce(operator.iconcat, match_list)
            if len(index_matches) > 0:
                for i in index_matches:
                    surprisal = 1.0
                    for j, index in enumerate(indexes):
                        id_ = i
                        id_[1] += j
                        match = surprisals[tuple(id_)]
                        sub_surprisal = match[index].item()
                        surprisal *= sub_surprisal
                    word_surprisals[word][0] += (surprisal + sys.float_info.epsilon)
                    word_surprisals[word][1] += 1
    return word_surprisals

def main():
    params = get_parameters()
    device = torch.device('cuda') if params.gpu_run == True else torch.device('cpu')
    model = torch.load(os.path.join(params.experiment_dir, params.model))
    model = model.to(device)
    data = Dataset(params.data_path)
    dataloader = DataLoader(data, batch_size=params.batch_size)
    word_list = utils.open_word_list_csv(params.aoa_word_list)
    word_pairs = make_token_word_pairs(word_list, data)
    word_surprisals = get_batched_surprisals(model, dataloader, word_pairs, device)
    file_name = params.split + "_average_surprisals.csv"
    utils.save_surprisals_as_csv(word_surprisals, params.experiment_dir, file_name)

if __name__=="__main__":
    main()
