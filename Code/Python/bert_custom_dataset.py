import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

## Custom Dataset class for CHILDES utterances
class CHILDESDataset(Dataset):
    def __init__(self, file_path):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        self.sentences = []
        with open(file_path, "r") as f:
            for line in f:
                text = line.strip()
                self.sentences.append(text)
        encoded_data = self.tokenizer(self.sentences, return_tensors='pt', padding=True, truncation=True, max_length=100)

        self.input_ids = encoded_data['input_ids']
        self.token_type_ids = encoded_data['token_type_ids']
        self.attention_mask = encoded_data['attention_mask']

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        batch_dict = {'input_ids': self.input_ids[index],
                      'token_type_ids': self.token_type_ids[index],
                      'attention_mask': self.attention_mask[index],
                      'labels': self.input_ids[index]}

        return batch_dict
