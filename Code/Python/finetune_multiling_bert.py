import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import BertLMHeadModel, AdamW
from stats import AverageMeterSet, StatTracker
from bert_custom_dataset import CHILDESDataset
from utils import save_pkl


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='../../Data/model_datasets/eng/train.txt',
                        help='file path for training data')
    parser.add_argument('--val_path', default='../../Data/model_datasets/eng/validation.txt',
                        help='file path for validation data')
    parser.add_argument('--result_dir', default='../../Results/experiments/',
                        help='directory where model and log files will be saved')
    parser.add_argument('--experiment_name', default='test', help='name of the experiment directory where model and log files will be stored')
    parser.add_argument('--lr', default='5e-5', type=float,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    return args


## Evaluation function for getting accuracy using argmax of token predictions
def test_finetuned_model(model, dataloader, device, stat_tracker, epoch=1, prefix='test'):
    model.eval()
    test_stats = AverageMeterSet()
    batch_size = dataloader.batch_size
    for batch in dataloader:
        for key in batch:
            batch[key] = batch[key].to(device)
        labels = batch['labels']
        outputs = model(**batch)
        predictions = F.log_softmax(outputs.logits, -1)
        n_matches = torch.eq(torch.argmax(predictions, dim = -1), labels).int()
        max_sequence_len = list(n_matches.size())[1]
        avg_sequence_accs = torch.sum(n_matches, 1)/max_sequence_len
        batch_accuracy = float((torch.sum(avg_sequence_accs, 0)/avg_sequence_accs.size()[0]).item())
        test_stats.update('accuracy', batch_accuracy, n=1)
    stat_tracker.record_stats(test_stats.averages(epoch, prefix=prefix))

    return test_stats.avgs['accuracy']


## Train function for finetuning multilingual BERT
def finetune_model(model, train_dataloader, val_dataloader, device, stat_tracker, n_epochs=5, lr=5e-5):
    optimizer = AdamW(model.parameters(), lr)
    for epoch in range(n_epochs):
        model.train()
        epoch_stats = AverageMeterSet()
        for step,batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            for key in batch:
                batch[key] = batch[key].to(device)
            outputs = model(**batch)
            ##get train accuracy
            predictions = F.log_softmax(outputs.logits, -1)
            n_matches = torch.eq(torch.argmax(predictions, dim = -1), batch['labels']).int()
            max_sequence_len = list(n_matches.size())[1]
            avg_sequence_accs = torch.sum(n_matches, 1)/max_sequence_len
            batch_accuracy = float((torch.sum(avg_sequence_accs, 0)/avg_sequence_accs.size()[0]).item())
            epoch_stats.update('accuracy', batch_accuracy, n=1)
            ## get loss
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            epoch_stats.update('loss', float(loss), n=1)
        val_accuracy = test_finetuned_model(model, val_dataloader, device, stat_tracker, epoch, prefix="val")
        stat_tracker.record_stats(epoch_stats.averages(epoch, prefix="train"))
        print(str(epoch))
        print("train acc: "+ str(epoch_stats.avgs['accuracy']))
        print("val acc: "+ str(val_accuracy))
        print("loss :"+ str(epoch_stats.avgs['loss']))
        
    return model

def main():
    args = get_args()
    experiment_dir = os.path.join(args.result_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    save_pkl(experiment_dir, vars(args), "hyperparameters.pkl")
    torch.manual_seed(args.seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    model = BertLMHeadModel.from_pretrained("bert-base-multilingual-uncased", return_dict=True, is_decoder = True)
    model = model.to(device)

    train_dataset = CHILDESDataset(args.train_path)
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
    val_dataset = CHILDESDataset(args.val_path)
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=True)

    stat_tracker = StatTracker(log_dir=os.path.join(experiment_dir,"tensorboard-log"))

    model = finetune_model(model, train_dl, val_dl, device, stat_tracker, args.n_epochs, args.lr)

    torch.save(model, os.path.join(experiment_dir,"model.pt"))


if __name__=="__main__":
    main()
