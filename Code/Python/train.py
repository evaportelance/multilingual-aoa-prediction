import data_loader as dl
from model import LSTM
from utils import save_hyperparameters
import torch
import torch.nn.functional as F
import stats
import os
import argparse
import extract_surprisal_values as sup_vals

'''
Gets command-line arguments and specifies defaults values for arguments that aren't specified.

    Returns:
        params: a dictionary with all command line values
'''
def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("-training_data_path", default="../../Data/model-sets/toy_train.txt", action="store_true")
    parser.add_argument("-validation_data_path",default="../../Data/model-sets/toy_validation.txt")
    parser.add_argument("-test_data_path", default="../../Data/model-sets/toy_test.txt")
    parser.add_argument("-output_dir", default="../Output/model-output/")
    parser.add_argument("-vocab_size", default=100, type=int)
    parser.add_argument("-batch_size", default=30, type=int)
    parser.add_argument ("-embedding_dim", default=21, type=int)
    parser.add_argument("-hidden_dim", default=10, type=int)
    parser.add_argument("-device_type", default="cpu")
    parser.add_argument("-learning_rate", default=.0001, type=float)
    parser.add_argument("-num_epochs", default=1, type=int)
    parser.add_argument("-dec_acc_epoch_cnt_stop_threshold", default=10, type=int)
    params = vars(parser.parse_args()) #converts namespace to dictionary
    return params

'''
Calculates the model's accuracy on a given data set by predicting the next word given some input and comparing the
prediction to the data. The overall accuracy is calculated by averaging the model's accuracy for each batch.
Batch-level accuracy is the average of the accuracy for each utterance. An utterances accuracy is the percentage of 
words in an utterance that the model predicted correctly.

Parameters: 
    model: a pytorch.nn.lstm object
    data_loader: a pytorch dataloader that is used to feed data into the model
    stat_tracker: an object that stores important information about the model and feeds it to tensorboard
    epoch: the current epoch number in the training cycle if evaluate_model is run on the validation set or 1 if it is
            on the test set
    prefix: tells stat_tracker what data_loader is used
    batch_size: the number of utterances in one batch
    
Returns: 
   overall_accuracy: the average of all the batch accuracies 
'''
def evaluate_model(model, data_loader, stat_tracker, epoch, prefix, batch_size, device):
    model.eval()
    torch.no_grad()
    test_stats = stats.AverageMeterSet()
    for i, batch in enumerate(data_loader):
        batch = batch.to(device)
        predictions = model(batch)
        distributions = F.log_softmax(predictions, dim=2)
        maxes = torch.argmax(distributions, 2)
        accuracy = torch.eq(maxes, batch).int()
        max_sequence_len = list(accuracy.size())[1]
        avg_sequence_accs = torch.sum(accuracy, 1)/max_sequence_len
        batch_accuracy = (torch.sum(avg_sequence_accs, 0)/batch_size).item()
        if avg_sequence_accs.size()[0] == batch_size:
            test_stats.update('accuracy', batch_accuracy, n=1)
    print(epoch)
    stat_tracker.record_stats(test_stats.averages(epoch, prefix=prefix))
    overall_accuracy = test_stats.avgs['accuracy']
    return overall_accuracy

'''
Trains the lstm for the number of epochs specified or until the accuracy decreases for the number of epochs defined
in the stopping threshold.

Parameters:
   model: an untrained LSTM object defined in model.py
   data_loader: the dataloader for the train set
   validation_dl: the dataloader for the validation set
   learning_rate: the rate at which the parameters are updated in gradient descent
   num_epochs: the maximum number of epochs to train for
   stat_tracker: records the loss and accuracy
   batch_size: the number of utterances in a dataloader batch
   dec_acc_epoch_cnt_stop_threshold: the number of decreasing accuracy epochs to stop training after

Returns:
    model: the trained model
'''
def train_model(model, data_loader, validation_dl, learning_rate, num_epochs, stat_tracker, batch_size,
                dec_acc_epoch_cnt_stop_threshold, device):
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_decreasing_acc_epochs = 0
    max_accuracy = 0
    for epoch in range(num_epochs):
        epoch_stats = stats.AverageMeterSet()
        model.train()
        for i, batch in enumerate(data_loader):
            batch = batch.to(device)
            # Step 1. Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2 Run our forward pass.
            predictions = model.forward(batch)
            predictions = predictions.permute(0, 2, 1)

            # Step 3. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            print(i)
            loss = loss_function(predictions, batch)
            loss.backward()
            optimizer.step()

            epoch_stats.update('loss', loss, n=1)
        stat_tracker.record_stats(epoch_stats.averages(epoch, prefix=str('/train/')))

        #Handles early stopping
        valid_acc_after_train_epoch = evaluate_model(model, validation_dl, stat_tracker, epoch,
                                                     "/validation/", batch_size, device)
        if max_accuracy > valid_acc_after_train_epoch:
            num_decreasing_acc_epochs += 1
        else:
            max_accuracy = valid_acc_after_train_epoch
            num_decreasing_acc_epochs = 0
        if num_decreasing_acc_epochs > dec_acc_epoch_cnt_stop_threshold:
            return model

'''
Takes in command line arguments specifying the parameters of the lstm to be trained, trains the model, saves the model
and hyperparameters to disk and evaluates the models accuracy on the test set.
'''
def main():
    params = get_parameters()
    train_dl, validation_dl, test_dl, vocabulary, all_data = dl.create_dataloaders(params["training_data_path"],
                                                             params["validation_data_path"],
                                                             params["test_data_path"],
                                                             params["vocab_size"],
                                                             params["batch_size"])
    model = LSTM(params["vocab_size"], params["batch_size"],
                 params["embedding_dim"], params["hidden_dim"])
    if params["device_type"] == "cuda":
        model.cuda()
    stat_tracker = stats.StatTracker(log_dir=os.path.join(params["output_dir"], "tensorboard-log"))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_model(model, train_dl, validation_dl, params["learning_rate"], params["num_epochs"], stat_tracker,
                params["batch_size"], params["dec_acc_epoch_cnt_stop_threshold"], device)
    surprisal_values = sup_vals.find_surprisal_values(vocabulary, model, all_data)
    torch.save(model, params["output_dir"] + "model")
    save_hyperparameters(params, params["output_dir"])
    test_accuracy = evaluate_model(model, test_dl, stat_tracker, 1, "/test/", params["batch_size"], device)
    print("test_accuracy" + str(test_accuracy))

if __name__=="__main__":
    main()
