import data_loader as dl
import torch.nn.functional as F
from model import LSTM
from utils import save_hyperparameters
import torch
import torch.nn.functional as F
import stats
import os
import pytorch_lightning as pl
#import nn.Fu
import sys


#Need to add stat tracker
#def stop_early():
'''
Parameters: model, data_loader
Returns: 
Dimensions:
'''
def evaluate_model(model, data_loader, stat_tracker, epoch, prefix, batch_size):
    model.eval()
    test_stats = stats.AverageMeterSet()
    for i, batch in enumerate(data_loader):
        predictions = model(batch)
        distributions = F.log_softmax(predictions, dim=2)
        maxes = torch.argmax(distributions, 2)
        accuracy = torch.eq(maxes, batch).int()
        max_sequence_len = list(accuracy.size())[1]
        avg_sequence_accs = torch.sum(accuracy, 1)/max_sequence_len
        batch_accuracy = (torch.sum(avg_sequence_accs, 0)/batch_size).item()
        #print(batch_accuracy)
        #hi = avg_sequence_accs.size()[0]
        if avg_sequence_accs.size()[0] == batch_size:
            test_stats.update('accuracy', batch_accuracy, n=1)
    stat_tracker.record_stats(test_stats.averages(epoch, prefix=prefix))
    #print(test_stats.avgs['accuracy'])
    return test_stats.avgs['accuracy']

def train_model(model, data_loader, validation_dl, learning_rate, num_epochs, stat_tracker, batch_size, dec_acc_epoch_cnt_stop_threshold):
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_decreasing_acc_epochs = 0
    max_accuracy = 0
    for epoch in range(num_epochs):
        epoch_stats = stats.AverageMeterSet()
        model.train()
        for i, batch in enumerate(data_loader):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2 Run our forward pass.
            predictions = model.forward(batch)
            predictions = predictions.permute(0, 2, 1)

            # Step 3. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()\
            loss = loss_function(predictions, batch)
            #print(loss)
            loss.backward()
            optimizer.step()

            #epoch_stats.update('mean_rewards', loss, n=1)
            epoch_stats.update('loss', loss, n=1)
        stat_tracker.record_stats(epoch_stats.averages(epoch, prefix=str('/train/')))
        valid_acc_after_train_epoch = evaluate_model(model, validation_dl, stat_tracker, epoch, "/validation/", batch_size)
        print(num_decreasing_acc_epochs)
        print("max_accuracy: " + str(max_accuracy))
        print("valid_acc_after_train_epoch: " + str(valid_acc_after_train_epoch))
        if max_accuracy > valid_acc_after_train_epoch:
            num_decreasing_acc_epochs += 1
        else:
            max_accuracy = valid_acc_after_train_epoch
            num_decreasing_acc_epochs = 0


        if num_decreasing_acc_epochs > dec_acc_epoch_cnt_stop_threshold:
            return model

def main():
    hyp_params = {}
    hyp_params["training_data_path"] = sys.argv[1]
    hyp_params["validation_data_path"] = sys.argv[2]
    hyp_params["test_data_path"] = sys.argv[3]
    hyp_params["output_dir"] = sys.argv[4]
    hyp_params["vocab_size"] = int(sys.argv[5])
    hyp_params["batch_size"] = int(sys.argv[6])
    hyp_params["embedding_dim"] = int(sys.argv[7])
    hyp_params["learning_rate"] = float(sys.argv[8])
    hyp_params["num_epochs"] = int(sys.argv[9])
    hyp_params["hidden_dim"] = int(sys.argv[10])
    hyp_params["dec_acc_epoch_cnt_stop_threshold"] = int(sys.argv[11])

    ##########temporary################
    #hyp_params["training_data_path"] = "../../data/model-sets/toy_train.txt"
    #hyp_params["validation_data_path"] = "../../data/model-sets/toy_validation.txt"
    #hyp_params["test_data_path"] = "../../data/model-sets/toy_test.txt"
    #hyp_params["output_dir"] = "../Output/model-output/"
    #hyp_params["vocab_size"] = 30
    #hyp_params["batch_size"] = 20
    #hyp_params["embedding_dim"] = 25
    #hyp_params["learning_rate"] = .0001
    #hyp_params["num_epochs"] = 1000
    #hyp_params["hidden_dim"] = 10
    #hyp_params["dec_acc_epoch_cnt_stop_threshold"] = 30
    ##########TEMPORARY###############

    train_dl, validation_dl, test_dl = dl.create_dataloaders(hyp_params["training_data_path"],
                                                             hyp_params["validation_data_path"],
                                                             hyp_params["test_data_path"],
                                                             hyp_params["vocab_size"],
                                                             hyp_params["batch_size"])
    model = LSTM(hyp_params["vocab_size"], hyp_params["batch_size"],
                 hyp_params["embedding_dim"], hyp_params["hidden_dim"], hyp_params["output_dir"])
    stat_tracker = stats.StatTracker(log_dir=os.path.join(hyp_params["output_dir"], "tensorboard-log"))
    train_model(model, train_dl, validation_dl, hyp_params["learning_rate"], hyp_params["num_epochs"], stat_tracker,
                hyp_params["batch_size"], hyp_params["dec_acc_epoch_cnt_stop_threshold"])
    torch.save(model, hyp_params["output_dir"] + "model")
    save_hyperparameters(hyp_params, hyp_params["output_dir"])
    test_accuracy = evaluate_model(model, test_dl, stat_tracker, 1, "/test/", hyp_params["batch_size"])
    print("test_accuracy" + str(test_accuracy))

if __name__=="__main__":
    main()
