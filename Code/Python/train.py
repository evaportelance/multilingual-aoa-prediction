import data_loader as dl
from model import LSTM
import torch
import sys

##########temporary################
vocab_size = 10 #actual 5000
batch_size = 30
training_data_path = "../../data/model-sets/toy_train.txt"
validation_data_path = "../../data/model-sets/toy_validation.txt"
test_data_path = "../../data/model-sets/toy_test.txt"
embedding_dim = 30 #hyperparameter
learning_rate = .0001
num_epochs = 300
##########TEMPORARY###############

def train_model(model, data_loader):
    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam([var1, var2], lr=learning_rate)

    for epoch in range(num_epochs):
        for batch in enumerate(data_loader):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            ## Step 2. Get our inputs ready for the network, that is, turn them into
            ## Tensors of word indices.
            #sentence_in = prepare_sequence(sentence, word_to_ix)
            #targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            scores = model(batch)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(scores)
            loss.backward()
            optimizer.step()


def main():
    #vocab_size = sys.argv[1]
    #batch_size = sys.argv[2]
    #training_data_path = sys.argv[3]
    #validation_data_path = sys.argv[4]
    #test_data_path = sys.argv[5]
    #embedding_dim = sys.argv[6]

    train_dl, validation_dl, test_dl = dl.create_dataloaders(training_data_path, validation_data_path, test_data_path,
                                                             vocab_size, batch_size)
    model = LSTM(vocab_size, batch_size, embedding_dim)
    train_model(model, train_dl)

if __name__=="__main__":
    main()
