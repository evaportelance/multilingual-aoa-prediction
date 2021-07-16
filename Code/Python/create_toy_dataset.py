import random

MODEL_SETS_DIR = "../../Data/model-sets"


#Open text files
#Get random lines
#Save random lines to a new file
#Path to directory where the final test, train, and validation sets will be stored

train_set = []
validation_set = []
test_set = []
with open(MODEL_SETS_DIR + '/train.txt','r') as f :
    for line in f:
       train_set.append(line)
with open(MODEL_SETS_DIR + '/validation.txt','r') as f :
    for line in f:
       validation_set.append(line)
with open(MODEL_SETS_DIR + '/test.txt','r') as f :
    for line in f:
       test_set.append(line)

