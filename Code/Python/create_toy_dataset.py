import random

MODEL_SETS_DIR = "../../Data/model-sets"
TRAIN_NUM_LINES = 100
VALIDATION_NUM_LINES = 30
TEST_NUM_LINES = 10

#Open text files
#Get random lines
#Save random lines to a new file
#Path to directory where the final test, train, and validation sets will be stored

toy_train_set = []
toy_validation_set = []
toy_test_set = []
with open(MODEL_SETS_DIR + '/train.txt','r') as f :
    for i in range(TRAIN_NUM_LINES):
        line_num = random.randint(0, 1000)
        toy_train_set.append(f.readline(line_num))

with open(MODEL_SETS_DIR + "/toy_train.txt", 'w') as f:
    for line in toy_train_set:
        f.write(line)

with open(MODEL_SETS_DIR + '/validation.txt','r') as f :
    for i in range(VALIDATION_NUM_LINES):
        line_num = random.randint(0, 1000)
        toy_validation_set.append(f.readline(line_num))

with open(MODEL_SETS_DIR + "/toy_validation.txt", 'w') as f:
    for line in toy_validation_set:
        f.write(line)

with open(MODEL_SETS_DIR + '/test.txt','r') as f :
    for i in range(TEST_NUM_LINES):
        line_num = random.randint(0, 1000)
        toy_test_set.append(f.readline(line_num))

with open(MODEL_SETS_DIR + "/toy_test.txt", 'w') as f:
    for line in toy_test_set:
        f.write(line)
