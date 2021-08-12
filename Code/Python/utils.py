import os

'''
hyperparameters: a dictionary containing the relevant hyperparameters."
'''
def save_hyperparameters(hyperparameters, output_dir):
    with open(os.path.join(output_dir, "hyper_parameters"), "w") as f:
        for hp in hyperparameters.keys():
            print(hp +": "+str(hyperparameters[hp])+"\n")
            f.write(hp +": "+str(hyperparameters[hp])+"\n")
