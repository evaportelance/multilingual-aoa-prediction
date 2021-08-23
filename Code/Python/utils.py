import pickle
import csv
import os

'''
    Saves a dictionary as a pickle file.

    Parameters:
        dir: a file path
        dict: a dictionary
'''
def save_pkl(dir, data, name):
    with open(os.path.join(dir, name), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

'''
    Opens a dictionary as a pickle file.

    Parameters:
        dir: a file path
        dict: a dictionary
'''
def open_pkl(dir):
    with open(dir, 'rb') as file:
        return pickle.load(file)


'''
    Opens a csv file.
'''

def open_word_list_csv(csv_path):
    with open(csv_path, newline='\n') as csv_file:
        word_list = []
        csv_contents = csv.reader(csv_file, delimiter=' ', quotechar='|')
        for row in csv_contents:
            word_list.append(row[0].split('\t')[0])
        word_list.remove(word_list[0])
        return word_list


def save_surprisals_as_csv(surprisals, experiment_dir, file_name):
    with open(os.path.join(experiment_dir, file_name), mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["word", "surprisal_value", "n_instances"])
        for word in surprisals:
            _sum, n = surprisals[word]
            if n == 0:
                writer.writerow([word, 'NA', 'NA'])
            else:
                avg = _sum/n
                writer.writerow([word, f"{avg:.16f}" , str(n)])


'''
    Opens a text file and creates a list whose elements are lists that
    correspond to file lines. The elements of each file line list are
    the words in the line.
    Parameters:
        file_path: a relative path to a text file
    Returns:
        file_list: a list of lines in the file at file_path
'''
def open_txt(file_path):
    lines = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip('\n')
            line_as_list = line.split(" ")
            lines.append(line_as_list)
    return lines
