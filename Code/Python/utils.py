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
        csv_contents = csv.DictReader(csv_file, delimiter=',', quotechar='|')
        for row in csv_contents:
            word_list.append(row)
        return word_list


def save_surprisals_as_csv(surprisals, experiment_dir, file_name):
    with open(os.path.join(experiment_dir, file_name), mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["word_clean", "n_tokens", "avg_surprisal", "avg_perplexity", "n_instances", "language"])
        for word in surprisals:
            n_tokens, suprisal_sum, perplexity_sum, n, language = surprisals[word]
            if n == 0:
                writer.writerow([word, n_tokens, 'NA', 'NA', 'NA', language])
            else:
                avg_surprisal = suprisal_sum/n
                avg_perplexity = perplexity_sum/n
                writer.writerow([word, str(n_tokens), f"{avg_surprisal:.16f}" , f"{avg_perplexity:.16f}", str(n), language])


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
