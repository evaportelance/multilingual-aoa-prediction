import pickle

'''
    Saves a dictionary as a pickle file.
    
    Parameters:
        dir: a file path
        dict: a dictionary
'''
def save_pkl(dir, file, name):
    with open(dir + name, 'wb') as f:
        pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)

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
    Opens a text file and creates a list whose elements are lists that 
    correspond to file lines. The elements of each file line list are
    the words in the line.
    Parameters:
        file_path: a relative path to a text file
    Returns:
        file_list: a list of lines in the file at file_path
'''
def open_file(file_path):
    lines = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip('\n')
            line_as_list = line.split(" ")
            lines.append(line_as_list)
    return lines