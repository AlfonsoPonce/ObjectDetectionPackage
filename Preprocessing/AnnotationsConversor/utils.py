import os





def checkClassList(input):
    string_list = input.split(',')
    list = []
    iterator = 1
    for element in string_list:
        if element == ' ':
            return False, 'Class list can not have blank spaces'
        if iterator % 2 == 0 and element != ',':
            return False, 'Class list must be separated by commas'

        list.append(element)
    return True, list

def checkDirs(root_dir, dir_name, dir_type):
    dir_decomposed = dir_name.split('/')

    if len(dir_decomposed) != 1:
        return False, 'Do not use paths, just the folder name'

    datasets_dir = os.listdir('./input/')
    data_dir = os.listdir('./input/'+dir_name+'/')
    if root_dir not in datasets_dir:
        return False, 'Root folder not found'


    if dir_type == 'root':
        return True, './input/' + root_dir + "/"
    else:
        if dir_name not in data_dir and dir_type != 'output':
            return False, 'Images or annotations folders not found'
        return True, './input/'+root_dir+"/"+dir_name+"/"


def checkOutputDir(root_dir, dir_name):
    dir_decomposed = dir_name.split('/')

    if len(dir_decomposed) != 1:
        return False, 'Do not use paths, just the folder name'

    return True, './input/'+root_dir+"/"+dir_name