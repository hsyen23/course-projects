import numpy as np
import glob

def twistHat(twist):
    tHat = np.zeros([4,4])
    tHat[0,1] = -twist[5]
    tHat[0,2] = twist[4]
    tHat[1,0] = twist[5]
    tHat[1,2] = -twist[3]
    tHat[2,0] = -twist[4]
    tHat[2,1] = twist[3]
    tHat[0,3] = twist[0]
    tHat[1,3] = twist[1]
    tHat[2,3] = twist[2]
    return tHat

def load_KITTI_txt(path, id_list):
    file_names = glob.glob(path + '/*.txt')
    file_names.sort()
    for i in range(len(file_names)):
        file1 = open(file_names[i], 'r')
        Lines = file1.readlines()
        str1 = Lines[0][:-1]
        str1 = str1.split(' ')
        float1 = np.array([float(j) for j in str1])
        float1 = float1[id_list]
        yield float1

def get_KITTI_size(path):
    file_names = glob.glob(path + '/*.txt')
    return len(file_names)

def save_KITTI_to_np(loader, size, name):
    data = next(loader)
    l = data.shape[0]
    np_array = np.zeros([size, l])
    np_array[0] = data
    for i in range(1, size):
        data = next(loader)
        np_array[i] = data
    np.save(name, np_array)
