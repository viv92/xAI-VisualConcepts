
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os, sys ; sys.path.append('..')
from PIL import Image
import util
from skimage import transform
import scipy.io as scio
import My_io
import pdb
import random


def Generate_training_onevsall_ExamplesForSemanticConcepts(Cname, label_ind, DataFile):

    # pos_index_xnn,neg_index_xnn: list of serial numbers of active xnns
    # pos_index_category, neg_index_category: list of ground truth category ids
    # positive_index, negative_index: list of serial numbers of images in dataset
    # label_ind, pos_index_concept_labels, neg_index_concept_labels: list of one-hot concept labels

    pos_index_xnn,neg_index_xnn,\
    pos_index_category, neg_index_category,\
    positive_index, negative_index, \
    label_ind, pos_index_concept_labels, neg_index_concept_labels, pos_fileName, neg_fileName = util.Generate_onevsall_ExamplesForSemanticConcepts(Cname, label_ind)

    test_pos_num = 10 * len(label_ind) # == 10 ?
    label_ind = np.array(label_ind)
    data_all = scio.loadmat(DataFile) # load training data - note that X_train are I-features (4096) and Y_train are logits (201) - was expecting class labels

    x = list(enumerate(positive_index))
    random.shuffle(x)
    indices, positive_index = zip(*x)
    pos_index_concept_labels_shuffled= [pos_index_concept_labels[i] for i in indices] # in order to have the same shuffling
    pos_index_category_shuffled= [pos_index_category[i] for i in indices]
    pos_index_xnn_shuffled = [pos_index_xnn[i] for i in indices]
    pos_fileName_shuffled = [pos_fileName[i] for i in indices]

    # resort the labels
    x = list(enumerate(negative_index))
    random.shuffle(x)
    indices, negative_index = zip(*x)
    neg_index_concept_labels_shuffled =  [neg_index_concept_labels[i] for i in indices]
    neg_index_category_shuffled= [neg_index_category[i] for i in indices]
    neg_index_xnn_shuffled = [neg_index_xnn[i] for i in indices]
    neg_fileName_shuffled = [neg_fileName[i] for i in indices]

    test_index = positive_index[0:test_pos_num]
    Yconcept_test_index = pos_index_concept_labels_shuffled[0:test_pos_num]
    category_test = pos_index_category_shuffled[0:test_pos_num]
    xnn_test = pos_index_xnn_shuffled[0:test_pos_num]
    fileName_test = pos_fileName_shuffled[0:test_pos_num]

    test_index_pos = test_index
    test_index = np.append(test_index, negative_index[0:2000]) # test dataset has 'test_pos_num' positive examples and 2000 negative examples
    Yconcept_test_index.extend( neg_index_concept_labels_shuffled[0:2000])
    category_test.extend(neg_index_category_shuffled[0:2000] )
    xnn_test.extend(neg_index_xnn_shuffled[0:2000])
    fileName_test.extend(neg_fileName_shuffled[0:2000])


    train_index = positive_index[test_pos_num:]
    Yconcept_train_index = pos_index_concept_labels_shuffled[test_pos_num:]
    category_train= pos_index_category_shuffled[test_pos_num:]
    xnn_train= pos_index_xnn_shuffled[test_pos_num:]
    fileName_train = pos_fileName_shuffled[test_pos_num:]

    train_index = np.append(train_index, negative_index[2000:10000]) # train dataset has 'len(positive_index) - test_pos_num' positive examples and 8000 negative examples
    Yconcept_train_index.extend( neg_index_concept_labels_shuffled[2000:10000])
    category_train.extend( neg_index_category_shuffled[2000:10000])
    xnn_train.extend( neg_index_xnn_shuffled[2000:10000])
    fileName_train.extend(neg_fileName_shuffled[2000:10000])

    print('train_index:', train_index)
    print('test_index:', test_index)
    print('label_ind:', label_ind)

    X_data = data_all['X_Train'] # 11788 x 4096
    X_train = X_data[train_index, :]
    X_test = X_data[test_index, :]

    Y_data = data_all['Y_Train'] # 11788 x 201 (logits, not labels)
    Y_train = Y_data[train_index, :]  # for categories
    Y_train = Y_train[:, label_ind]  # picking logits only for the concerned class - Y just reduced from 201 columns to 1 column
    #Y_pre = np.transpose([Y_pre])
    Y_test = Y_data[test_index, :]
    Y_test = Y_test[:, label_ind]


    Yconcept_train_index=torch.FloatTensor(Yconcept_train_index) # torch tensor (not one hot)
    Yconcept_test_index=torch.FloatTensor(Yconcept_test_index)

    return(X_train, X_test, Y_train, Y_test, label_ind, test_index_pos, test_index, train_index, Yconcept_train_index, Yconcept_test_index, category_train,category_test,xnn_train, xnn_test, fileName_train,fileName_test)

def Generate_onevsall_index(Cname, label_ind):                                                                   # 1 of 3


    all_pos_index = []
    #all_neg_index = []
    all_label_ind = []
    pos_len=0
    #neg_len=0


    with open('Data/thefilename.txt','r') as f1:
        linenum = 0
        pos_index = []
        #neg_index = []
        for line in f1.readlines():
            line = line.strip()
            #print(line)
            if Cname in line:
                pos_index.append(linenum)
            #else:
            #    neg_index.append(linenum)

            linenum = linenum + 1

    #print('p_start:', p_start)
    #print('p_end:', p_end)
    #print('label_ind:',label_ind)


    pos_len +=len(pos_index)
    #neg_len +=len(neg_index)

    all_pos_index.append(pos_index)
    #all_neg_index.append(neg_index)
    all_label_ind.append(label_ind)


    print('sum of len of pos:', pos_len)
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]

    all_pos_index = flatten(all_pos_index)
    print('all_pos_index len:', len(all_pos_index))

    print('linenum:', linenum)
    ALL_index = list(range(linenum))
    print('ALL_index:', ALL_index)

    for pos_i in all_pos_index:
        ALL_index.remove(pos_i)

    all_neg_index = ALL_index
    print('all_neg_index len:', len(all_neg_index))

    print('all_pos_index:', all_pos_index)
    print('all_neg_index:', all_neg_index)
    print('all_label_ind:', all_label_ind)
    return (all_pos_index, all_neg_index, all_label_ind)

def Generate_training_onevsall_retrained(Cname, label_ind, DataFile):

    positive_index, negative_index, label_ind, pos_index_concept_labels, neg_index_concept_labels= util.Generate_onevsall_index(Cname, label_ind)
    test_pos_num = 10*len(label_ind)
    label_ind = np.array(label_ind)
    # dataFile = '/home/phoenixqq/My_program/My_autoencoder/autoencoder_orthogonal/TrainData_CUBALL.mat'
    # dataFile2 = '/home/phoenixqq/My_program/My_autoencoder/autoencoder_orthogonal/PCAALL.mat'
    # dataFile2 = '/home/phoenixqq/My_program/My_autoencoder/autoencoder_orthogonal/PCAALLwithoutPCA.mat'
    # p_start = 3546 #line-1  (1)
    # p_end = 3606  #line      (2)
    # label_ind = 62    # for 062 (3)

    data_all = scio.loadmat(DataFile)
    #data_all2 = scio.loadmat(dataFile2)

    #positive_index = np.array(range(p_start, p_end))  # for categories
    #if p_start > 0 and p_end < 11788:
    #    negative_index = np.array(range(p_start))
    #    negative_index = np.append(negative_index, range(p_end, 11788))
    #elif p_start == 0:
    #    negative_index = np.array(range(p_end, 11788))
    #elif p_end == 11788:
    #    negative_index = np.array(range(p_start))
    # random.shuffle(positive_index)
    # positive_index1=positive_index
    x = list(enumerate(positive_index))
    random.shuffle(x)
    indices, positive_index = zip(*x)
    pos_index_concept_labels_shuffled= [pos_index_concept_labels[i] for i in indices]

    # resort the labels

    # random.shuffle(negative_index)
    x = list(enumerate(negative_index))
    random.shuffle(x)
    indices, negative_index = zip(*x)
    neg_index_concept_labels_shuffled= [neg_index_concept_labels[i] for i in indices]

    test_index = positive_index[0:test_pos_num]
    Yconcept_test_index = pos_index_concept_labels[0:test_pos_num]
    test_index_pos = test_index
    test_index = np.append(test_index, negative_index[0:2000])
    Yconcept_test_index.extend( neg_index_concept_labels[0:2000])


    train_index = positive_index[test_pos_num:]
    Yconcept_train_index = pos_index_concept_labels[test_pos_num:]
    train_index = np.append(train_index, negative_index[2000:10000])
    Yconcept_train_index.extend( neg_index_concept_labels[2000:10000])

    print('train_index:', train_index)
    print('test_index:', test_index)
    print('label_ind:', label_ind)

    X_data = data_all['X_Train']
    X_train = X_data[train_index, :]
    X_test = X_data[test_index, :]

    Y_data = data_all['Y_Train']
    Y_train = Y_data[train_index, :]  # for categories
    Y_train = Y_train[:, label_ind]  # for categories
    #Y_pre = np.transpose([Y_pre])
    Y_test = Y_data[test_index, :]
    Y_test = Y_test[:, label_ind]
    #Y_pretest = np.transpose([Y_pretest])
    # print('X_train:', X_train)
    # print('Y_train:', Y_train)
    # print('Yconcept_train:', Yconcept_train_index)
    # print('X_test:', X_test)
    # print('Y_test:', Y_test)
    # print('Yconcept_test:', Yconcept_test_index)

    # print()
    Yconcept_train_index=torch.FloatTensor(Yconcept_train_index)
    Yconcept_test_index=torch.FloatTensor(Yconcept_test_index)


    return(X_train, X_test, Y_train, Y_test, label_ind, test_index_pos, test_index, train_index, Yconcept_train_index, Yconcept_test_index)
def Generate_training_onevsall(Cname, label_ind, DataFile):

    positive_index, negative_index, label_ind= util.Generate_onevsall_index(Cname, label_ind)
    test_pos_num = 10*len(label_ind)
    label_ind = np.array(label_ind)
    # dataFile = '/home/phoenixqq/My_program/My_autoencoder/autoencoder_orthogonal/TrainData_CUBALL.mat'
    # dataFile2 = '/home/phoenixqq/My_program/My_autoencoder/autoencoder_orthogonal/PCAALL.mat'
    # dataFile2 = '/home/phoenixqq/My_program/My_autoencoder/autoencoder_orthogonal/PCAALLwithoutPCA.mat'
    # p_start = 3546 #line-1  (1)
    # p_end = 3606  #line      (2)
    # label_ind = 62    # for 062 (3)

    data_all = scio.loadmat(DataFile)
    #data_all2 = scio.loadmat(dataFile2)

    #positive_index = np.array(range(p_start, p_end))  # for categories
    #if p_start > 0 and p_end < 11788:
    #    negative_index = np.array(range(p_start))
    #    negative_index = np.append(negative_index, range(p_end, 11788))
    #elif p_start == 0:
    #    negative_index = np.array(range(p_end, 11788))
    #elif p_end == 11788:
    #    negative_index = np.array(range(p_start))
    random.shuffle(positive_index)
    random.shuffle(negative_index)

    test_index = positive_index[0:test_pos_num]
    test_index_pos = test_index
    test_index = np.append(test_index, negative_index[0:2000])

    train_index = positive_index[test_pos_num:]
    train_index = np.append(train_index, negative_index[2000:10000])

    print('train_index:', train_index)
    print('test_index:', test_index)
    print('label_ind:', label_ind)

    X_data = data_all['X_Train']
    X_train = X_data[train_index, :]
    X_test = X_data[test_index, :]

    Y_data = data_all['Y_Train']
    Y_train = Y_data[train_index, :]  # for categories
    Y_train = Y_train[:, label_ind]  # for categories
    #Y_pre = np.transpose([Y_pre])
    Y_test = Y_data[test_index, :]
    Y_test = Y_test[:, label_ind]
    #Y_pretest = np.transpose([Y_pretest])
    print('X_train:', X_train)
    print('Y_train:', Y_train)
    print('X_test:', X_test)
    print('Y_test:', Y_test)


    return(X_train, X_test, Y_train, Y_test, label_ind, test_index_pos)



def Generate_training(CategoryListFile, DataFile):

    positive_index, negative_index, label_ind = util.Generate_multi_index(CategoryListFile)
    test_pos_num = 10*len(label_ind)
    label_ind = np.array(label_ind)
    # dataFile = '/home/phoenixqq/My_program/My_autoencoder/autoencoder_orthogonal/TrainData_CUBALL.mat'
    # dataFile2 = '/home/phoenixqq/My_program/My_autoencoder/autoencoder_orthogonal/PCAALL.mat'
    # dataFile2 = '/home/phoenixqq/My_program/My_autoencoder/autoencoder_orthogonal/PCAALLwithoutPCA.mat'
    # p_start = 3546 #line-1  (1)
    # p_end = 3606  #line      (2)
    # label_ind = 62    # for 062 (3)

    data_all = scio.loadmat(DataFile)
    #data_all2 = scio.loadmat(dataFile2)

    #positive_index = np.array(range(p_start, p_end))  # for categories
    #if p_start > 0 and p_end < 11788:
    #    negative_index = np.array(range(p_start))
    #    negative_index = np.append(negative_index, range(p_end, 11788))
    #elif p_start == 0:
    #    negative_index = np.array(range(p_end, 11788))
    #elif p_end == 11788:
    #    negative_index = np.array(range(p_start))
    random.shuffle(positive_index)
    random.shuffle(negative_index)

    test_index = positive_index[0:test_pos_num]
    test_index_pos = test_index
    test_index = np.append(test_index, negative_index[0:2000])

    train_index = positive_index[test_pos_num:]
    train_index = np.append(train_index, negative_index[2000:10000])

    print('train_index:', train_index)
    print('test_index:', test_index)
    print('label_ind:', label_ind)

    X_data = data_all['X_Train']
    X_train = X_data[train_index, :]
    X_test = X_data[test_index, :]

    Y_data = data_all['Y_Train']
    Y_train = Y_data[train_index, :]  # for categories
    Y_train = Y_train[:, label_ind]  # for categories
    #Y_pre = np.transpose([Y_pre])
    Y_test = Y_data[test_index, :]
    Y_test = Y_test[:, label_ind]
    #Y_pretest = np.transpose([Y_pretest])
    print('X_train:', X_train)
    print('Y_train:', Y_train)
    print('X_test:', X_test)
    print('Y_test:', Y_test)


    return(X_train, X_test, Y_train, Y_test, label_ind, test_index_pos)


####################################################################################################################


def main():
    CategoryListFile = 'Data/CUB200_Multi_csvnamelist.csv'
    DataFile = '../../../../DataSetforXAI/Results_newEBP/Model/TrainData_CUBALL.mat'

    print(CategoryListFile, DataFile)
    X_train, X_test, Y_train, Y_test, label_ind, test_index_pos = Generate_training(CategoryListFile, DataFile)  # useGPU=0, it seems there is no gpu version for excitationBP in pytorch



if __name__ == '__main__':
    main()
