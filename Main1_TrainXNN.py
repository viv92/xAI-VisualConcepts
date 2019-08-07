
import numpy as np
import os
import time
import torch
import SparseRAE_CUB as SRAE_CUB
# import ModelCombineTorch as MC
# import ModelCombineIGOS as MCIGOS
# import XnnVisualizeIGOS_test as XV
# import XnnVisualizeTorch as BP
import scipy.io as scio
from Generate_Training_Multi import Generate_training_onevsall
import csv
import sys

Thershold = 0.3
use_gpu  = torch.cuda.is_available()
CategoryListFile = 'data/CUB200_Multi_csvnamelist_onevsall.csv'   #todo 1


def generate_model(out_path_base):
    DataFile = 'data/TrainData_CUBALL.mat'

    with open(CategoryListFile, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader]

    for rowi in rows:
        print(rowi)
        Cname = rowi['Name']
        label_ind = rowi['No']
        label_ind = int(label_ind)

        allowed_classes = np.array([2,5])
        label_ind_list = np.ones(allowed_classes.shape[0]) * label_ind
        if not np.any(allowed_classes == label_ind_list):
            continue

        print(CategoryListFile, DataFile)
        X_train, X_test, Y_train, Y_test, label_ind, test_index_pos = Generate_training_onevsall(Cname, label_ind, DataFile)

        print(len(label_ind))
        print(str(label_ind))
        logname = 'LOG_' + str(label_ind) + '_' + '.log'
        print(logname)
        Neuron_n = 5  # the number of x features #todo
        PT_inverse = 1  # todo 2 # EXPERIMENT HERE

        ###########################################################train the SRAE
        out_path = out_path_base + str(label_ind)  + '/'  # output path

        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        scio.savemat(out_path + 'test_index_pos' + str(label_ind)  + '.mat',  mdict={'test_index_pos': test_index_pos}, oned_as='column')

        if PT_inverse == 0:
            autoParam = [1, 100, 1, 15, 5]
        elif PT_inverse == 1:
            autoParam = [1, 100, 1, 10, 10]
        # sparseR = parameters[0], Qq = parameters[1], alpha_pred = parameters[2], alpha_recons = parameters[3], alpha_pull_away = parameters[4]
        # sparseR = 0, using traditional reconstruction loss; sparseR = 1, using sparse reconstruction loss;
        # Qq is the parameter q for the log penalty;
        # alpha_pred is the weight for prediction loss;
        # alpha_recons is the weight for reconstruction loss;
        # alpha_pull_away is the weight for pull-away term.

        for i in range(5):

            print('autoParam:', autoParam)

            MeanActivWeight_all, CorrActivWeight_all, predict_loss_train = SRAE_CUB.SparseRAE_multi(X_train, X_test, Y_train,
                                                                                                    Y_test, out_path, label_ind,
                                                                                                    Neuron_n, autoParam,
                                                                                                    useGPU=use_gpu,
                                                                                                    PT_inverse=PT_inverse)
            MeanActivWeight = MeanActivWeight_all[0,:]
            CorrActivWeight = CorrActivWeight_all[0,:]

            print('CorrActivWeight: ', type(CorrActivWeight), CorrActivWeight.shape, CorrActivWeight)
            print('MeanActivWeight: ', type(MeanActivWeight), MeanActivWeight.shape, MeanActivWeight)

            SortActivWeight = -np.sort(-MeanActivWeight)
            print('SortActiveWeight: ', SortActivWeight)

            PositiveNum = np.sum(SortActivWeight > 0)
            WeightRatio = SortActivWeight[1] / SortActivWeight[0]
            print('PositiveNum: ', PositiveNum)
            print('WeightRatio: ', WeightRatio)
            if PositiveNum >= 2 and WeightRatio >= 0.3:
                print('Good! Finish! in round ', i)
                break
            else:
                autoParam[3] = autoParam[3] * 1.2
                autoParam[4] = autoParam[4] * 0.8
                print('Round: ', i)

    return out_path_base



def main():

    out_path_base = 'models/'
    out_path_base = generate_model(  out_path_base )

if __name__ == '__main__':
    main()
