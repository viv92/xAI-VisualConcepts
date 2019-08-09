#coding=utf-8
# SparseSAE model training
# python Version: python3.6

import torch
from torch.autograd import Variable
import numpy as np
import os
import scipy.io as scio
import random
import autoencoder
import time
import sys


from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")


from Generate_Training_Multi import Generate_training
def SparseRAE_multi(X_train, X_test, GY_train, GY_test, target_path, label_ind, Neuron_n, autoParam, useGPU =1, PT_inverse=1):
    ########################
    # Train a sparse reconstruction autoencoder model (SRAE)

    # Parameters:
    # -------------
    # dataFile_pos: the positive training and testing data
    # dataFile_neg: the negative training and testing data
    # target_path: the output path to save the results
    # label_ind: the label index, represents which category need to be trained
    # filetime: local time
    # Neuron_n: the number of the x-features
    # autoParam: Parameters for fine-tuning SRAE
    # useGPU =1: Whether to use GPU (1) or not (0)
    # PT_inverse=1: 1 means pull away term along images, 0 means pull away term along features

    # Returns:
    # -------------
    # MeanActivWeight: the mean value of activation * weight for x-features
    # CorrActivWeight: the correlation coefficient of activation * weight for x-features
    ####################################################



    logname = target_path+'LOG_'+str(label_ind)+'.log'
    print(logname)
    log_file = open(logname, 'w')
    stdout_backup = sys.stdout

    log_true = 0                      # 1 means outputting to the log file, 0 means outputting to the screen #todo
    EPOCH = 1 #10                         # the epoch for parameter pretrain
    BATCH_SIZE = 50
    LR = 0.001


    #############the structure of SRAE
    NN_input = X_train.shape[1]
    NN_middle1 = 800
    NN_middle2 = 100
    NN_middle3 = Neuron_n
    NN_output = len(label_ind)


    print(X_train.shape)
    print(GY_train.shape)

    ##################################################################################################################
    # pretrain the first layer of SRAE
    AE1 = autoencoder.AE(NN_input, NN_middle1)
    print(('Autoencoder 1: ', AE1))


    train_x1 = torch.from_numpy(X_train).float()
    test_x1 = torch.from_numpy(X_test).float()

    AE1 = autoencoder.AutoencoderOptim(train_x1, test_x1, AE1, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU)

    if useGPU==1:
        AE1 = AE1.cuda()
        (train_x2, trainDe) = AE1(Variable(train_x1).cuda())
        train_x2 = train_x2.cpu().data
        print(train_x2)
        (test_x2, testDe) = AE1(Variable(test_x1).cuda())
        test_x2 = test_x2.cpu().data
    else:
        (train_x2, trainDe) = AE1(Variable(train_x1))
        train_x2 = train_x2.data
        print(train_x2)
        (test_x2, testDe) = AE1(Variable(test_x1))
        test_x2 = test_x2.data


    SAE_weights = dict()  # save the weights
    if useGPU==1:
        SAE_weights['W1'] = AE1.encoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b1'] = AE1.encoder[0].bias.data.cpu().numpy().copy()
        SAE_weights['W-1'] = AE1.decoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b-1'] = AE1.decoder[0].bias.data.cpu().numpy().copy()
    else:
        SAE_weights['W1'] = AE1.encoder[0].weight.data.numpy().copy()
        SAE_weights['b1'] = AE1.encoder[0].bias.data.numpy().copy()
        SAE_weights['W-1'] = AE1.decoder[0].weight.data.numpy().copy()
        SAE_weights['b-1'] = AE1.decoder[0].bias.data.numpy().copy()


    ####################################################################################################################
    # pretrain the second layer of SRAE
    AE2 = autoencoder.AE(NN_middle1, NN_middle2)
    print(('Autoencoder 2: ', AE2))



    AE2 = autoencoder.AutoencoderOptim(train_x2, test_x2, AE2, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU)

    if useGPU==1:
        AE2 = AE2.cuda()
        (train_x3, trainDe) = AE2(Variable(train_x2).cuda())
        train_x3 = train_x3.cpu().data
        (test_x3, trainDe) = AE2(Variable(test_x2).cuda())
        test_x3 = test_x3.cpu().data
    else:
        (train_x3, trainDe) = AE2(Variable(train_x2))
        train_x3 = train_x3.data
        (test_x3, trainDe) = AE2(Variable(test_x2))
        test_x3 = test_x3.data

    if useGPU == 1:
        SAE_weights['W2'] = AE2.encoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b2'] = AE2.encoder[0].bias.data.cpu().numpy().copy()
        SAE_weights['W-2'] = AE2.decoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b-2'] = AE2.decoder[0].bias.data.cpu().numpy().copy()

    else:
        SAE_weights['W2'] = AE2.encoder[0].weight.data.numpy().copy()
        SAE_weights['b2'] = AE2.encoder[0].bias.data.numpy().copy()
        SAE_weights['W-2'] = AE2.decoder[0].weight.data.numpy().copy()
        SAE_weights['b-2'] = AE2.decoder[0].bias.data.numpy().copy()


    ####################################################################################################################
    # pretrain the third layer of SRAE

    AE3 = autoencoder.AE(NN_middle2, NN_middle3)
    print(('Autoencoder 3: ', AE3))



    AE3 = autoencoder.AutoencoderOptim(train_x3, test_x3, AE3, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU)

    if useGPU==1:
        AE3 = AE3.cuda()
        (train_x4, trainDe) = AE3(Variable(train_x3).cuda())
        train_x4 = train_x4.cpu().data
        (test_x4, trainDe) = AE3(Variable(test_x3).cuda())
        test_x4 = test_x4.cpu().data

    else:
        (train_x4, trainDe) = AE3(Variable(train_x3))
        train_x4 = train_x4.data
        (test_x4, trainDe) = AE3(Variable(test_x3))
        test_x4 = test_x4.data

    if useGPU == 1:
        SAE_weights['W3'] = AE3.encoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b3'] = AE3.encoder[0].bias.data.cpu().numpy().copy()
        SAE_weights['W-3'] = AE3.decoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b-3'] = AE3.decoder[0].bias.data.cpu().numpy().copy()
    else:
        SAE_weights['W3'] = AE3.encoder[0].weight.data.numpy().copy()
        SAE_weights['b3'] = AE3.encoder[0].bias.data.numpy().copy()
        SAE_weights['W-3'] = AE3.decoder[0].weight.data.numpy().copy()
        SAE_weights['b-3'] = AE3.decoder[0].bias.data.numpy().copy()

    ####################################################################################################################
    # pretrain the last layer of SRAE

    LR4 = autoencoder.LR(NN_middle3, NN_output)
    print(('Linear Regression: ', LR4))

    train_Gy = torch.from_numpy(GY_train).unsqueeze(1).float()
    test_Gy = torch.from_numpy(GY_test).unsqueeze(1).float()

    LR4 = autoencoder.LR_Optim(train_x4, train_Gy, test_x4, test_Gy, LR4, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU)

    if useGPU==1:
        SAE_weights['Wout'] = LR4.regression.weight.data.cpu().numpy().copy()
        SAE_weights['bout'] = LR4.regression.bias.data.cpu().numpy().copy()
    else:
        SAE_weights['Wout'] = LR4.regression.weight.data.numpy().copy()
        SAE_weights['bout'] = LR4.regression.bias.data.numpy().copy()

    ####################################################################################################################
    # pretrain the whole network of SRAE

    StackAE = autoencoder.SRAE(NN_input, NN_middle1, NN_middle2, NN_middle3, NN_output)
    print(('Stacked Autoencoder: ', StackAE))

    trainX = torch.from_numpy(X_train).float()
    testX = torch.from_numpy(X_test).float()

    parameters = [0, 0, 1, 300, 0]    # #todo
    #Only use stacked autoencoder to perform pretrain
    #sparseR = parameters[0], Qq = parameters[1], alpha_pred = parameters[2], alpha_recons = parameters[3], alpha_pull_away = parameters[4]
    #sparseR = 0, using traditional reconstruction loss; sparseR = 1, using sparse reconstruction loss;
    #Qq is the parameter q for the log penalty;
    #alpha_pred is the weight for prediction loss;
    #alpha_recons is the weight for reconstruction loss;
    #alpha_pull_away is the weight for pull-away term.




    ###########weights initialization:
    StackAE.encoder1[0].weight.data.copy_(torch.from_numpy(SAE_weights['W1']))
    StackAE.encoder1[0].bias.data.copy_(torch.from_numpy(SAE_weights['b1']))
    StackAE.encoder2[0].weight.data.copy_(torch.from_numpy(SAE_weights['W2']))
    StackAE.encoder2[0].bias.data.copy_(torch.from_numpy(SAE_weights['b2']))
    StackAE.encoder3[0].weight.data.copy_(torch.from_numpy(SAE_weights['W3']))
    StackAE.encoder3[0].bias.data.copy_(torch.from_numpy(SAE_weights['b3']))

    StackAE.decoder3[0].weight.data.copy_(torch.from_numpy(SAE_weights['W-3']))
    StackAE.decoder3[0].bias.data.copy_(torch.from_numpy(SAE_weights['b-3']))
    StackAE.decoder2[0].weight.data.copy_(torch.from_numpy(SAE_weights['W-2']))
    StackAE.decoder2[0].bias.data.copy_(torch.from_numpy(SAE_weights['b-2']))
    StackAE.decoder1[0].weight.data.copy_(torch.from_numpy(SAE_weights['W-1']))
    StackAE.decoder1[0].bias.data.copy_(torch.from_numpy(SAE_weights['b-1']))

    StackAE.classifier.weight.data.copy_(torch.from_numpy(SAE_weights['Wout']))
    StackAE.classifier.bias.data.copy_(torch.from_numpy(SAE_weights['bout']))

    ####################weights initialization end


    #StackAE, MeanActivWeight0, CorrActivWeight0, predict_loss_train0 = autoencoder.SRAE_Optim(trainX, train_Gy, testX, test_Gy, StackAE, parameters, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU, PT_inverse=PT_inverse)


    ####################################################################################################################

    SRAE = StackAE
    print(('Stacked Autoencoder: ', SRAE))


    #parametersSRAE = [1, 100, 1, 100, 0.5]  #Sparse reconstruction autoencoder
    parametersSRAE = autoParam
    # sparseR = parameters[0], Qq = parameters[1], alpha_pred = parameters[2], alpha_recons = parameters[3], alpha_pull_away = parameters[4]
    # sparseR = 0, using traditional reconstruction loss; sparseR = 1, using sparse reconstruction loss;
    # Qq is the parameter q for the log penalty;
    # alpha_pred is the weight for prediction loss;
    # alpha_recons is the weight for reconstruction loss;
    # alpha_pull_away is the weight for pull-away term.



    print('waiting...')
    if log_true == 1:    # if log_true == 1, print all the messages in the log file.
        sys.stdout = log_file



    SRAE, MeanActivWeight, CorrActivWeight, predict_loss_train = autoencoder.SRAE_Optim(trainX, train_Gy, testX, test_Gy, SRAE, parametersSRAE, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU, PT_inverse=PT_inverse)

    ########################################
    SRAE.eval()
    if useGPU==1:
        Ltest_x = Variable(testX).cuda()
        Ltrain_x = Variable(trainX).cuda()

    else:
        Ltest_x = Variable(testX)
        Ltrain_x = Variable(trainX)
    (trainX_encoder, trainX_decoder, trainY_pred) = SRAE(Ltrain_x)
    (testX_encoder, testX_decoder, testY_pred) = SRAE(Ltest_x)

    trainY_pred_np = trainY_pred.data.cpu().numpy().copy()
    testY_pred_np = testY_pred.data.cpu().numpy().copy()

    scio.savemat(target_path + 'Prediction' + str(label_ind)  + '.mat',
                 mdict={'trainY_pred': trainY_pred_np, 'train_Gy': GY_train, 'testY_pred': testY_pred_np, 'test_Gy': GY_test}, oned_as='column')

    ########################################################################

    ####################################################################################################################
    #save the model


    modelname = target_path+'SRAE_'+str(label_ind)+'.pth'
    torch.save(SRAE, modelname)


    SRAE_weights = dict()
    SRAE_weights['W1'] = SRAE.encoder1[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b1'] = SRAE.encoder1[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W2'] = SRAE.encoder2[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b2'] = SRAE.encoder2[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W3'] = SRAE.encoder3[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b3'] = SRAE.encoder3[0].bias.data.cpu().numpy().copy()
    SRAE_weights['Wo'] = SRAE.classifier.weight.data.cpu().numpy().copy()
    SRAE_weights['bo'] = SRAE.classifier.bias.data.cpu().numpy().copy()

    SRAE_weights['W-3'] = SRAE.decoder3[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b-3'] = SRAE.decoder3[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W-2'] = SRAE.decoder2[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b-2'] = SRAE.decoder2[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W-1'] = SRAE.decoder1[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b-1'] = SRAE.decoder1[0].bias.data.cpu().numpy().copy()



    scio.savemat(target_path + 'SRAEweight_'+str(label_ind)+'.mat', mdict={'SRAE_weights': SRAE_weights}, oned_as='column')

    log_file.close()
    sys.stdout = stdout_backup

    return (MeanActivWeight, CorrActivWeight, predict_loss_train)
    print('Finished!')

def SparseRAE_multi_retrainedModel_AllDatalossfunction(useXNNOrNOT,X_train, X_test, GY_train, GY_test, Yconcept_train, Yconcept_test,  MissingLabel_Train, MissingLabel_Test,
                                           test_index, train_index,target_path, label_ind, Neuron_n, autoParam, useGPU =1, PT_inverse=1):
    ########################
    # Train a sparse reconstruction autoencoder model (SRAE)
    # Parameters:
    # -------------
    # dataFile_pos: the positive training and testing data
    # dataFile_neg: the negative training and testing data
    # target_path: the output path to save te results

    # label_ind: the label index, represents which category need to be trained
    # filetime: local time
    # Neuron_n: the number of the x-features
    # autoParam: Parameters for fine-tuning SRAE
    # useGPU =1: Whether to use GPU (1) or not (0)
    # PT_inverse=1: 1 means pull away term along images, 0 means pull away term along features
    #Neuron_concepts_n: the number of concepts
    # Returns:
    # -------------
    # MeanActivWeight: the mean value of activation * weight for x-features
    # CorrActivWeight: the correlation coefficient of activation * weight for x-features
    ####################################################
    Yconcept_train_np = Yconcept_train.numpy().copy()
    Yconcept_test_np = Yconcept_test.numpy().copy()
    scio.savemat(target_path + 'Prediction_Concept_PREEEEE_' + str(label_ind) + '1.mat',
                 mdict={'train_Gy': GY_train, 'test_Gy': GY_test,
                        'train_index': train_index, 'test_index': test_index,
                        'Yconcept_train': Yconcept_train_np, 'Yconcept_test': Yconcept_test_np}, oned_as='column')

    scio.savemat(target_path + 'Prediction_Concept_indices' + str(label_ind) + '.mat', mdict={ 'train_index': train_index, 'test_index': test_index,}, oned_as='column')

    logname = target_path + 'LOG_'+ str(label_ind)  +'.log'
    print(logname)
    log_file = open(logname, 'w')
    stdout_backup = sys.stdout

    log_true = 0                      # 1 means outputting to the log file, 0 means outputting to the screen #todo
    EPOCH = 10                         # the epoch for parameter pretrain
    BATCH_SIZE = 50
    LR = 0.001

    Neuron_concepts_n = Yconcept_train[0].shape[0]

    #############the structure of SRAE
    NN_input = X_train.shape[1]
    NN_middle1 = 800
    NN_middle2 = 100
    if (useXNNOrNOT==1):
        if Neuron_concepts_n > Neuron_n:
            NN_middle3 = Neuron_concepts_n
        else:
            NN_middle3 = Neuron_n
    else:
        NN_middle3= Neuron_concepts_n
    NN_output = len(label_ind)


    print(X_train.shape)
    #NZ_X = X_train[0:100,:]
    #NZ_index = NZ_X.nonzero()
    #NZ_X2 = NZ_X[NZ_index]
    #print(NZ_X2.shape)
    print(GY_train.shape)

    ##################################################################################################################
    # pretrain the first layer of SRAE
    AE1 = autoencoder.AE(NN_input, NN_middle1)
    print(('Autoencoder 1: ', AE1))


    train_x1 = torch.from_numpy(X_train).float()
    test_x1 = torch.from_numpy(X_test).float()

    AE1 = autoencoder.AutoencoderOptim(train_x1, test_x1, AE1, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU)

    if useGPU==1:
        AE1 = AE1.cuda()
        (train_x2, trainDe) = AE1(Variable(train_x1).cuda())
        train_x2 = train_x2.cpu().data
        print(train_x2)
        (test_x2, testDe) = AE1(Variable(test_x1).cuda())
        test_x2 = test_x2.cpu().data
    else:
        (train_x2, trainDe) = AE1(Variable(train_x1))
        train_x2 = train_x2.data
        print(train_x2)
        (test_x2, testDe) = AE1(Variable(test_x1))
        test_x2 = test_x2.data


    SAE_weights = dict()  # save the weights
    if useGPU==1:
        SAE_weights['W1'] = AE1.encoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b1'] = AE1.encoder[0].bias.data.cpu().numpy().copy()
        SAE_weights['W-1'] = AE1.decoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b-1'] = AE1.decoder[0].bias.data.cpu().numpy().copy()
    else:
        SAE_weights['W1'] = AE1.encoder[0].weight.data.numpy().copy()
        SAE_weights['b1'] = AE1.encoder[0].bias.data.numpy().copy()
        SAE_weights['W-1'] = AE1.decoder[0].weight.data.numpy().copy()
        SAE_weights['b-1'] = AE1.decoder[0].bias.data.numpy().copy()


    ####################################################################################################################
    # pretrain the second layer of SRAE
    AE2 = autoencoder.AE(NN_middle1, NN_middle2)
    print(('Autoencoder 2: ', AE2))



    AE2 = autoencoder.AutoencoderOptim(train_x2, test_x2, AE2, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU)

    if useGPU==1:
        AE2 = AE2.cuda()
        (train_x3, trainDe) = AE2(Variable(train_x2).cuda())
        train_x3 = train_x3.cpu().data
        (test_x3, trainDe) = AE2(Variable(test_x2).cuda())
        test_x3 = test_x3.cpu().data
    else:
        (train_x3, trainDe) = AE2(Variable(train_x2))
        train_x3 = train_x3.data
        (test_x3, trainDe) = AE2(Variable(test_x2))
        test_x3 = test_x3.data

    if useGPU == 1:
        SAE_weights['W2'] = AE2.encoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b2'] = AE2.encoder[0].bias.data.cpu().numpy().copy()
        SAE_weights['W-2'] = AE2.decoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b-2'] = AE2.decoder[0].bias.data.cpu().numpy().copy()

    else:
        SAE_weights['W2'] = AE2.encoder[0].weight.data.numpy().copy()
        SAE_weights['b2'] = AE2.encoder[0].bias.data.numpy().copy()
        SAE_weights['W-2'] = AE2.decoder[0].weight.data.numpy().copy()
        SAE_weights['b-2'] = AE2.decoder[0].bias.data.numpy().copy()


    ####################################################################################################################
    # pretrain the third layer of SRAE

    AE3 = autoencoder.AE(NN_middle2, NN_middle3)
    print(('Autoencoder 3: ', AE3))



    AE3 = autoencoder.AutoencoderOptim(train_x3, test_x3, AE3, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU)

    if useGPU==1:
        AE3 = AE3.cuda()
        (train_x4, trainDe) = AE3(Variable(train_x3).cuda())
        train_x4 = train_x4.cpu().data
        (test_x4, trainDe) = AE3(Variable(test_x3).cuda())
        test_x4 = test_x4.cpu().data

    else:
        (train_x4, trainDe) = AE3(Variable(train_x3))
        train_x4 = train_x4.data
        (test_x4, trainDe) = AE3(Variable(test_x3))
        test_x4 = test_x4.data

    if useGPU == 1:
        SAE_weights['W3'] = AE3.encoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b3'] = AE3.encoder[0].bias.data.cpu().numpy().copy()
        SAE_weights['W-3'] = AE3.decoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b-3'] = AE3.decoder[0].bias.data.cpu().numpy().copy()
    else:
        SAE_weights['W3'] = AE3.encoder[0].weight.data.numpy().copy()
        SAE_weights['b3'] = AE3.encoder[0].bias.data.numpy().copy()
        SAE_weights['W-3'] = AE3.decoder[0].weight.data.numpy().copy()
        SAE_weights['b-3'] = AE3.decoder[0].bias.data.numpy().copy()

    ####################################################################################################################
    # pretrain the last layer of SRAE

    LR4 = autoencoder.LR(NN_middle3, NN_output)
    print(('Linear Regression: ', LR4))

    train_Gy = torch.from_numpy(GY_train).unsqueeze(1).float()
    test_Gy = torch.from_numpy(GY_test).unsqueeze(1).float()

    LR4 = autoencoder.LR_Optim(train_x4, train_Gy, test_x4, test_Gy, LR4, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU)

    if useGPU==1:
        SAE_weights['Wout'] = LR4.regression.weight.data.cpu().numpy().copy()
        SAE_weights['bout'] = LR4.regression.bias.data.cpu().numpy().copy()
    else:
        SAE_weights['Wout'] = LR4.regression.weight.data.numpy().copy()
        SAE_weights['bout'] = LR4.regression.bias.data.numpy().copy()

    ####################################################################################################################
    # pretrain the whole network of SRAE

    StackAE = autoencoder.SRAE(NN_input, NN_middle1, NN_middle2, NN_middle3, NN_output)
    print(('Stacked Autoencoder: ', StackAE))

    trainX = torch.from_numpy(X_train).float()
    testX = torch.from_numpy(X_test).float()

    parameters = autoParam#[0, 0, 1, 10, 0.5, 500]    # #todo
    #Only use stacked autoencoder to perform pretrain
    #sparseR = parameters[0], Qq = parameters[1], alpha_pred = parameters[2], alpha_recons = parameters[3], alpha_pull_away = parameters[4]
    #sparseR = 0, using traditional reconstruction loss; sparseR = 1, using sparse reconstruction loss;
    #Qq is the parameter q for the log penalty;
    #alpha_pred is the weight for prediction loss;
    #alpha_recons is the weight for reconstruction loss;
    #alpha_pull_away is the weight for pull-away term.


    ###########weights initialization:
    StackAE.encoder1[0].weight.data.copy_(torch.from_numpy(SAE_weights['W1']))
    StackAE.encoder1[0].bias.data.copy_(torch.from_numpy(SAE_weights['b1']))
    StackAE.encoder2[0].weight.data.copy_(torch.from_numpy(SAE_weights['W2']))
    StackAE.encoder2[0].bias.data.copy_(torch.from_numpy(SAE_weights['b2']))
    StackAE.encoder3[0].weight.data.copy_(torch.from_numpy(SAE_weights['W3']))
    StackAE.encoder3[0].bias.data.copy_(torch.from_numpy(SAE_weights['b3']))

    StackAE.decoder3[0].weight.data.copy_(torch.from_numpy(SAE_weights['W-3']))
    StackAE.decoder3[0].bias.data.copy_(torch.from_numpy(SAE_weights['b-3']))
    StackAE.decoder2[0].weight.data.copy_(torch.from_numpy(SAE_weights['W-2']))
    StackAE.decoder2[0].bias.data.copy_(torch.from_numpy(SAE_weights['b-2']))
    StackAE.decoder1[0].weight.data.copy_(torch.from_numpy(SAE_weights['W-1']))
    StackAE.decoder1[0].bias.data.copy_(torch.from_numpy(SAE_weights['b-1']))

    StackAE.classifier.weight.data.copy_(torch.from_numpy(SAE_weights['Wout']))
    StackAE.classifier.bias.data.copy_(torch.from_numpy(SAE_weights['bout']))

    ####################weights initialization end


    StackAE, MeanActivWeight0, CorrActivWeight0, predict_loss_train0 = autoencoder.SRAE_Optim_retrained_AllDataLossFunction(
        trainX, train_Gy, testX, test_Gy,
        Yconcept_train, Yconcept_test,
        StackAE, parameters, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU, PT_inverse=PT_inverse)


    ####################################################################################################################

    SRAE = StackAE
    print(('Stacked Autoencoder: ', SRAE))


    #parametersSRAE = [1, 100, 1, 100, 0.5]  #Sparse reconstruction autoencoder
    parametersSRAE = autoParam
    # sparseR = parameters[0], Qq = parameters[1], alpha_pred = parameters[2], alpha_recons = parameters[3], alpha_pull_away = parameters[4]
    # sparseR = 0, using traditional reconstruction loss; sparseR = 1, using sparse reconstruction loss;
    # Qq is the parameter q for the log penalty;
    # alpha_pred is the weight for prediction loss;
    # alpha_recons is the weight for reconstruction loss;
    # alpha_pull_away is the weight for pull-away term.


    print('waiting...')
    if log_true == 1:    # if log_true == 1, print all the messages in the log file.
        sys.stdout = log_file



    SRAE, MeanActivWeight, CorrActivWeight, predict_loss_train = autoencoder.SRAE_Optim_retrained_AllDataLossFunction(
        trainX, train_Gy, testX, test_Gy,
        Yconcept_train, Yconcept_test,
        SRAE, parametersSRAE, BatchSize=BATCH_SIZE,
        # Lr=LR, EPOCH=EPOCH, useGPU=useGPU, PT_inverse=PT_inverse)
        Lr=LR, EPOCH=EPOCH+120, useGPU=useGPU, PT_inverse=PT_inverse)

    ########################################
    SRAE.eval()
    if useGPU==1:
        Ltest_x = Variable(testX).cuda()
        Ltrain_x = Variable(trainX).cuda()

    else:
        Ltest_x = Variable(testX)
        Ltrain_x = Variable(trainX)
    (trainX_encoder, trainX_decoder, trainY_pred) = SRAE(Ltrain_x)
    (testX_encoder, testX_decoder, testY_pred) = SRAE(Ltest_x)

    trainY_pred_np = trainY_pred.data.cpu().numpy().copy()
    testY_pred_np = testY_pred.data.cpu().numpy().copy()



    scio.savemat(target_path + 'Prediction' + str(label_ind) + '.mat',
                 mdict={'trainY_pred': trainY_pred_np, 'train_Gy': GY_train, 'testY_pred': testY_pred_np, 'test_Gy': GY_test}, oned_as='column')

    trainX_encoder_np = trainX_encoder.data.cpu().numpy().copy()
    testX_encoder_np = testX_encoder.data.cpu().numpy().copy()
    #
    Yconcept_train_np = Yconcept_train.numpy().copy()
    Yconcept_test_np = Yconcept_test.numpy().copy()

    scio.savemat(target_path + 'Prediction_Concept' + str(label_ind) + '1.mat',
                 mdict={'trainY_pred': trainY_pred_np, 'train_Gy': GY_train, 'testY_pred': testY_pred_np, 'test_Gy': GY_test,
                        'trainX_encoder': trainX_encoder_np, 'testX_encoder': testX_encoder_np,
                        'train_index': train_index, 'test_index': test_index,
                        'Yconcept_train': Yconcept_train_np, 'Yconcept_test': Yconcept_test_np}, oned_as='column')



    ###################################################################################################################

    ####################################################################################################################
    #save the model

    modelname = target_path+'SRAE_'+str(label_ind)+'.pth'
    torch.save(SRAE, modelname)

    SRAE_weights = dict()
    SRAE_weights['W1'] = SRAE.encoder1[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b1'] = SRAE.encoder1[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W2'] = SRAE.encoder2[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b2'] = SRAE.encoder2[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W3'] = SRAE.encoder3[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b3'] = SRAE.encoder3[0].bias.data.cpu().numpy().copy()
    SRAE_weights['Wo'] = SRAE.classifier.weight.data.cpu().numpy().copy()
    SRAE_weights['bo'] = SRAE.classifier.bias.data.cpu().numpy().copy()

    SRAE_weights['W-3'] = SRAE.decoder3[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b-3'] = SRAE.decoder3[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W-2'] = SRAE.decoder2[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b-2'] = SRAE.decoder2[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W-1'] = SRAE.decoder1[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b-1'] = SRAE.decoder1[0].bias.data.cpu().numpy().copy()


    scio.savemat(target_path + 'SRAEweight_'+str(label_ind)+'.mat', mdict={'SRAE_weights': SRAE_weights}, oned_as='column')

    log_file.close()
    sys.stdout = stdout_backup

    return (MeanActivWeight, CorrActivWeight, predict_loss_train)
    print('Finished!')


def SparseRAE_multi_finetunedModel(useXNNOrNOT,conceptLoss_1allData_0nonMissingData, X_train, X_test, GY_train, GY_test, Yconcept_train, Yconcept_test,
                                   MissingLabel_Train, MissingLabel_Test,
                                   test_index, train_index,target_path, label_ind, Neuron_n, autoParam, useGPU =1, PT_inverse=1):
    Yconcept_train_np = Yconcept_train.numpy().copy()
    Yconcept_test_np = Yconcept_test.numpy().copy()
    scio.savemat(target_path + 'Prediction_Concept_PREEEEE_' + str(label_ind) + '1.mat',
                 mdict={'train_Gy': GY_train, 'test_Gy': GY_test,
                        'train_index': train_index, 'test_index': test_index,
                        'Yconcept_train': Yconcept_train_np, 'Yconcept_test': Yconcept_test_np}, oned_as='column')
    scio.savemat(target_path + 'Prediction_Concept_indices' + str(label_ind) + '.mat',
                 mdict={ 'train_index': train_index, 'test_index': test_index,}, oned_as='column')

    logname = target_path + 'LOG_'+ str(label_ind)  +'.log'
    print(logname)
    log_file = open(logname, 'w')
    stdout_backup = sys.stdout

    log_true = 0                      # 1 means outputting to the log file, 0 means outputting to the screen #todo
    EPOCH = 2 #10                       # the epoch for parameter pretrain
    BATCH_SIZE = 50
    LR = 0.0001 #0.001

    Neuron_concepts_n = Yconcept_train[0].shape[0]

    #############the structure of SRAE
    NN_input = X_train.shape[1]
    NN_middle1 = 800
    NN_middle2 = 100
    if (useXNNOrNOT==1): #this is the only place where we use the parameter useXNNOrNOT
        if Neuron_concepts_n > Neuron_n: # Neuron_n = 5. We take greater of the two numbers when XNN==1
           NN_middle3 = Neuron_concepts_n
        else:
            NN_middle3 = Neuron_n
    else:
        NN_middle3= Neuron_concepts_n # we take number of concepts when XNN==0
    NN_output = len(label_ind) # 1 (just the logit)

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("---------------- X_train.shape: {}".format(X_train.shape))
    print("---------------- X_test.shape: {}".format(X_test.shape))
    print("---------------- GY_train.shape: {}".format(GY_train.shape))
    print("---------------- GY_test.shape: {}".format(GY_test.shape))
    print("---------------- Yconcept_train.shape: {}".format(Yconcept_train.shape))
    print("---------------- Yconcept_test.shape: {}".format(Yconcept_test.shape))
    #print("---------------- Neuron_concepts_n: {}".format(Neuron_concepts_n))
    print("---------------- MissingLabel_Train.shape: {}".format(len(MissingLabel_Train)))
    print("---------------- MissingLabel_Test.shape: {}".format(len(MissingLabel_Test)))
    print("---------------- autoParam: {}".format(autoParam))
    print("---------------- target_path: {}".format(target_path))
    #print(X_train)
    #NZ_X = X_train[0:100,:]
    #NZ_index = NZ_X.nonzero()
    #NZ_X2 = NZ_X[NZ_index]
    #print(NZ_X2.shape)

    ##################################################################################################################

    # load the SRAE model obtained at the end of XNN training
    SRAE_model_path = target_path + "SRAE_" + str(label_ind) + ".pth"
    SRAE = torch.load(SRAE_model_path, map_location=lambda storage, loc: storage, pickle_module=pickle)  # model trained in GPU could be deployed in CPU machine like this!

    # fresh model
    StackAE = autoencoder.SRAE(NN_input, NN_middle1, NN_middle2, NN_middle3, NN_output)

    #populate fresh model for fine tuning
    # StackAE.encoder1[0].weight.data.copy_(SRAE.encoder1[0].weight.data.cpu())
    # StackAE.encoder1[0].bias.data.copy_(SRAE.encoder1[0].bias.data.cpu())
    # StackAE.encoder2[0].weight.data.copy_(SRAE.encoder2[0].weight.data.cpu())
    # StackAE.encoder2[0].bias.data.copy_(SRAE.encoder2[0].bias.data.cpu())
    #StackAE.encoder3[0].weight.data.copy_(SRAE.encoder3[0].weight.data.cpu().numpy())
    #StackAE.encoder3[0].bias.data.copy_(SRAE.encoder3[0].bias.data.cpu().numpy())

    # StackAE.decoder1[0].weight.data.copy_(SRAE.decoder1[0].weight.data.cpu())
    # StackAE.decoder1[0].bias.data.copy_(SRAE.decoder1[0].bias.data.cpu())
    # StackAE.decoder2[0].weight.data.copy_(SRAE.decoder2[0].weight.data.cpu())
    # StackAE.decoder2[0].bias.data.copy_(SRAE.decoder2[0].bias.data.cpu())
    #StackAE.decoder3[0].weight.data.copy_(SRAE.decoder3[0].weight.data.cpu().numpy())
    #StackAE.decoder3[0].bias.data.copy_(SRAE.decoder3[0].bias.data.cpu().numpy())

    #StackAE.classifier.weight.data.copy_(SRAE.classifier.weight.data.cpu())
    #StackAE.classifier.bias.data.copy_(SRAE.classifier.bias.data.cpu())

    print(('Stacked Autoencoder: ', StackAE))

    trainX = torch.from_numpy(X_train).float()
    testX = torch.from_numpy(X_test).float()

    train_Gy = torch.from_numpy(GY_train).unsqueeze(1).float() # why unsqueeze ?
    test_Gy = torch.from_numpy(GY_test).unsqueeze(1).float()

    parameters = autoParam#[0, 0, 1, 10, 0.5, 500]    # #todo
    #Only use stacked autoencoder to perform pretrain
    #sparseR = parameters[0], Qq = parameters[1], alpha_pred = parameters[2], alpha_recons = parameters[3], alpha_pull_away = parameters[4]
    #sparseR = 0, using traditional reconstruction loss; sparseR = 1, using sparse reconstruction loss;
    #Qq is the parameter q for the log penalty;
    #alpha_pred is the weight for prediction loss;
    #alpha_recons is the weight for reconstruction loss;
    #alpha_pull_away is the weight for pull-away term.


    ####################################################################################################################

    SRAE = StackAE
    print(('Stacked Autoencoder: ', SRAE))


    #parametersSRAE = [1, 100, 1, 100, 0.5]  #Sparse reconstruction autoencoder
    parametersSRAE = autoParam
    # sparseR = parameters[0], Qq = parameters[1], alpha_pred = parameters[2], alpha_recons = parameters[3], alpha_pull_away = parameters[4]
    # sparseR = 0, using traditional reconstruction loss; sparseR = 1, using sparse reconstruction loss;
    # Qq is the parameter q for the log penalty;
    # alpha_pred is the weight for prediction loss;
    # alpha_recons is the weight for reconstruction loss;
    # alpha_pull_away is the weight for pull-away term.



    print('waiting...')
    if log_true == 1:    # if log_true == 1, print all the messages in the log file.
        sys.stdout = log_file



    SRAE, MeanActivWeight, CorrActivWeight, predict_loss_train = autoencoder.SRAE_Optim_retrained(conceptLoss_1allData_0nonMissingData,
        trainX, train_Gy, testX, test_Gy,
        Yconcept_train, Yconcept_test,
        MissingLabel_Train, MissingLabel_Test,
        SRAE, parametersSRAE, BatchSize=BATCH_SIZE,
        Lr=LR, EPOCH=EPOCH, useGPU=useGPU, PT_inverse=PT_inverse)
        #Lr=LR, EPOCH=EPOCH+120, useGPU=useGPU, PT_inverse=PT_inverse)

########################################
    SRAE.eval()
    if useGPU==1:
        Ltest_x = Variable(testX).cuda()
        Ltrain_x = Variable(trainX).cuda()

    else:
        Ltest_x = Variable(testX)
        Ltrain_x = Variable(trainX)
    (trainX_encoder, trainX_decoder, trainY_pred) = SRAE(Ltrain_x)
    (testX_encoder, testX_decoder, testY_pred) = SRAE(Ltest_x)

    trainY_pred_np = trainY_pred.data.cpu().numpy().copy()
    testY_pred_np = testY_pred.data.cpu().numpy().copy()



    scio.savemat(target_path + 'Prediction' + str(label_ind) + '.mat',
                 mdict={'trainY_pred': trainY_pred_np, 'train_Gy': GY_train, 'testY_pred': testY_pred_np, 'test_Gy': GY_test}, oned_as='column')


    trainX_encoder_np = trainX_encoder.data.cpu().numpy().copy()
    testX_encoder_np = testX_encoder.data.cpu().numpy().copy()
    #
    Yconcept_train_np = Yconcept_train.numpy().copy()
    Yconcept_test_np = Yconcept_test.numpy().copy()

    scio.savemat(target_path + 'newPrediction_Concept' + str(label_ind) + '1.mat',
                 mdict={'trainY_pred': trainY_pred_np, 'train_Gy': GY_train, 'testY_pred': testY_pred_np, 'test_Gy': GY_test,
                        'trainX_encoder': trainX_encoder_np, 'testX_encoder': testX_encoder_np,
                        'train_index': train_index, 'test_index': test_index,
                        'Yconcept_train': Yconcept_train_np, 'Yconcept_test': Yconcept_test_np}, oned_as='column')

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    correct = 0
    total = 0
    tmp_trainX_encoder_np = trainX_encoder_np[:,0:Yconcept_train_np.shape[1]]
    concept_predictions = np.power((1+np.exp(-1*tmp_trainX_encoder_np)), -1)
    threshold_indices = concept_predictions > 0.5
    concept_predictions *= 0.0
    concept_predictions[threshold_indices] = 1.0
    for i in range(Yconcept_train_np.shape[0]):
        gt_concept_class = np.argmax(Yconcept_train_np[i])
        pred_concept_class = np.argmax(concept_predictions[i])
        if Yconcept_train_np[i][gt_concept_class] == 0:
            continue
        else:
            total += sum(Yconcept_train_np[i])
            correct += np.dot(Yconcept_train_np[i], concept_predictions[i])
    concept_accuracy_train = correct / float(total)
    print("concept_accuracy_train: {}".format(concept_accuracy_train))

    correct = 0
    total = 0
    tmp_testX_encoder_np = testX_encoder_np[:,0:Yconcept_test_np.shape[1]]
    concept_predictions = np.power((1+np.exp(-1*tmp_testX_encoder_np)), -1)
    threshold_indices = concept_predictions > 0.5
    concept_predictions *= 0.0
    concept_predictions[threshold_indices] = 1.0
    for i in range(Yconcept_test_np.shape[0]):
        gt_concept_class = np.argmax(Yconcept_test_np[i])
        pred_concept_class = np.argmax(concept_predictions[i])
        if Yconcept_test_np[i][gt_concept_class] == 0:
            continue
        else:
            total += sum(Yconcept_test_np[i])
            correct += np.dot(Yconcept_test_np[i], concept_predictions[i])
    concept_accuracy_test = correct / float(total)
    print("concept_accuracy_test: {}".format(concept_accuracy_test))



    ###################################################################################################################

    ####################################################################################################################
    #save the model

    modelname = target_path+'SRAE_'+str(label_ind)+'.pth'
    torch.save(SRAE, modelname)

    SRAE_weights = dict()
    SRAE_weights['W1'] = SRAE.encoder1[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b1'] = SRAE.encoder1[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W2'] = SRAE.encoder2[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b2'] = SRAE.encoder2[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W3'] = SRAE.encoder3[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b3'] = SRAE.encoder3[0].bias.data.cpu().numpy().copy()
    SRAE_weights['Wo'] = SRAE.classifier.weight.data.cpu().numpy().copy()
    SRAE_weights['bo'] = SRAE.classifier.bias.data.cpu().numpy().copy()

    SRAE_weights['W-3'] = SRAE.decoder3[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b-3'] = SRAE.decoder3[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W-2'] = SRAE.decoder2[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b-2'] = SRAE.decoder2[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W-1'] = SRAE.decoder1[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b-1'] = SRAE.decoder1[0].bias.data.cpu().numpy().copy()


    scio.savemat(target_path + 'SRAEweight_'+str(label_ind)+'.mat', mdict={'SRAE_weights': SRAE_weights}, oned_as='column')

    log_file.close()
    sys.stdout = stdout_backup

    return (MeanActivWeight, CorrActivWeight, predict_loss_train)
    print('Finished!')



def SparseRAE_multi_retrainedModel(useXNNOrNOT,conceptLoss_1allData_0nonMissingData, X_train, X_test, GY_train, GY_test, Yconcept_train, Yconcept_test,
                                   MissingLabel_Train, MissingLabel_Test,
                                   test_index, train_index,target_path, label_ind, Neuron_n, autoParam, useGPU =1, PT_inverse=1):
    Yconcept_train_np = Yconcept_train.numpy().copy()
    Yconcept_test_np = Yconcept_test.numpy().copy()
    scio.savemat(target_path + 'Prediction_Concept_PREEEEE_' + str(label_ind) + '1.mat',
                 mdict={'train_Gy': GY_train, 'test_Gy': GY_test,
                        'train_index': train_index, 'test_index': test_index,
                        'Yconcept_train': Yconcept_train_np, 'Yconcept_test': Yconcept_test_np}, oned_as='column')
    scio.savemat(target_path + 'Prediction_Concept_indices' + str(label_ind) + '.mat',
                 mdict={ 'train_index': train_index, 'test_index': test_index,}, oned_as='column')

    logname = target_path + 'LOG_'+ str(label_ind)  +'.log'
    print(logname)
    log_file = open(logname, 'w')
    stdout_backup = sys.stdout

    log_true = 0                      # 1 means outputting to the log file, 0 means outputting to the screen #todo
    EPOCH = 2 #10                       # the epoch for parameter pretrain
    BATCH_SIZE = 50
    LR = 0.0001 #0.001

    Neuron_concepts_n = Yconcept_train[0].shape[0]

    #############the structure of SRAE
    NN_input = X_train.shape[1]
    NN_middle1 = 800
    NN_middle2 = 100
    if (useXNNOrNOT==1):
        if Neuron_concepts_n > Neuron_n:
           NN_middle3 = Neuron_concepts_n
        else:
            NN_middle3 = Neuron_n
    else:
        NN_middle3= Neuron_concepts_n
    NN_output = len(label_ind)

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("---------------- X_train.shape: {}".format(X_train.shape))
    print("---------------- X_test.shape: {}".format(X_test.shape))
    print("---------------- GY_train.shape: {}".format(GY_train.shape))
    print("---------------- GY_test.shape: {}".format(GY_test.shape))
    print("---------------- Yconcept_train.shape: {}".format(Yconcept_train.shape))
    print("---------------- Yconcept_test.shape: {}".format(Yconcept_test.shape))
    print("---------------- Neuron_concepts_n: {}".format(Neuron_concepts_n))
    print("---------------- MissingLabel_Train.shape: {}".format(len(MissingLabel_Train)))
    print("---------------- MissingLabel_Test.shape: {}".format(len(MissingLabel_Test)))
    print("---------------- autoParam: {}".format(autoParam))
    #print(X_train)
    #NZ_X = X_train[0:100,:]
    #NZ_index = NZ_X.nonzero()
    #NZ_X2 = NZ_X[NZ_index]
    #print(NZ_X2.shape)

    ##################################################################################################################
    # pretrain the first layer of SRAE
    AE1 = autoencoder.AE(NN_input, NN_middle1)
    print(('Autoencoder 1: ', AE1))


    train_x1 = torch.from_numpy(X_train).float()
    test_x1 = torch.from_numpy(X_test).float()

    #print("--------- train_x1.shape: {}".format(train_x1.shape))
    #print("--------- test_x1.shape: {}".format(test_x1.shape))


    AE1 = autoencoder.AutoencoderOptim(train_x1, test_x1, AE1, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU)

    if useGPU==1:
        AE1 = AE1.cuda()
        (train_x2, trainDe) = AE1(Variable(train_x1).cuda())
        train_x2 = train_x2.cpu().data
        print(train_x2)
        (test_x2, testDe) = AE1(Variable(test_x1).cuda())
        test_x2 = test_x2.cpu().data
    else:
        (train_x2, trainDe) = AE1(Variable(train_x1))
        train_x2 = train_x2.data
        print(train_x2)
        (test_x2, testDe) = AE1(Variable(test_x1))
        test_x2 = test_x2.data

    #print("--------- train_x2.shape: {}".format(train_x2.shape))
    #print("--------- test_x2.shape: {}".format(test_x2.shape))


    SAE_weights = dict()  # save the weights
    if useGPU==1:
        SAE_weights['W1'] = AE1.encoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b1'] = AE1.encoder[0].bias.data.cpu().numpy().copy()
        SAE_weights['W-1'] = AE1.decoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b-1'] = AE1.decoder[0].bias.data.cpu().numpy().copy()
    else:
        SAE_weights['W1'] = AE1.encoder[0].weight.data.numpy().copy()
        SAE_weights['b1'] = AE1.encoder[0].bias.data.numpy().copy()
        SAE_weights['W-1'] = AE1.decoder[0].weight.data.numpy().copy()
        SAE_weights['b-1'] = AE1.decoder[0].bias.data.numpy().copy()


    ####################################################################################################################
    # pretrain the second layer of SRAE
    AE2 = autoencoder.AE(NN_middle1, NN_middle2)
    print(('Autoencoder 2: ', AE2))



    AE2 = autoencoder.AutoencoderOptim(train_x2, test_x2, AE2, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU)

    if useGPU==1:
        AE2 = AE2.cuda()
        (train_x3, trainDe) = AE2(Variable(train_x2).cuda())
        train_x3 = train_x3.cpu().data
        (test_x3, trainDe) = AE2(Variable(test_x2).cuda())
        test_x3 = test_x3.cpu().data
    else:
        (train_x3, trainDe) = AE2(Variable(train_x2))
        train_x3 = train_x3.data
        (test_x3, trainDe) = AE2(Variable(test_x2))
        test_x3 = test_x3.data

    #print("--------- train_x3.shape: {}".format(train_x3.shape))
    #print("--------- test_x3.shape: {}".format(test_x3.shape))

    if useGPU == 1:
        SAE_weights['W2'] = AE2.encoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b2'] = AE2.encoder[0].bias.data.cpu().numpy().copy()
        SAE_weights['W-2'] = AE2.decoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b-2'] = AE2.decoder[0].bias.data.cpu().numpy().copy()

    else:
        SAE_weights['W2'] = AE2.encoder[0].weight.data.numpy().copy()
        SAE_weights['b2'] = AE2.encoder[0].bias.data.numpy().copy()
        SAE_weights['W-2'] = AE2.decoder[0].weight.data.numpy().copy()
        SAE_weights['b-2'] = AE2.decoder[0].bias.data.numpy().copy()


    ####################################################################################################################
    # pretrain the third layer of SRAE

    AE3 = autoencoder.AE(NN_middle2, NN_middle3)
    print(('Autoencoder 3: ', AE3))



    AE3 = autoencoder.AutoencoderOptim(train_x3, test_x3, AE3, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU)

    if useGPU==1:
        AE3 = AE3.cuda()
        (train_x4, trainDe) = AE3(Variable(train_x3).cuda())
        train_x4 = train_x4.cpu().data
        (test_x4, trainDe) = AE3(Variable(test_x3).cuda())
        test_x4 = test_x4.cpu().data

    else:
        (train_x4, trainDe) = AE3(Variable(train_x3))
        train_x4 = train_x4.data
        (test_x4, trainDe) = AE3(Variable(test_x3))
        test_x4 = test_x4.data

    #print("--------- train_x4.shape: {}".format(train_x4.shape))
    #print("--------- test_x4.shape: {}".format(test_x4.shape))

    if useGPU == 1:
        SAE_weights['W3'] = AE3.encoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b3'] = AE3.encoder[0].bias.data.cpu().numpy().copy()
        SAE_weights['W-3'] = AE3.decoder[0].weight.data.cpu().numpy().copy()
        SAE_weights['b-3'] = AE3.decoder[0].bias.data.cpu().numpy().copy()
    else:
        SAE_weights['W3'] = AE3.encoder[0].weight.data.numpy().copy()
        SAE_weights['b3'] = AE3.encoder[0].bias.data.numpy().copy()
        SAE_weights['W-3'] = AE3.decoder[0].weight.data.numpy().copy()
        SAE_weights['b-3'] = AE3.decoder[0].bias.data.numpy().copy()

    ####################################################################################################################
    # pretrain the last layer of SRAE

    LR4 = autoencoder.LR(NN_middle3, NN_output)
    print(('Linear Regression: ', LR4))

    train_Gy = torch.from_numpy(GY_train).unsqueeze(1).float()
    test_Gy = torch.from_numpy(GY_test).unsqueeze(1).float()

    LR4 = autoencoder.LR_Optim(train_x4, train_Gy, test_x4, test_Gy, LR4, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU)

    if useGPU==1:
        SAE_weights['Wout'] = LR4.regression.weight.data.cpu().numpy().copy()
        SAE_weights['bout'] = LR4.regression.bias.data.cpu().numpy().copy()
    else:
        SAE_weights['Wout'] = LR4.regression.weight.data.numpy().copy()
        SAE_weights['bout'] = LR4.regression.bias.data.numpy().copy()

    ####################################################################################################################
    # pretrain the whole network of SRAE

    StackAE = autoencoder.SRAE(NN_input, NN_middle1, NN_middle2, NN_middle3, NN_output)
    print(('Stacked Autoencoder: ', StackAE))

    trainX = torch.from_numpy(X_train).float()
    testX = torch.from_numpy(X_test).float()

    parameters = autoParam#[0, 0, 1, 10, 0.5, 500]    # #todo
    #Only use stacked autoencoder to perform pretrain
    #sparseR = parameters[0], Qq = parameters[1], alpha_pred = parameters[2], alpha_recons = parameters[3], alpha_pull_away = parameters[4]
    #sparseR = 0, using traditional reconstruction loss; sparseR = 1, using sparse reconstruction loss;
    #Qq is the parameter q for the log penalty;
    #alpha_pred is the weight for prediction loss;
    #alpha_recons is the weight for reconstruction loss;
    #alpha_pull_away is the weight for pull-away term.




    ###########weights initialization: - Determine Pre-training or not

    StackAE.encoder1[0].weight.data.copy_(torch.from_numpy(SAE_weights['W1']))
    StackAE.encoder1[0].bias.data.copy_(torch.from_numpy(SAE_weights['b1']))
    StackAE.encoder2[0].weight.data.copy_(torch.from_numpy(SAE_weights['W2']))
    StackAE.encoder2[0].bias.data.copy_(torch.from_numpy(SAE_weights['b2']))
    StackAE.encoder3[0].weight.data.copy_(torch.from_numpy(SAE_weights['W3']))
    StackAE.encoder3[0].bias.data.copy_(torch.from_numpy(SAE_weights['b3']))

    StackAE.decoder3[0].weight.data.copy_(torch.from_numpy(SAE_weights['W-3']))
    StackAE.decoder3[0].bias.data.copy_(torch.from_numpy(SAE_weights['b-3']))
    StackAE.decoder2[0].weight.data.copy_(torch.from_numpy(SAE_weights['W-2']))
    StackAE.decoder2[0].bias.data.copy_(torch.from_numpy(SAE_weights['b-2']))
    StackAE.decoder1[0].weight.data.copy_(torch.from_numpy(SAE_weights['W-1']))
    StackAE.decoder1[0].bias.data.copy_(torch.from_numpy(SAE_weights['b-1']))

    StackAE.classifier.weight.data.copy_(torch.from_numpy(SAE_weights['Wout']))
    StackAE.classifier.bias.data.copy_(torch.from_numpy(SAE_weights['bout']))

    ####################weights initialization end


    # StackAE, MeanActivWeight0, CorrActivWeight0, predict_loss_train0 = autoencoder.SRAE_Optim_retrained(conceptLoss_1allData_0nonMissingData,
    #     trainX, train_Gy, testX, test_Gy,
    #     Yconcept_train, Yconcept_test, MissingLabel_Train, MissingLabel_Test,
    #     StackAE, parameters, BatchSize=BATCH_SIZE, Lr=LR, EPOCH=EPOCH, useGPU=useGPU, PT_inverse=PT_inverse)


    ####################################################################################################################

    SRAE = StackAE
    print(('Stacked Autoencoder: ', SRAE))


    #parametersSRAE = [1, 100, 1, 100, 0.5]  #Sparse reconstruction autoencoder
    parametersSRAE = autoParam
    # sparseR = parameters[0], Qq = parameters[1], alpha_pred = parameters[2], alpha_recons = parameters[3], alpha_pull_away = parameters[4]
    # sparseR = 0, using traditional reconstruction loss; sparseR = 1, using sparse reconstruction loss;
    # Qq is the parameter q for the log penalty;
    # alpha_pred is the weight for prediction loss;
    # alpha_recons is the weight for reconstruction loss;
    # alpha_pull_away is the weight for pull-away term.



    print('waiting...')
    if log_true == 1:    # if log_true == 1, print all the messages in the log file.
        sys.stdout = log_file



    SRAE, MeanActivWeight, CorrActivWeight, predict_loss_train = autoencoder.SRAE_Optim_retrained(conceptLoss_1allData_0nonMissingData,
        trainX, train_Gy, testX, test_Gy,
        Yconcept_train, Yconcept_test,
        MissingLabel_Train, MissingLabel_Test,
        SRAE, parametersSRAE, BatchSize=BATCH_SIZE,
        Lr=LR, EPOCH=EPOCH, useGPU=useGPU, PT_inverse=PT_inverse)
        #Lr=LR, EPOCH=EPOCH+120, useGPU=useGPU, PT_inverse=PT_inverse)

########################################
    SRAE.eval()
    if useGPU==1:
        Ltest_x = Variable(testX).cuda()
        Ltrain_x = Variable(trainX).cuda()

    else:
        Ltest_x = Variable(testX)
        Ltrain_x = Variable(trainX)
    (trainX_encoder, trainX_decoder, trainY_pred) = SRAE(Ltrain_x)
    (testX_encoder, testX_decoder, testY_pred) = SRAE(Ltest_x)

    trainY_pred_np = trainY_pred.data.cpu().numpy().copy()
    testY_pred_np = testY_pred.data.cpu().numpy().copy()



    scio.savemat(target_path + 'Prediction' + str(label_ind) + '.mat',
                 mdict={'trainY_pred': trainY_pred_np, 'train_Gy': GY_train, 'testY_pred': testY_pred_np, 'test_Gy': GY_test}, oned_as='column')


    trainX_encoder_np = trainX_encoder.data.cpu().numpy().copy()
    testX_encoder_np = testX_encoder.data.cpu().numpy().copy()
    #
    Yconcept_train_np = Yconcept_train.numpy().copy()
    Yconcept_test_np = Yconcept_test.numpy().copy()

    scio.savemat(target_path + 'Prediction_Concept' + str(label_ind) + '1.mat',
                 mdict={'trainY_pred': trainY_pred_np, 'train_Gy': GY_train, 'testY_pred': testY_pred_np, 'test_Gy': GY_test,
                        'trainX_encoder': trainX_encoder_np, 'testX_encoder': testX_encoder_np,
                        'train_index': train_index, 'test_index': test_index,
                        'Yconcept_train': Yconcept_train_np, 'Yconcept_test': Yconcept_test_np}, oned_as='column')

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    correct = 0
    total = 0
    tmp_trainX_encoder_np = trainX_encoder_np[:,0:Yconcept_train_np.shape[1]]
    concept_predictions = np.power((1+np.exp(-1*tmp_trainX_encoder_np)), -1)
    threshold_indices = concept_predictions > 0.5
    concept_predictions *= 0.0
    concept_predictions[threshold_indices] = 1.0
    for i in range(Yconcept_train_np.shape[0]):
        gt_concept_class = np.argmax(Yconcept_train_np[i])
        pred_concept_class = np.argmax(concept_predictions[i])
        if Yconcept_train_np[i][gt_concept_class] == 0:
            continue
        else:
            total += sum(Yconcept_train_np[i])
            correct += np.dot(Yconcept_train_np[i], concept_predictions[i])
    concept_accuracy_train = correct / float(total)
    print("concept_accuracy_train: {}".format(concept_accuracy_train))

    correct = 0
    total = 0
    tmp_testX_encoder_np = testX_encoder_np[:,0:Yconcept_test_np.shape[1]]
    concept_predictions = np.power((1+np.exp(-1*tmp_testX_encoder_np)), -1)
    threshold_indices = concept_predictions > 0.5
    concept_predictions *= 0.0
    concept_predictions[threshold_indices] = 1.0
    for i in range(Yconcept_test_np.shape[0]):
        gt_concept_class = np.argmax(Yconcept_test_np[i])
        pred_concept_class = np.argmax(concept_predictions[i])
        if Yconcept_test_np[i][gt_concept_class] == 0:
            continue
        else:
            total += sum(Yconcept_test_np[i])
            correct += np.dot(Yconcept_test_np[i], concept_predictions[i])
    concept_accuracy_test = correct / float(total)
    print("concept_accuracy_test: {}".format(concept_accuracy_test))



    ###################################################################################################################

    ####################################################################################################################
    #save the model

    modelname = target_path+'SRAE_'+str(label_ind)+'.pth'
    torch.save(SRAE, modelname)

    SRAE_weights = dict()
    SRAE_weights['W1'] = SRAE.encoder1[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b1'] = SRAE.encoder1[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W2'] = SRAE.encoder2[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b2'] = SRAE.encoder2[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W3'] = SRAE.encoder3[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b3'] = SRAE.encoder3[0].bias.data.cpu().numpy().copy()
    SRAE_weights['Wo'] = SRAE.classifier.weight.data.cpu().numpy().copy()
    SRAE_weights['bo'] = SRAE.classifier.bias.data.cpu().numpy().copy()

    SRAE_weights['W-3'] = SRAE.decoder3[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b-3'] = SRAE.decoder3[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W-2'] = SRAE.decoder2[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b-2'] = SRAE.decoder2[0].bias.data.cpu().numpy().copy()
    SRAE_weights['W-1'] = SRAE.decoder1[0].weight.data.cpu().numpy().copy()
    SRAE_weights['b-1'] = SRAE.decoder1[0].bias.data.cpu().numpy().copy()


    scio.savemat(target_path + 'SRAEweight_'+str(label_ind)+'.mat', mdict={'SRAE_weights': SRAE_weights}, oned_as='column')

    log_file.close()
    sys.stdout = stdout_backup

    return (MeanActivWeight, CorrActivWeight, predict_loss_train)
    print('Finished!')

####################################################################################################################


def main():
    CategoryListFile = 'Data/CUB200_Multi_csvnamelist.csv'
    DataFile = '../../../../DataSetforXAI/Results_newEBP/Model/TrainData_CUBALL.mat'

    print(CategoryListFile, DataFile)
    X_train, X_test, GY_train, GY_test, label_ind, test_index_pos = Generate_training(CategoryListFile, DataFile)

    print(len(label_ind))
    print(str(label_ind))
    logname = 'LOG_' + str(label_ind) + '_' + '.log'
    print(logname)
    # (e.g., street 319, highway 175, bathroom 45, bedroom 52, building_facade 67, dining_room 121, kitchen 203)
    Neuron_n = 12  # the number of x features #todo
    PT_inverse = 1  # todo
    num_iter = 5  # todo

    ###########################################################train the SRAE


    out_path = './Results/' + str(label_ind) + '/'   # output path

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    scio.savemat(out_path + 'test_index_pos' + str(label_ind) + '.mat', mdict={'test_index_pos': test_index_pos}, oned_as='column')

    if PT_inverse == 0:
        autoParam = [1, 100, 1, 100, 0.5]  # pt_loss = util.PT_loss_true #Pull away term along features
    elif PT_inverse == 1:
        autoParam = [1, 100, 50, 200, 10]  # pt_loss = util.PT_loss #Pull away term along images
    # sparseR = parameters[0], Qq = parameters[1], alpha_pred = parameters[2], alpha_recons = parameters[3], alpha_pull_away = parameters[4]
    # sparseR = 0, using traditional reconstruction loss; sparseR = 1, using sparse reconstruction loss;
    # Qq is the parameter q for the log penalty;
    # alpha_pred is the weight for prediction loss;
    # alpha_recons is the weight for reconstruction loss;
    # alpha_pull_away is the weight for pull-away term.




    SparseRAE_multi(X_train, X_test, GY_train, GY_test, out_path, label_ind, Neuron_n, autoParam, useGPU=1,
                    PT_inverse=1)


if __name__ == '__main__':
    main()
