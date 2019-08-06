#coding=utf-8

# autoencoder
# python Version: python3.6
# by Zhongang Qi (qiz@oregonstate.edu)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import util
import numpy as np


def adjust_lr(optimizer, epoch, init_lr):
    ########################
    # adjust learning rate

    # Parameters:
    # -------------
    # optimizer: optimizer
    # epoch: epoch
    # init_lr: the initial learning rate
    ####################################################
    lr = init_lr * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class CAE(nn.Module):
    ########################
    # fully convolutional autoencoder
    ####################################################
    def __init__(self, inchannels, outchannels, kernelsize=1, stri=1, pad=0):
        super(CAE, self).__init__()
        self.convC1 = nn.Sequential(
                     nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=kernelsize,
                               stride=stri, padding=pad),   #fully convolutional encoder
                     nn.ReLU()
                     )
        self.convCN1 = nn.Sequential(
                     nn.Conv2d(outchannels, inchannels, kernelsize, stri, pad),  #fully convolutional decoder
                     nn.ReLU()
                     )

    def forward(self, x):
        encoder = self.convC1(x)
        decoder = self.convCN1(encoder)
        return (encoder, decoder)




class AE(nn.Module):
    ########################
    # fully connected autoencoder
    ####################################################
    def __init__(self, inputs, outputs):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
                     nn.Linear(inputs,outputs),   #fully connected encoder
                     nn.ReLU()
                     )
        self.decoder = nn.Sequential(
                     nn.Linear(outputs,inputs),  #fully connedted decoder
                     nn.ReLU()
                     )

    def forward(self, x):
        AEencoder = self.encoder(x)
        AEdecoder = self.decoder(AEencoder)
        return (AEencoder, AEdecoder)



class LR(nn.Module):
    ########################
    #Linear Regression
    ####################################################
    def __init__(self, inputs, outputs):
        super(LR, self).__init__()
        self.regression = nn.Linear(inputs,outputs)


    def forward(self, x):
        y = self.regression(x)
        return y



class SRAE(nn.Module):
    ########################
    # Sparse Reconstruction autoencoder (SRAE)
    ####################################################
    def __init__(self, input_num, middle1, middle2, middle3, output_num):
        super(SRAE, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(input_num, middle1),   # fully connected encoder
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(middle1, middle2),  # fully connected encoder
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Linear(middle2, middle3),  # fully connected encoder
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.Linear(middle3, middle2),  # fully connected decoder
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(middle2, middle1),  # fully connected decoder
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(middle1, input_num),  # fully connected decoder
            nn.ReLU()
        )
        self.classifier = nn.Linear(middle3, output_num)  # prediction

        self.dropfc_p2 = nn.Dropout(p=0.02)
        self.dropfc_p1 = nn.Dropout(p=0.05)


    def forward(self, x):
        x = self.encoder1(x)
        x = self.dropfc_p2(x)   # whether to use the dropout or not #todo
        x = self.encoder2(x)
        #x = self.dropfc_p2(x)   # whether to use the dropout or not #todo
        Xencoder = self.encoder3(x)
        #Xencoder = self.dropfc_p1(Xencoder)   # whether to use the dropout or not #todo
        y = self.classifier(Xencoder)

        #FIX - Vivswan
        y = y.squeeze()

        x = self.decoder3(Xencoder)
        x = self.decoder2(x)
        Xdecoder = self.decoder1(x)

        return (Xencoder, Xdecoder, y)



def AutoencoderOptim(TrainX, TestX, AE_model, BatchSize=50, Lr=0.001, EPOCH=1, useGPU=1):
    ########################
    # Optimization for autoencoder

    # Parameters:
    # -------------
    # TrainX: training data_X
    # TestX: testing data_X
    # AE_model: the autoencoder model
    # BatchSize: batch size
    # Lr: learning rate
    # EPOCH: epoch
    # useGPU: use GPU or not
    ####################################################
    trainData = Data.TensorDataset(TrainX, TrainX)
    train_loader = Data.DataLoader(trainData, batch_size=BatchSize, shuffle=True)

    if useGPU==1:
        AE_model=AE_model.cuda()
        test_x = Variable(TestX).cuda()
    else:
        test_x = Variable(TestX)

    test_y = test_x

    optimizer = torch.optim.Adam(AE_model.parameters(), lr=Lr, weight_decay=0)    #weight_decay is L2 norm
    loss_function = nn.MSELoss()

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            if useGPU==1:
                b_x = Variable(x).cuda()
                b_y = Variable(y).cuda()
            else:
                b_x = Variable(x)
                b_y = Variable(y)

            #forward
            (X_encoder, X_decoder) = AE_model(b_x)
            loss = loss_function(X_decoder, b_y)

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                (test_encoder, test_decoder) = AE_model(test_x)
                testloss = loss_function(test_decoder, test_y)
                print(('Epoch:', epoch, '|Step:', step))
                      # '|train loss:%.4f'%loss.data[0], '|test accuracy:%.4f'%testloss.data[0]))

    return AE_model


def LR_Optim(TrainX, TrainY, TestX, TestY, LR_model, BatchSize=50, Lr=0.001, EPOCH=1, useGPU=1):
    ########################
    # Optimization for Linear Regression

    # Parameters:
    # -------------
    # TrainX: training data_X
    # TrainY: training data_Y
    # TestX: testing data_X
    # TestY: testing data_Y
    # LR_model: the linear regression model
    # BatchSize: batch size
    # Lr: learning rate
    # EPOCH: epoch
    # useGPU: use GPU or not
    ####################################################

    trainData = Data.TensorDataset(TrainX, TrainY)
    train_loader = Data.DataLoader(trainData, batch_size=BatchSize, shuffle=True)

    if useGPU==1:
        LR_model=LR_model.cuda()
        test_x = Variable(TestX).cuda()
        test_y = Variable(TestY).cuda()
    else:
        test_x = Variable(TestX)
        test_y = Variable(TestY)


    optimizer = torch.optim.Adam(LR_model.parameters(), lr=Lr, weight_decay=0)    #weight_decay is L2 norm
    loss_function = nn.MSELoss()

    for epoch in range(EPOCH):

        for step, (x, y) in enumerate(train_loader):
            if useGPU==1:
                b_x = Variable(x).cuda()
                b_y = Variable(y).cuda()
            else:
                b_x = Variable(x)
                b_y = Variable(y)

            #forward
            b_y_pred = LR_model(b_x)
            loss = loss_function(b_y_pred, b_y)

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                test_y_pred = LR_model(test_x)
                testloss = loss_function(test_y_pred, test_y)
                print(('Epoch:', epoch, '|Step:', step,
                       '|train loss:%.4f'%loss.data, '|test accuracy:%.4f'%testloss.data))
                       # '|train loss:%.4f'%loss.data[0], '|test accuracy:%.4f'%testloss.data[0]))

    return LR_model

def SRAE_Optim_retrained_AllDataLossFunction(TrainX, TrainY, TestX, TestY, Yconcept_train, Yconcept_test,SRAE_model, parameters, BatchSize=50, Lr=0.001, EPOCH=1, useGPU=1, PT_inverse=1):
    ########################
    # Optimization for Sparse reconstruction autoencoder (SRAE)

    # Parameters:
    # -------------
    # TrainX: training data_X
    # TrainY: training data_Y
    # TestX: testing data_X
    # TestY: testing data_Y
    # SRAE_model: the sparse reconstruction autoencoder model
    # parameters: the parameters for different loss
    # BatchSize: batch size
    # Lr: learning rate
    # EPOCH: epoch
    # useGPU: use GPU or not
    # PT_inverse: 1 means pull-away term along the images; 0 means pull-away term along the features
    ####################################################

    # sparseR = 0, using traditional reconstruction loss; sparseR = 1, using sparse reconstruction loss:
    sparseR = parameters[0]
    # Qq is the parameter q for the log penalty:
    Qq = parameters[1]
    # alpha_pred is the weight for prediction loss:
    alpha_pred = parameters[2]
    # alpha_recons is the weight for reconstruction loss:
    alpha_recons = parameters[3]
    # alpha_pull_away is the weight for pull-away term:
    alpha_pt_loss = parameters[4]

    alpha_concepts_loss = parameters[5]



    number_non_multi=0
    number_multi=0
    for row in range (Yconcept_train.size(0)):
        if ( (Yconcept_train[row,:]==0).all()) :
            number_non_multi=number_non_multi+1
        else:
            number_multi=number_multi+1


    ###################### new Train for mulit-labled Data  #############################
    TrainX_multi = torch.zeros(number_multi, TrainX.size(1))#torch.tensor([])
    TrainX_non_multi = torch.zeros(number_non_multi, TrainX.size(1))#torch.tensor([])

    TrainY_multi = torch.zeros(number_multi,TrainY.size(1)+ Yconcept_train.size(1))#torch.tensor([])
    TrainY_non_multi = torch.zeros(number_non_multi,TrainY.size(1)+ Yconcept_train.size(1))#torch.tensor([])

    ####################  new Train for not labeled Data ########################
    index_multi=0
    index_non_multi=0
    for row in range (Yconcept_train.size(0)):

        allLables=torch.zeros((1,Yconcept_train.size(1)+TrainY.size(1)))
        allLables[0,0]= TrainY[row,0][0]
        for column in range (Yconcept_train.size(1)):
            allLables[0, column+1] = Yconcept_train[row, column]

        if ( (Yconcept_train[row,:]==0).all()) :
            TrainX_non_multi[index_non_multi] = TrainX[row]
            TrainY_non_multi[index_non_multi] = allLables[0]
            index_non_multi = index_non_multi + 1

        else:
            TrainX_multi[index_multi] = TrainX[row]
            TrainY_multi[index_multi] = allLables[0]
            index_multi = index_multi + 1
    #
    trainData = Data.TensorDataset(TrainX_non_multi, TrainY_non_multi)
    train_loader = Data.DataLoader(trainData, batch_size = ( round(TrainX.size(0)/number_multi )), shuffle=True)

    if useGPU==1:
        SRAE_model=SRAE_model.cuda()
        test_x = Variable(TestX).cuda()
        test_y = Variable(TestY).cuda()
        train_x = Variable(TrainX).cuda()
        train_y = Variable(TrainY).cuda()
        y_concept_test = Variable(Yconcept_test).cuda()
    else:
        test_x = Variable(TestX)
        test_y = Variable(TestY)
        train_x = Variable(TrainX)
        train_y = Variable(TrainY)
        y_concept_test = Variable(Yconcept_test)

    optimizer = torch.optim.Adam(SRAE_model.parameters(), lr=Lr, weight_decay=0.0001)      #weight_decay is L2 norm
    loss_function = nn.MSELoss()

    for epoch in range(EPOCH):

        adjust_lr(optimizer, epoch, Lr)

        for step, (x, newy) in enumerate(train_loader):
            newyTemp = torch.zeros(newy.size(0)+ 1, newy.size(1))
            newyTemp[0:newy.size(0), :]= newy
            if (step< TrainY_multi.size(0) ):
                newyTemp[newy.size(0), :]= TrainY_multi[step]
            newy= newyTemp

            xTemp = torch.zeros(x.size(0)+ 1, x.size(1))
            xTemp[0:x.size(0), :] = x
            if (step< TrainX_multi.size(0) ):
                xTemp[x.size(0), :] = TrainX_multi[step]
            x = xTemp

            y = newy[:,0]
            multi_label= newy[:,1:Yconcept_train.size(1)+1]

            if useGPU==1:
                b_x = Variable(x).cuda()
                b_y = Variable(y).cuda()
                b_multi_label = Variable(multi_label).cuda()
            else:
                b_x = Variable(x)
                b_y = Variable(y)
                b_multi_label = Variable(multi_label)

            #forward
            (b_x_encoder, b_x_decoder, b_y_pred) = SRAE_model(b_x)

            predict_loss = loss_function(b_y_pred, b_y)

            if sparseR==1:
                recons_loss = util.logpenaltyFC(b_x_decoder, b_x, Qq) # sparse reconstruction loss
            else:
                recons_loss = loss_function(b_x_decoder, b_x) # traditional reconstruction loss

            FWeight = SRAE_model.classifier.weight

            pt_loss = 0
            for Fi in range(FWeight.shape[0]):
                b_x_FW = torch.mul(b_x_encoder, FWeight[Fi,:])
                b_x_FW = torch.add(b_x_FW, 1e-8) # to avoid nan
                if PT_inverse==1:
                    pt_loss_i = util.PT_loss(b_x_FW, useGPU) # pull-away term along the images
                elif PT_inverse==0:
                    pt_loss_i = util.PT_loss_true(b_x_FW, useGPU) # pull-away term along the features

                pt_loss = pt_loss + pt_loss_i

            mulit_label_losss = nn.BCEWithLogitsLoss()

            # concepts_loss = mulit_label_losss(b_x_encoder, multi_label)
            # another way to compute concept_loss
            concept_loss1 = 0
            for indexData in range(multi_label.size(0)):
                if ( (b_multi_label[indexData,:]==0).all()) :
                    a=1# print("All zero")
                else:# if (sumLables>0).numpy() :
                    concept_loss1 = concept_loss1 + mulit_label_losss( b_x_encoder[indexData][0:multi_label.size(1)], b_multi_label[indexData])

            loss = alpha_pred * predict_loss + alpha_recons * recons_loss + alpha_pt_loss * pt_loss +  alpha_concepts_loss * concept_loss1

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1 == 0:
                SRAE_model.eval()
                (test_x_encoder,test_x_decoder,test_y_pred) = SRAE_model(test_x)

                #--------------------------------------------------------------
                number_non_multi_test = 0
                number_multi_test = 0
                for row in range (Yconcept_test.size(0)):
                    if ( (Yconcept_test[row,:]==0).all()) :
                        number_non_multi_test = number_non_multi_test + 1
                    else:
                        number_multi_test = number_multi_test + 1

                # new Train for mulit-labled Data
                TestX_multi = torch.zeros(number_multi, test_x.size(1))#torch.tensor([])
                TestX_non_multi = torch.zeros(number_non_multi, test_x.size(1))#torch.tensor([])

                TestY_multi = torch.zeros(number_multi , TestY.size(1) + Yconcept_test.size(1))#torch.tensor([])
                TestY_non_multi = torch.zeros(number_non_multi , test_y.size(1) + Yconcept_test.size(1))#torch.tensor([])


                # # new Train for not labled Data
                # index_multi_test = 0
                # index_non_multi_test = 0
                # for row in range (Yconcept_test.size(0)):
                #
                #     allLables = torch.zeros((1,Yconcept_test.size(1)+test_y.size(1)))
                #     allLables[0,0] = test_y[row,0][0]
                #     for column in range (Yconcept_test.size(1)):
                #         allLables[0, column+1] = Yconcept_test[row, column]
                #
                #     if ( (Yconcept_train[row,:]==0).all()) :
                #         TestX_non_multi[index_non_multi_test] = test_x[row]
                #         TestY_non_multi[index_non_multi_test] = allLables[0]
                #         index_non_multi_test = index_non_multi_test + 1
                #
                #     else:
                #         TestX_multi[index_multi_test] = test_x[row]
                #         TestY_multi[index_multi_test] = allLables[0]
                #         index_multi_test = index_multi_test + 1
                #---------------------------------------------
                test_predict_loss = loss_function(test_y_pred, test_y)
                if sparseR==1:
                    test_recons_loss =util.logpenaltyFC(test_x_decoder, test_x, Qq)
                else:
                    test_recons_loss = loss_function(test_x_decoder, test_x)

                FWeight = SRAE_model.classifier.weight

                test_pt_loss = 0
                for Fi in range(FWeight.shape[0]):
                    test_FW = torch.mul(test_x_encoder, FWeight[Fi,:])
                    test_FW = torch.add(test_FW, 1e-8)
                    if PT_inverse== 1:
                        test_pt_loss_i = util.PT_loss(test_FW, useGPU)
                    elif PT_inverse==0:
                        test_pt_loss_i = util.PT_loss_true(test_FW, useGPU)

                    test_pt_loss = test_pt_loss+test_pt_loss_i

                concept_loss_test = 0
                for indexData in range(y_concept_test.size(0)):
                    if ( (y_concept_test[indexData,:]==0).all()) :
                        a=1# print("All zero")
                    else:#
                        concept_loss_test = concept_loss_test +  mulit_label_losss(test_x_encoder[indexData][0:y_concept_test.size(1)], y_concept_test[indexData])

                print(('Epoch:', epoch, '|Step:', step,
                       '|train loss:%.4f' % loss.data, '|train pred loss:%.4f' % predict_loss.data,
                       '|train recons loss:%.4f' % recons_loss.data,
                       '|train PT loss:%.4f' % pt_loss.data,
                       '|train concept loss:%.4f' % concept_loss1,
                       '|test pred loss:%.4f' % test_predict_loss.data, '|test recons loss:%.4f' % test_recons_loss.data,
                       '|test PT loss:%.4f' % test_pt_loss.data,
                       '|test concept loss:%.4f' % concept_loss_test.data

                       ))
                # '|train loss:%.4f' % loss.data[0], '|train pred loss:%.4f' % predict_loss.data[0],
                # '|train recons loss:%.4f' % recons_loss.data[0],
                # '|train PT loss:%.4f' % pt_loss.data[0],
                # '|test pred loss:%.4f' % test_predict_loss.data[0], '|test recons loss:%.4f' % test_recons_loss.data[0],
                # '|test PT loss:%.4f' % test_pt_loss.data[0]))

                SRAE_model.train()


    ###################################################################################Adjust
    # compute the mean value of activation * weight for x-features
    # compute the correlation coefficient of activation * weight for x-features
    (train_x_encoder, train_x_decoder, train_y_pred) = SRAE_model(train_x)

    predict_loss_train = loss_function(train_y_pred, train_y)

    if useGPU==1:
        TrainActiv = train_x_encoder.data.cpu().numpy().copy()
        Weight_out = SRAE_model.classifier.weight.data.cpu().numpy().copy()
        predict_loss_data = predict_loss_train.data.cpu().numpy().copy()

    else:
        TrainActiv = train_x_encoder.data.numpy().copy()
        Weight_out = SRAE_model.classifier.weight.data.numpy().copy()
        predict_loss_data = predict_loss_train.data.numpy().copy()

    print('TrainActiv:', TrainActiv.shape)
    print('Weight_out:', Weight_out.shape)

    for Ai in range(Weight_out.shape[0]):
        ActivWeight = np.multiply(TrainActiv, Weight_out[Ai,:])   #todo need to find another method zhongang

        CorrActivWeight = np.corrcoef(np.transpose(ActivWeight))
        MeanActivWeight = np.mean(ActivWeight, axis=0)

        #print(('Weight_out: ', type(Weight_out), Weight_out.shape, Weight_out))
        #print(('CorrActivWeight: ', type(CorrActivWeight), CorrActivWeight.shape, CorrActivWeight))
        #print(('MeanActivWeight: ', type(MeanActivWeight), MeanActivWeight.shape, MeanActivWeight))

        SortActivWeight = -np.sort(-MeanActivWeight)
        #print(('SortActiveWeight: ', SortActivWeight))

        MeanActivWeight = np.expand_dims(MeanActivWeight, axis=0)
        CorrActivWeight = np.expand_dims(CorrActivWeight, axis=0)

        if Ai == 0:
            MeanActivWeight_All = MeanActivWeight.copy()
            CorrActivWeight_All = CorrActivWeight.copy()
            #print(MeanActivWeight_All.shape, CorrActivWeight_All.shape)
        else:
            # print(imagepool5, imagefc6, imagefc8)
            MeanActivWeight_All = np.concatenate((MeanActivWeight_All, MeanActivWeight), axis=0)
            CorrActivWeight_All = np.concatenate((CorrActivWeight_All, CorrActivWeight), axis=0)
            #print(MeanActivWeight_All.shape, CorrActivWeight_All.shape)
    ##############################################################################################################Adjust end

    return (SRAE_model, MeanActivWeight_All, CorrActivWeight_All, predict_loss_data)



def SRAE_Optim_retrained(conceptLoss_1allData_0nonMissingData, TrainX, TrainY, TestX, TestY, Yconcept_train, Yconcept_test, MissingLabel_Train, MissingLabel_Test,SRAE_model, parameters, BatchSize=50, Lr=0.001, EPOCH=1, useGPU=1, PT_inverse=1):
    ########################
    # Optimization for Sparse reconstruction autoencoder (SRAE)

    # Parameters:
    # -------------
    # TrainX: training data_X (8039 x 4096)
    # TrainY: training data_Y (8039 x 1 x 1)
    # TestX: testing data_X
    # TestY: testing data_Y
    # SRAE_model: the sparse reconstruction autoencoder model
    # parameters: the parameters for different loss
    # BatchSize: batch size
    # Lr: learning rate
    # EPOCH: epoch
    # useGPU: use GPU or not
    # PT_inverse: 1 means pull-away term along the images; 0 means pull-away term along the features
    ####################################################

    # sparseR = 0, using traditional reconstruction loss; sparseR = 1, using sparse reconstruction loss:
    sparseR = parameters[0]
    # Qq is the parameter q for the log penalty:
    Qq = parameters[1]
    # alpha_pred is the weight for prediction loss:
    alpha_pred = parameters[2]
    # alpha_recons is the weight for reconstruction loss:
    alpha_recons = parameters[3]
    # alpha_pull_away is the weight for pull-away term:
    alpha_pt_loss = parameters[4]

    alpha_concepts_loss = parameters[5]

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("PT_inverse: {}".format(PT_inverse))
    print("sparseR: {}".format(sparseR))
    print("Qq: {}".format(Qq))
    print("alpha_pred: {}".format(alpha_pred))
    print("alpha_recons: {}".format(alpha_recons))
    print("alpha_pt_loss: {}".format(alpha_pt_loss))
    print("alpha_concepts_loss: {}".format(alpha_concepts_loss))

    numberOf_NonMissingData =5 # this is used <in some way> for batch sizes

    number_non_multi = 0 #counts number of examples with zero concept vector (negative example)
    number_multi = 0 #counts number of examples with non-zero concept vector
    num_classes = torch.zeros( Yconcept_train.size(1) ) # = 3 (this is number of concepts)
    for row in range (Yconcept_train.size(0)):
        if ( (MissingLabel_Train[row] == 1)) : #no concept label for this example => negative example
            number_non_multi = number_non_multi + 1
        else:
            number_multi = number_multi + 1 #atleat one concept label for this example
        for class_Index in range (Yconcept_train.size(1)):
            if(Yconcept_train[row, class_Index] == 1):
                num_classes[class_Index] = num_classes[class_Index]+1 # num_classes is equivalent to sum(Yconcept_train)


            # new Train for mulit-labled Data
    TrainX_multi = torch.zeros(number_multi, TrainX.size(1)) # (number_multi x 4096) #ITS NOT MULTI LABELLED - JUST LABELLED
    TrainX_non_multi = torch.zeros(number_non_multi, TrainX.size(1)) # (number_non_multi x 4096) #ITS NO LABELS (negative examples)

    TrainY_multi = torch.zeros(number_multi,TrainY.size(1)+ Yconcept_train.size(1)) # (number_multi x (1+3)) = (39 x 4)
    TrainY_non_multi = torch.zeros(number_non_multi,TrainY.size(1)+ Yconcept_train.size(1)) # (number_non_multi x (1+3)) = (8000 x 4)

    index_multi = 0
    index_non_multi = 0
    # new Train for not labled Data
    for row in range (Yconcept_train.size(0)):
        allLables = torch.zeros((1,Yconcept_train.size(1)+TrainY.size(1))) # (1 x 4) - new variable for each row
        allLables[0,0] = TrainY[row,0][0] # get 1st label - the y logit
        for column in range (Yconcept_train.size(1)):
            allLables[0, column + 1] = Yconcept_train[row, column] # get the remaining 3 labels - concept labels

        # if ( (Yconcept_train[row,:]==0).all()) :
        if (MissingLabel_Train[row] == 1): # negative example, no concept label
            TrainX_non_multi[index_non_multi] = TrainX[row]
            TrainY_non_multi[index_non_multi] = allLables[0] # y logit and the (3x1) concept vector
            index_non_multi= index_non_multi+1

        else: #one or more concept labels
            TrainX_multi[index_multi] = TrainX[row]
            TrainY_multi[index_multi] = allLables[0] # y logit and the (3x1) concept vector
            index_multi= index_multi+1


    batch_size = ( round((TrainX.size(0)-number_multi)/(number_multi *numberOf_NonMissingData) )  +numberOf_NonMissingData) # 41 + 5
    trainData = Data.TensorDataset(TrainX_non_multi, TrainY_non_multi) # all negative exmples - why?
    train_loader = Data.DataLoader(trainData,batch_size=batch_size , shuffle=True)




    if useGPU == 1:
        SRAE_model = SRAE_model.cuda()
        test_x = Variable(TestX).cuda()
        test_y = Variable(TestY).cuda()
        train_x = Variable(TrainX).cuda()
        train_y = Variable(TrainY).cuda()
        y_concept_test = Variable(Yconcept_test).cuda()

    else:
        test_x = Variable(TestX) # (2010 x 4096)
        test_y = Variable(TestY) # (2010 x 1 x 1)
        train_x = Variable(TrainX) # (8039 x 4096)
        train_y = Variable(TrainY) # (8039 x 1 x 1)
        y_concept_test = Variable(Yconcept_test) # (2010 x 3) # WHY NO y_concept_train ?

    optimizer = torch.optim.Adam(SRAE_model.parameters(), lr=Lr, weight_decay=0.0001)      #weight_decay is L2 norm
    loss_function = nn.MSELoss()


    class_Weights = torch.zeros(  Yconcept_train.size(1) )
    for class_index in range(num_classes.size(0)):
        class_Weights[class_index] = max(num_classes)/num_classes[class_index]

    # num_classes = [26, 12, 5]
    # class_Weights = [1.0, 2.1667, 5.20]


    for epoch in range(EPOCH):

        adjust_lr(optimizer, epoch, Lr)

        for step, (x, newy) in enumerate(train_loader): # step ranges from 0 to 173 (8000/46)
            newyTemp = torch.zeros(newy.size(0)+ numberOf_NonMissingData, newy.size(1)) # (46+5) x 4
            newyTemp[0:newy.size(0), :] = newy
            #print("+++++++++++ TrainY_multi.size(0): {}".format(TrainY_multi.size(0))) # 39
            if (step < TrainY_multi.size(0) ): #todo list mandana: code this section for , I must write a loop for this section
               newyTemp[newy.size(0), :] = TrainY_multi[step] # one positive example added in each of the first 39 batches - this way all positive training examples are also used up for training
            newy= newyTemp # newy with 5 extra rows of zeros: 1st of which might have a positive example

            x_allDataInBatch = torch.zeros(x.size(0)+ numberOf_NonMissingData, x.size(1)) # (46+5) x 4096
            x_allDataInBatch[0:x.size(0), :] = x # x with 5 extra rows of zeros

            if (step < TrainX_multi.size(0) ):#todo list mandana: code this section for , I must write a loop for this section
                x_allDataInBatch[x.size(0), :] = TrainX_multi[step] # one positive example added in each of the first 39 batches - this way all positive training examples are also used up for training

            y = newy[:,0] # just taking the y logit  (51 x 1)
            multi_label = newy[:,1:Yconcept_train.size(1)+1] # just taking the concept labels (51 x 3)

            if useGPU == 1:
                b_x = Variable(x_allDataInBatch).cuda()
                b_y = Variable(y).cuda()
                b_multi_label = Variable(multi_label).cuda()
            else:
                b_x = Variable(x_allDataInBatch)
                b_y = Variable(y)
                b_multi_label = Variable(multi_label)

            #forward
            (b_x_encoder, b_x_decoder, b_y_pred) = SRAE_model(b_x)
            predict_loss = loss_function(b_y_pred, b_y) # note that prediction loss is MSE loss between logits and not loss between category labels # DOES UNSQUEEZE CAUSE A PROBLEM HERE ?

            if sparseR == 1:
                recons_loss = util.logpenaltyFC(b_x_decoder, b_x, Qq) # sparse reconstruction loss
            else:
                recons_loss = loss_function(b_x_decoder, b_x) # traditional reconstruction loss - just MSE

            FWeight = SRAE_model.classifier.weight # weights of linear classifier (1 x NN_middle3)
            pt_loss = 0
            for Fi in range(FWeight.shape[0]): # no looping - this runs only once since FWeight.shape[0] = 1
                # b_x_encoder (51 x 3)
                # FWeight[Fi,:] (1 x 3)
                # b_x_FW (51 x 3)
                b_x_FW = torch.mul(b_x_encoder, FWeight[Fi,:]) # each row of b_x_encoder (1x3) multiplied by FWeight (1x3) elementwise - the paper does not specify multiplying weights of linear layer in the pull away term ?!
                b_x_FW = torch.add(b_x_FW, 1e-8) # to avoid nan
                if PT_inverse == 1:
                    pt_loss_i = util.PT_loss(b_x_FW, useGPU) # pull-away term along the images
                elif PT_inverse == 0:
                    pt_loss_i = util.PT_loss_true(b_x_FW, useGPU) # pull-away term along the features

                pt_loss = pt_loss + pt_loss_i

            concept_loss = 0
            element_weights = torch.zeros(  b_x_encoder.size(0) )
            # element_weights = torch.FloatTensor([1]* (batch_size- numberOf_NonMissingData) + [(  round((batch_size- numberOf_NonMissingData) /numberOf_NonMissingData)  )]*numberOf_NonMissingData).view(-1, 1)
            element_weights = torch.FloatTensor([1]* (x.size(0) ) + [(  round((x.size(0)) /numberOf_NonMissingData)  )]*numberOf_NonMissingData).view(-1, 1) # (51x1) vector of ones with last five elements as 9s
            element_weights = element_weights.repeat(1, num_classes.size(0)) # (51 x 3)

            #========================#========================#========================#========================#========================
            # another way to compute pos_classes
            pos_weigths_withElement = torch.zeros(  num_classes.size(0) )
            numberofNegative = torch.zeros(  num_classes.size(0) )
            numberofPositives = torch.zeros(  num_classes.size(0) )
            for class_index in range(num_classes.size(0)): # 3
                for element_index in range(y.size(0)): # 51
                    if (b_multi_label[element_index, class_index ] == 0): # note only one or zero positive example in the 51 examples
                        numberofNegative[class_index] = numberofNegative[class_index]+1*element_weights[element_index, class_index] # +1
                    else:
                        numberofPositives[class_index] = numberofPositives[class_index]+1*element_weights[element_index, class_index] # +9

            for class_index in range(num_classes.size(0)):
                if numberofPositives[class_index] > 0: # will be true for first 39 steps
                    pos_weigths_withElement[class_index] = numberofNegative[class_index]/numberofPositives[class_index] # (3x1) vector of zeros with one or two entry = 50/9
                else:
                    pos_weigths_withElement[class_index] = 1
            #========================#========================#========================#========================#========================


            if useGPU == 1:
                b_element_weights = Variable(element_weights).cuda() # 51 x 3
                b_pos_weigths_withElement = Variable(pos_weigths_withElement).cuda() # 1 x 3
                b_class_Weights = Variable(class_Weights).cuda() # 1 x 3
            else:
                b_element_weights = Variable(element_weights)
                b_pos_weigths_withElement = Variable(pos_weigths_withElement)
                b_class_Weights = Variable(class_Weights)

            if (conceptLoss_1allData_0nonMissingData == 1): #this is where conceptLoss_1allData_0nonMissingData is used
                bce_criterion = nn.BCEWithLogitsLoss(weight = None, reduce = None)
                bce_criterion_class = nn.BCEWithLogitsLoss(weight = b_class_Weights, reduce = None)
                bce_criterion_elements = nn.BCEWithLogitsLoss(weight = b_element_weights, reduce = None)
                bce_criterion_pos_class = nn.BCEWithLogitsLoss( reduce = None, pos_weight= b_element_weights)
                bce_criterion_elements_pos_class = nn.BCEWithLogitsLoss( weight = b_element_weights,reduce = None, pos_weight= b_pos_weigths_withElement) # binary cross entropy loss after applying softmax to logits with some weighting scheme

                b_x_encoder_concept = b_x_encoder[:,0:b_multi_label.size(1)] # THIS IS WHERE DIMENSION MISMATCH (IF PRESENT) BETWEEN b_x_encoder AND b_multi_label IS HANDLED - BY IGNORING SOME NEURONS OF b_x_encoder
                bce_loss = bce_criterion(b_x_encoder_concept, b_multi_label)
                bce_loss_class = bce_criterion_class(b_x_encoder_concept, b_multi_label)
                bce_loss_elements = bce_criterion_elements(b_x_encoder_concept, b_multi_label)
                bce_loss_pos_class = bce_criterion_pos_class(b_x_encoder_concept, b_multi_label)
                bce_loss_elements_pos_class = bce_criterion_elements_pos_class(b_x_encoder_concept, b_multi_label) # only this one used ?

                mulit_label_losss = bce_loss_elements_pos_class#nn.BCEWithLogitsLoss(weight = class_Weights)
                concept_loss = mulit_label_losss

            else:
                for indexData in range(multi_label.size(0)):
                    if (indexData> x.size(0)):
                        mulit_label_losss = nn.BCEWithLogitsLoss()
                        concept_loss = concept_loss + mulit_label_losss( b_x_encoder[indexData][0:multi_label.size(1)], b_multi_label[indexData])


            loss = alpha_pred * predict_loss + alpha_recons * recons_loss + alpha_pt_loss * pt_loss +  alpha_concepts_loss * concept_loss

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1 == 0:
                SRAE_model.eval()
                (test_x_encoder,test_x_decoder,test_y_pred) = SRAE_model(test_x)

                #--------------------------------------------------------------

                number_non_multi_test = 0
                number_multi_test = 0
                for row in range (Yconcept_test.size(0)):
                    if ( (MissingLabel_Test[row]==1)) :
                        number_non_multi_test = number_non_multi_test + 1
                    else:
                        number_multi_test = number_multi_test + 1

                test_predict_loss = loss_function(test_y_pred, test_y)
                if sparseR==1:
                    test_recons_loss =util.logpenaltyFC(test_x_decoder, test_x, Qq)
                else:
                    test_recons_loss = loss_function(test_x_decoder, test_x)

                FWeight = SRAE_model.classifier.weight

                test_pt_loss = 0
                for Fi in range(FWeight.shape[0]):
                    test_FW = torch.mul(test_x_encoder, FWeight[Fi,:])
                    test_FW = torch.add(test_FW, 1e-8)
                    if PT_inverse== 1:
                        test_pt_loss_i = util.PT_loss(test_FW, useGPU)
                    elif PT_inverse==0:
                        test_pt_loss_i = util.PT_loss_true(test_FW, useGPU)

                    test_pt_loss = test_pt_loss+test_pt_loss_i


                print(('Epoch:', epoch, '|Step:', step,
                      '|train loss:%.4f' % loss.data, '|train pred loss:%.4f' % predict_loss.data,
                      '|train recons loss:%.4f' % recons_loss.data,
                      '|train PT loss:%.4f' % pt_loss.data,
                      '|train concept loss:%.4f' % concept_loss,
                      '|test pred loss:%.4f' % test_predict_loss.data, '|test recons loss:%.4f' % test_recons_loss.data,
                      '|test PT loss:%.4f' % test_pt_loss.data
                      # '|test concept loss:%.4f' % concept_loss_test.data
                       ))
                      # '|train recons loss:%.4f' % recons_loss.data[0],
                      # '|train PT loss:%.4f' % pt_loss.data[0],
                      # '|test pred loss:%.4f' % test_predict_loss.data[0], '|test recons loss:%.4f' % test_recons_loss.data[0],
                      # '|test PT loss:%.4f' % test_pt_loss.data[0]))

                SRAE_model.train()


    ###################################################################################Adjust
    # compute the mean value of activation * weight for x-features
    # compute the correlation coefficient of activation * weight for x-features
    (train_x_encoder, train_x_decoder, train_y_pred) = SRAE_model(train_x)

    predict_loss_train = loss_function(train_y_pred, train_y) # HERE train_y SHAPE HAS EXTRA DIMENSION (MIGHT BE A PROBLEM)

    if useGPU==1:
        TrainActiv = train_x_encoder.data.cpu().numpy().copy()
        Weight_out = SRAE_model.classifier.weight.data.cpu().numpy().copy()
        predict_loss_data = predict_loss_train.data.cpu().numpy().copy()

    else:
        TrainActiv = train_x_encoder.data.numpy().copy() # concept layer encoding
        Weight_out = SRAE_model.classifier.weight.data.numpy().copy() # predicted logit
        predict_loss_data = predict_loss_train.data.numpy().copy()

    print('~~~~~~~~~~~~~~ TrainActiv:', TrainActiv.shape) # 8039 x 3
    print('~~~~~~~~~~~~~~ Weight_out:', Weight_out.shape) # 1 x 3

    for Ai in range(Weight_out.shape[0]):
        ActivWeight = np.multiply(TrainActiv, Weight_out[Ai,:])   #todo need to find another method zhongang - this is like the embedding used for pull away term

        CorrActivWeight = np.corrcoef(np.transpose(ActivWeight))
        MeanActivWeight = np.mean(ActivWeight, axis=0)

        #print(('Weight_out: ', type(Weight_out), Weight_out.shape, Weight_out))
        #print(('CorrActivWeight: ', type(CorrActivWeight), CorrActivWeight.shape, CorrActivWeight))
        #print(('MeanActivWeight: ', type(MeanActivWeight), MeanActivWeight.shape, MeanActivWeight))

        SortActivWeight = -np.sort(-MeanActivWeight)
        #print(('SortActiveWeight: ', SortActivWeight))

        MeanActivWeight = np.expand_dims(MeanActivWeight, axis=0)
        CorrActivWeight = np.expand_dims(CorrActivWeight, axis=0)

        if Ai == 0:
            MeanActivWeight_All = MeanActivWeight.copy()
            CorrActivWeight_All = CorrActivWeight.copy()
            #print(MeanActivWeight_All.shape, CorrActivWeight_All.shape)
        else:
            # print(imagepool5, imagefc6, imagefc8)
            MeanActivWeight_All = np.concatenate((MeanActivWeight_All, MeanActivWeight), axis=0)
            CorrActivWeight_All = np.concatenate((CorrActivWeight_All, CorrActivWeight), axis=0)
            #print(MeanActivWeight_All.shape, CorrActivWeight_All.shape)
    ##############################################################################################################Adjust end

    return (SRAE_model, MeanActivWeight_All, CorrActivWeight_All, predict_loss_data)

def SRAE_Optim_retrained_NOweightsloss(conceptLoss_1allData_0nonMissingData, TrainX, TrainY, TestX, TestY, Yconcept_train, Yconcept_test, MissingLabel_Train, MissingLabel_Test,SRAE_model, parameters, BatchSize=50, Lr=0.001, EPOCH=1, useGPU=1, PT_inverse=1):
    ########################
    # Optimization for Sparse reconstruction autoencoder (SRAE)

    # Parameters:
    # -------------
    # TrainX: training data_X
    # TrainY: training data_Y
    # TestX: testing data_X
    # TestY: testing data_Y
    # SRAE_model: the sparse reconstruction autoencoder model
    # parameters: the parameters for different loss
    # BatchSize: batch size
    # Lr: learning rate
    # EPOCH: epoch
    # useGPU: use GPU or not
    # PT_inverse: 1 means pull-away term along the images; 0 means pull-away term along the features
    ####################################################

    # sparseR = 0, using traditional reconstruction loss; sparseR = 1, using sparse reconstruction loss:
    sparseR = parameters[0]
    # Qq is the parameter q for the log penalty:
    Qq = parameters[1]
    # alpha_pred is the weight for prediction loss:
    alpha_pred = parameters[2]
    # alpha_recons is the weight for reconstruction loss:
    alpha_recons = parameters[3]
    # alpha_pull_away is the weight for pull-away term:
    alpha_pt_loss = parameters[4]

    alpha_concepts_loss = parameters[5]

    number_non_multi = 0
    number_multi = 0
    num_classes = torch.zeros( Yconcept_train.size(1) )
    for row in range (Yconcept_train.size(0)):
        if ( (MissingLabel_Train[row] == 1)) :
            number_non_multi = number_non_multi + 1
        else:
            number_multi = number_multi + 1
        for class_Index in range (Yconcept_train.size(1)):
            if(Yconcept_train[row, class_Index] == 1):
                num_classes[class_Index] = num_classes[class_Index]+1


            # new Train for mulit-labled Data
    TrainX_multi = torch.zeros(number_multi, TrainX.size(1))#torch.tensor([])
    TrainX_non_multi = torch.zeros(number_non_multi, TrainX.size(1))#torch.tensor([])

    TrainY_multi = torch.zeros(number_multi,TrainY.size(1)+ Yconcept_train.size(1))#torch.tensor([])
    TrainY_non_multi = torch.zeros(number_non_multi,TrainY.size(1)+ Yconcept_train.size(1))#torch.tensor([])

    index_multi = 0
    index_non_multi = 0
    # new Train for not labled Data
    for row in range (Yconcept_train.size(0)):
        allLables = torch.zeros((1,Yconcept_train.size(1)+TrainY.size(1)))
        allLables[0,0] = TrainY[row,0][0]
        for column in range (Yconcept_train.size(1)):
            allLables[0, column + 1] = Yconcept_train[row, column]

        # if ( (Yconcept_train[row,:]==0).all()) :
        if (MissingLabel_Train[row] == 1):
            TrainX_non_multi[index_non_multi] = TrainX[row]
            TrainY_non_multi[index_non_multi] = allLables[0]
            index_non_multi= index_non_multi+1

        else:
            TrainX_multi[index_multi] = TrainX[row]
            TrainY_multi[index_multi] = allLables[0]
            index_multi= index_multi+1

    trainData = Data.TensorDataset(TrainX_non_multi, TrainY_non_multi)
    train_loader = Data.DataLoader(trainData, batch_size = ( round(TrainX.size(0)/number_multi )), shuffle=True)

    if useGPU == 1:
        SRAE_model = SRAE_model.cuda()
        test_x = Variable(TestX).cuda()
        test_y = Variable(TestY).cuda()
        train_x = Variable(TrainX).cuda()
        train_y = Variable(TrainY).cuda()
        y_concept_test = Variable(Yconcept_test).cuda()

    else:
        test_x = Variable(TestX)
        test_y = Variable(TestY)
        train_x = Variable(TrainX)
        train_y = Variable(TrainY)
        y_concept_test = Variable(Yconcept_test)

    optimizer = torch.optim.Adam(SRAE_model.parameters(), lr=Lr, weight_decay=0.0001)      #weight_decay is L2 norm
    loss_function = nn.MSELoss()

    class_Weights = torch.zeros(  Yconcept_train.size(1) )
    for class_index in range(num_classes.size(0)):
        class_Weights[class_index] = max(num_classes)/num_classes[class_index]


    numberOf_NonMissingData =1
    for epoch in range(EPOCH):

        adjust_lr(optimizer, epoch, Lr)

        for step, (x, newy) in enumerate(train_loader):
            newyTemp = torch.zeros(newy.size(0)+ numberOf_NonMissingData, newy.size(1))
            newyTemp[0:newy.size(0), :] = newy
            if (step < TrainY_multi.size(0) ): #todo list mandana: code this section for , I must write a loop for this section
                newyTemp[newy.size(0), :] = TrainY_multi[step]
            newy= newyTemp

            x_allDataInBatch = torch.zeros(x.size(0)+ numberOf_NonMissingData, x.size(1))
            x_allDataInBatch[0:x.size(0), :] = x

            if (step < TrainX_multi.size(0) ):#todo list mandana: code this section for , I must write a loop for this section
                x_allDataInBatch[x.size(0), :] = TrainX_multi[step]

            y = newy[:,0]
            multi_label = newy[:,1:Yconcept_train.size(1)+1]

            if useGPU == 1:
                b_x = Variable(x_allDataInBatch).cuda()
                b_y = Variable(y).cuda()
                b_multi_label = Variable(multi_label).cuda()
            else:
                b_x = Variable(x_allDataInBatch)
                b_y = Variable(y)
                b_multi_label = Variable(multi_label)

            #forward
            (b_x_encoder, b_x_decoder, b_y_pred) = SRAE_model(b_x)

            predict_loss = loss_function(b_y_pred, b_y)

            if sparseR == 1:
                recons_loss = util.logpenaltyFC(b_x_decoder, b_x, Qq) # sparse reconstruction loss
            else:
                recons_loss = loss_function(b_x_decoder, b_x) # traditional reconstruction loss

            FWeight = SRAE_model.classifier.weight

            pt_loss = 0
            for Fi in range(FWeight.shape[0]):
                b_x_FW = torch.mul(b_x_encoder, FWeight[Fi,:])
                b_x_FW = torch.add(b_x_FW, 1e-8) # to avoid nan
                if PT_inverse == 1:
                    pt_loss_i = util.PT_loss(b_x_FW, useGPU) # pull-away term along the images
                elif PT_inverse == 0:
                    pt_loss_i = util.PT_loss_true(b_x_FW, useGPU) # pull-away term along the features

                pt_loss = pt_loss + pt_loss_i

            concept_loss = 0
            # concept_loss1 = 0
            if (conceptLoss_1allData_0nonMissingData == 1):
                mulit_label_losss = nn.BCEWithLogitsLoss(weight = class_Weights , reduce = None)
                concept_loss = mulit_label_losss(b_x_encoder, multi_label)
                # concept_loss1 = mulit_label_losss1(b_x_encoder, multi_label)

            else:
                for indexData in range(multi_label.size(0)):
                    if (indexData> x.size(0)):
                        concept_loss = concept_loss + mulit_label_losss( b_x_encoder[indexData][0:multi_label.size(1)], b_multi_label[indexData])

            loss = alpha_pred * predict_loss + alpha_recons * recons_loss + alpha_pt_loss * pt_loss +  alpha_concepts_loss * concept_loss

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1 == 0:
                SRAE_model.eval()
                (test_x_encoder,test_x_decoder,test_y_pred) = SRAE_model(test_x)

                #--------------------------------------------------------------

                number_non_multi_test = 0
                number_multi_test = 0
                for row in range (Yconcept_test.size(0)):
                    if ( (MissingLabel_Test[row]==1)) :
                        number_non_multi_test = number_non_multi_test + 1
                    else:
                        number_multi_test = number_multi_test + 1

                # new Train for mulit-labled Data
                TestX_multi = torch.zeros(number_multi, test_x.size(1))#torch.tensor([])
                TestX_non_multi = torch.zeros(number_non_multi, test_x.size(1))#torch.tensor([])

                TestY_multi = torch.zeros(number_multi , TestY.size(1) + Yconcept_test.size(1))#torch.tensor([])
                TestY_non_multi = torch.zeros(number_non_multi , test_y.size(1) + Yconcept_test.size(1))#torch.tensor([])


                # # new Train for not labled Data
                # index_multi_test = 0
                # index_non_multi_test = 0
                # for row in range (Yconcept_test.size(0)):
                #
                #     allLables = torch.zeros((1,Yconcept_test.size(1)+test_y.size(1)))
                #     allLables[0,0] = test_y[row,0][0]
                #     for column in range (Yconcept_test.size(1)):
                #         allLables[0, column+1] = Yconcept_test[row, column]
                #
                #     if ( (Yconcept_train[row,:]==0).all()) :
                #         TestX_non_multi[index_non_multi_test] = test_x[row]
                #         TestY_non_multi[index_non_multi_test] = allLables[0]
                #         index_non_multi_test = index_non_multi_test + 1
                #
                #     else:
                #         TestX_multi[index_multi_test] = test_x[row]
                #         TestY_multi[index_multi_test] = allLables[0]
                #         index_multi_test = index_multi_test + 1
                #---------------------------------------------
                test_predict_loss = loss_function(test_y_pred, test_y)
                if sparseR==1:
                    test_recons_loss =util.logpenaltyFC(test_x_decoder, test_x, Qq)
                else:
                    test_recons_loss = loss_function(test_x_decoder, test_x)

                FWeight = SRAE_model.classifier.weight

                test_pt_loss = 0
                for Fi in range(FWeight.shape[0]):
                    test_FW = torch.mul(test_x_encoder, FWeight[Fi,:])
                    test_FW = torch.add(test_FW, 1e-8)
                    if PT_inverse== 1:
                        test_pt_loss_i = util.PT_loss(test_FW, useGPU)
                    elif PT_inverse==0:
                        test_pt_loss_i = util.PT_loss_true(test_FW, useGPU)

                    test_pt_loss = test_pt_loss+test_pt_loss_i

                concept_loss_test = 0
                for indexData in range(y_concept_test.size(0)):
                    if ( (y_concept_test[indexData,:]==0).all()) :
                        a=1# print("All zero")
                    else:#
                        concept_loss_test = concept_loss_test +  mulit_label_losss(test_x_encoder[indexData][0:y_concept_test.size(1)], y_concept_test[indexData])

                print(('Epoch:', epoch, '|Step:', step,
                       '|train loss:%.4f' % loss.data, '|train pred loss:%.4f' % predict_loss.data,
                       '|train recons loss:%.4f' % recons_loss.data,
                       '|train PT loss:%.4f' % pt_loss.data,
                       '|train concept loss:%.4f' % concept_loss,
                       # '|train concept loss1:%.4f' % concept_loss1,
                       '|test pred loss:%.4f' % test_predict_loss.data, '|test recons loss:%.4f' % test_recons_loss.data,
                       '|test PT loss:%.4f' % test_pt_loss.data,
                       '|test concept loss:%.4f' % concept_loss_test.data
                       ))
                # '|train loss:%.4f' % loss.data[0], '|train pred loss:%.4f' % predict_loss.data[0],
                # '|train recons loss:%.4f' % recons_loss.data[0],
                # '|train PT loss:%.4f' % pt_loss.data[0],
                # '|test pred loss:%.4f' % test_predict_loss.data[0], '|test recons loss:%.4f' % test_recons_loss.data[0],
                # '|test PT loss:%.4f' % test_pt_loss.data[0]))

                SRAE_model.train()


    ###################################################################################Adjust
    # compute the mean value of activation * weight for x-features
    # compute the correlation coefficient of activation * weight for x-features
    (train_x_encoder, train_x_decoder, train_y_pred) = SRAE_model(train_x)

    predict_loss_train = loss_function(train_y_pred, train_y)

    if useGPU==1:
        TrainActiv = train_x_encoder.data.cpu().numpy().copy()
        Weight_out = SRAE_model.classifier.weight.data.cpu().numpy().copy()
        predict_loss_data = predict_loss_train.data.cpu().numpy().copy()

    else:
        TrainActiv = train_x_encoder.data.numpy().copy()
        Weight_out = SRAE_model.classifier.weight.data.numpy().copy()
        predict_loss_data = predict_loss_train.data.numpy().copy()

    print('TrainActiv:', TrainActiv.shape)
    print('Weight_out:', Weight_out.shape)

    for Ai in range(Weight_out.shape[0]):
        ActivWeight = np.multiply(TrainActiv, Weight_out[Ai,:])   #todo need to find another method zhongang

        CorrActivWeight = np.corrcoef(np.transpose(ActivWeight))
        MeanActivWeight = np.mean(ActivWeight, axis=0)

        #print(('Weight_out: ', type(Weight_out), Weight_out.shape, Weight_out))
        #print(('CorrActivWeight: ', type(CorrActivWeight), CorrActivWeight.shape, CorrActivWeight))
        #print(('MeanActivWeight: ', type(MeanActivWeight), MeanActivWeight.shape, MeanActivWeight))

        SortActivWeight = -np.sort(-MeanActivWeight)
        #print(('SortActiveWeight: ', SortActivWeight))

        MeanActivWeight = np.expand_dims(MeanActivWeight, axis=0)
        CorrActivWeight = np.expand_dims(CorrActivWeight, axis=0)

        if Ai == 0:
            MeanActivWeight_All = MeanActivWeight.copy()
            CorrActivWeight_All = CorrActivWeight.copy()
            #print(MeanActivWeight_All.shape, CorrActivWeight_All.shape)
        else:
            # print(imagepool5, imagefc6, imagefc8)
            MeanActivWeight_All = np.concatenate((MeanActivWeight_All, MeanActivWeight), axis=0)
            CorrActivWeight_All = np.concatenate((CorrActivWeight_All, CorrActivWeight), axis=0)
            #print(MeanActivWeight_All.shape, CorrActivWeight_All.shape)
    ##############################################################################################################Adjust end

    return (SRAE_model, MeanActivWeight_All, CorrActivWeight_All, predict_loss_data)

def SRAE_Optim(TrainX, TrainY, TestX, TestY, SRAE_model, parameters, BatchSize=50, Lr=0.001, EPOCH=1, useGPU=1, PT_inverse=1):
    ########################
    # Optimization for Sparse reconstruction autoencoder (SRAE)

    # Parameters:
    # -------------
    # TrainX: training data_X
    # TrainY: training data_Y
    # TestX: testing data_X
    # TestY: testing data_Y
    # SRAE_model: the sparse reconstruction autoencoder model
    # parameters: the parameters for different loss
    # BatchSize: batch size
    # Lr: learning rate
    # EPOCH: epoch
    # useGPU: use GPU or not
    # PT_inverse: 1 means pull-away term along the images; 0 means pull-away term along the features
    ####################################################

    # sparseR = 0, using traditional reconstruction loss; sparseR = 1, using sparse reconstruction loss:
    sparseR = parameters[0]
    # Qq is the parameter q for the log penalty:
    Qq = parameters[1]
    # alpha_pred is the weight for prediction loss:
    alpha_pred = parameters[2]
    # alpha_recons is the weight for reconstruction loss:
    alpha_recons = parameters[3]
    # alpha_pull_away is the weight for pull-away term:
    alpha_pt_loss = parameters[4]

    trainData = Data.TensorDataset(TrainX, TrainY)
    train_loader = Data.DataLoader(trainData, batch_size=BatchSize, shuffle=True)

    if useGPU==1:
        SRAE_model = SRAE_model.cuda()
        test_x = Variable(TestX).cuda()
        test_y = Variable(TestY).cuda()
        train_x = Variable(TrainX).cuda()
        train_y = Variable(TrainY).cuda()
    else:
        test_x = Variable(TestX)
        test_y = Variable(TestY)
        train_x = Variable(TrainX)
        train_y = Variable(TrainY)


    optimizer = torch.optim.Adam(SRAE_model.parameters(), lr=Lr, weight_decay=0.0001)      #weight_decay is L2 norm
    loss_function = nn.MSELoss()

    for epoch in range(EPOCH):

        adjust_lr(optimizer, epoch, Lr)

        for step, (x, y) in enumerate(train_loader):
            if useGPU==1:
                b_x = Variable(x).cuda()
                b_y = Variable(y).cuda()
            else:
                b_x = Variable(x)
                b_y = Variable(y)

            #forward
            (b_x_encoder, b_x_decoder, b_y_pred) = SRAE_model(b_x)

            predict_loss = loss_function(b_y_pred, b_y)
            if sparseR==1:
                recons_loss = util.logpenaltyFC(b_x_decoder, b_x, Qq) # sparse reconstruction loss
            else:
                recons_loss = loss_function(b_x_decoder, b_x) # traditional reconstruction loss

            FWeight = SRAE_model.classifier.weight
            #print(FWeight.shape)
            #print(b_x_encoder.shape)

            pt_loss = 0
            for Fi in range(FWeight.shape[0]):
                b_x_FW = torch.mul(b_x_encoder, FWeight[Fi,:])
                b_x_FW = torch.add(b_x_FW, 1e-8) # to avoid nan
                if PT_inverse==1:
                    pt_loss_i = util.PT_loss(b_x_FW, useGPU) # pull-away term along the images
                elif PT_inverse==0:
                    pt_loss_i = util.PT_loss_true(b_x_FW, useGPU) # pull-away term along the features

                pt_loss = pt_loss + pt_loss_i

            loss = alpha_pred * predict_loss + alpha_recons * recons_loss + alpha_pt_loss * pt_loss


            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                SRAE_model.eval()
                (test_x_encoder,test_x_decoder,test_y_pred) = SRAE_model(test_x)
                test_predict_loss = loss_function(test_y_pred, test_y)
                if sparseR==1:
                    test_recons_loss =util.logpenaltyFC(test_x_decoder, test_x, Qq)
                else:
                    test_recons_loss = loss_function(test_x_decoder, test_x)


                FWeight = SRAE_model.classifier.weight

                test_pt_loss = 0
                for Fi in range(FWeight.shape[0]):
                    test_FW = torch.mul(test_x_encoder, FWeight[Fi,:])
                    test_FW = torch.add(test_FW, 1e-8)
                    if PT_inverse==1:
                        test_pt_loss_i = util.PT_loss(test_FW, useGPU)
                    elif PT_inverse==0:
                        test_pt_loss_i = util.PT_loss_true(test_FW, useGPU)

                    test_pt_loss = test_pt_loss+test_pt_loss_i

                print(('Epoch:', epoch, '|Step:', step,
                       '|train loss:%.4f' % loss.data, '|train pred loss:%.4f' % predict_loss.data,
                       '|train recons loss:%.4f' % recons_loss.data,
                       '|train PT loss:%.4f' % pt_loss.data,
                       '|test pred loss:%.4f' % test_predict_loss.data, '|test recons loss:%.4f' % test_recons_loss.data,
                       '|test PT loss:%.4f' % test_pt_loss.data))

                SRAE_model.train()


    ###################################################################################Adjust
    # compute the mean value of activation * weight for x-features
    # compute the correlation coefficient of activation * weight for x-features
    (train_x_encoder, train_x_decoder, train_y_pred) = SRAE_model(train_x)

    predict_loss_train = loss_function(train_y_pred, train_y)

    if useGPU==1:
        TrainActiv = train_x_encoder.data.cpu().numpy().copy()
        Weight_out = SRAE_model.classifier.weight.data.cpu().numpy().copy()
        predict_loss_data = predict_loss_train.data.cpu().numpy().copy()

    else:
        TrainActiv = train_x_encoder.data.numpy().copy()
        Weight_out = SRAE_model.classifier.weight.data.numpy().copy()
        predict_loss_data = predict_loss_train.data.numpy().copy()

    print('TrainActiv:', TrainActiv.shape)
    print('Weight_out:', Weight_out.shape)

    for Ai in range(Weight_out.shape[0]):
        ActivWeight = np.multiply(TrainActiv, Weight_out[Ai,:])   #todo need to find another method zhongang

        CorrActivWeight = np.corrcoef(np.transpose(ActivWeight))
        MeanActivWeight = np.mean(ActivWeight, axis=0)

        #print(('Weight_out: ', type(Weight_out), Weight_out.shape, Weight_out))
        #print(('CorrActivWeight: ', type(CorrActivWeight), CorrActivWeight.shape, CorrActivWeight))
        #print(('MeanActivWeight: ', type(MeanActivWeight), MeanActivWeight.shape, MeanActivWeight))

        SortActivWeight = -np.sort(-MeanActivWeight)
        #print(('SortActiveWeight: ', SortActivWeight))

        MeanActivWeight = np.expand_dims(MeanActivWeight, axis=0)
        CorrActivWeight = np.expand_dims(CorrActivWeight, axis=0)

        if Ai == 0:
            MeanActivWeight_All = MeanActivWeight.copy()
            CorrActivWeight_All = CorrActivWeight.copy()
            #print(MeanActivWeight_All.shape, CorrActivWeight_All.shape)
        else:
            # print(imagepool5, imagefc6, imagefc8)
            MeanActivWeight_All = np.concatenate((MeanActivWeight_All, MeanActivWeight), axis=0)
            CorrActivWeight_All = np.concatenate((CorrActivWeight_All, CorrActivWeight), axis=0)
            #print(MeanActivWeight_All.shape, CorrActivWeight_All.shape)
    ##############################################################################################################Adjust end

    return (SRAE_model, MeanActivWeight_All, CorrActivWeight_All, predict_loss_data)
