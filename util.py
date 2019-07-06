#coding=utf-8
# util
# python Version: python3.6
# by Zhongang Qi (qiz@oregonstate.edu)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pylab
import os
import csv
from skimage import transform, filters
from textwrap import wrap
import scipy.io as scio
#from compiler.ast import flatten

def Generate_onevsall_ExamplesForSemanticConcepts(Cname, label_ind):

    all_pos_index = []
    all_label_ind = []
    pos_len=0
    totalConcepts =[] #set of all concepts for label_ind class
    groundTruthListFile = '../../../dataExperts/Expert1/featureSelection=MSF_1/IFeature_imgfc6_threshold=0.3/groundTruthVisualWords_cat'+ str(label_ind)+'.csv'
    with open(groundTruthListFile, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader]

        for row in rows:
            if not int(row[' visual Word Inexd']) in totalConcepts:
                print(row[' visual Word Inexd'])
                totalConcepts.append(int(row[' visual Word Inexd']))

    # check how many concepts do we have here
    totalConcepts = sorted(totalConcepts, key=int)
    mulit_label_concepts=[]
    with open('../../../../DataSetforXAI/Results_newEBP/Model/thefilename.txt','r') as f1: #file with all image paths
        linenum = 0
        pos_index = [] #stores index (serial number) of all image file in dataset classified as positive for label_ind class by NN
        neg_index = []

        pos_fileName = [] #stores name of all image files classified as positive for class label_ind by NN
        neg_fileName = []

        pos_index_category = [] #stores ground truth category_index for all positively classified images
        neg_index_category = []

        pos_index_xnn = [] #stores xnn active for the concept
        neg_index_xnn = []



        pos_index_concept_labels = [] #stores one-hot concept vector for each positive classified image
        neg_index_concept_labels = []

        #neg_index = []
        for line in f1.readlines():
            line = line.strip()

            # find the example in the ground-truth file
            mulit_label_concepts = np.zeros(len(totalConcepts)) #a concept vector (3x1)
            positveClassifiedImage=0
            xnn=-1
            for row in rows:
                if row[' FileName'][3:-4] in line:# find the file - note: happens only for one entry
                    mulit_label_concepts[  totalConcepts.index(int(row[' visual Word Inexd']))] =  1
                    positveClassifiedImage=1
                    xnn= row[' xnnIndex']
                    # print(row[' visual Word Inexd'])

            # find the category of the file
            subStrs= line.split('/')
            categoryName= subStrs[7]
            categoryIndex= int(categoryName[0:3]) #same as label_ind unless false positive
            fileName= subStrs[8] #image file name

            if positveClassifiedImage==1 : #file exists in the visual concept labels file for class label_ind
            # if Cname in line:
                pos_index.append(linenum)
                pos_index_concept_labels.append(mulit_label_concepts) # note that pos_index_concept_labels is not one hot
                pos_index_category.append(categoryIndex)
                pos_index_xnn.append(xnn)
                pos_fileName.append(fileName)


            else: # any image that is not positively classified is negative
                neg_index.append(linenum)
                neg_index_concept_labels.append(mulit_label_concepts) # all entries are zero anyway
                neg_index_category.append(categoryIndex)
                neg_index_xnn.append(xnn)
                neg_fileName.append(fileName)

            linenum = linenum + 1

            # find the category of the file

    pos_len += len(pos_index) # number of positive examples
    #neg_len +=len(neg_index)

    all_pos_index.append(pos_index) # wouldnt this just have one element = pos_index ?
    #all_neg_index.append(neg_index)
    all_label_ind.append(label_ind) # this will just be label_ind


    print('sum of len of pos:', pos_len)
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]

    all_pos_index = flatten(all_pos_index)
    print('all_pos_index len:', len(all_pos_index)) # == len(pos_index) ?

    print('linenum:', linenum) # total number of images
    ALL_index = list(range(linenum)) # can be used as indices for all images in dataset
    # print('ALL_index:', ALL_index)

    for pos_i in all_pos_index:
        ALL_index.remove(pos_i)

    all_neg_index = ALL_index # won't neg_index give the same thing ?
    print('all_neg_index len:', len(all_neg_index))

    print('all_pos_index:', all_pos_index)
    # print('all_neg_index:', neg_index_concept_labels)
    print('all_label_ind:', all_label_ind)
    return (pos_index_xnn,neg_index_xnn,pos_index_category, neg_index_category, pos_index, neg_index, all_label_ind, pos_index_concept_labels, neg_index_concept_labels, pos_fileName, neg_fileName)
    # return (all_pos_index, all_neg_index, all_label_ind, pos_index_concept_labels)


def Generate_onevsall_index(Cname, label_ind):

    all_pos_index = []
    #all_neg_index = []
    all_label_ind = []
    pos_len=0

    with open('data/thefilename.txt','r') as f1:
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


    #print('linevec:', linevec)
    #p_start = min(linevec)-1  # line-1
    #p_end = max(linevec)  # line

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



def Generate_onevsall_index_bad(Cname, label_ind):

    #CategoryListFile = '/media/phoenixqq/新加卷/Ubuntu_home/Caffe/ExcitationBP/ZhongangFinal_MainCUB/Data/CUB200_csvnamelist.csv'
    #CategoryListFile = 'Data/CUB200_Multi_csvnamelist.csv'                                                                    # 1 of 3

    # todo Mandana: I added this part for generating multi-label concepts
    concepts=[]
    all_pos_index = []
    #all_neg_index = []
    all_label_ind = []
    pos_len=0
    #neg_len=0


        #if rowi['Subfile']==None:
        #    continue
    totalConcepts =[]
    groundTruthListFile = '../../../dataExperts/Expert1/featureSelection=MSF_1/IFeature_imgfc6_threshold=0.3/groundTruthVisualWords_cat'+ str(label_ind)+'.csv'
    with open(groundTruthListFile, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader]

        for row in rows:
            if not int(row[' visual Word Inexd']) in totalConcepts:
                print(row[' visual Word Inexd'])
                totalConcepts.append(int(row[' visual Word Inexd']))

        # check how many concepts do we have here
    totalConcepts=sorted(totalConcepts, key=int)
    mulit_label_concepts=[]
    with open('../../../../DataSetforXAI/Results_newEBP/Model/thefilename.txt','r') as f1:
        linenum = 0
        pos_index = []
        neg_index = []
        pos_index_concept_labels = []
        neg_index_concept_labels = []

        #neg_index = []
        for line in f1.readlines():
            line = line.strip()
            #print(line)
            # find the example in the ground-truth file
            mulit_label_concepts=np.zeros(len(totalConcepts))
            for row in rows:
                if row[' FileName'][3:-4] in line:# find the file
                    mulit_label_concepts[  totalConcepts.index(int(row[' visual Word Inexd']))] =  1
                    # print(row[' visual Word Inexd'])

            if Cname in line:
                pos_index.append(linenum)
                pos_index_concept_labels.append(mulit_label_concepts)

            else:
                neg_index.append(linenum)
                neg_index_concept_labels.append(mulit_label_concepts)

            linenum = linenum + 1



    pos_len += len(pos_index)
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
    # print('ALL_index:', ALL_index)

    for pos_i in all_pos_index:
        ALL_index.remove(pos_i)

    all_neg_index = ALL_index
    print('all_neg_index len:', len(all_neg_index))

    print('all_pos_index:', all_pos_index)
    # print('all_neg_index:', neg_index_concept_labels)
    print('all_label_ind:', all_label_ind)
    return (pos_index, neg_index, all_label_ind, pos_index_concept_labels, neg_index_concept_labels)
    # return (all_pos_index, all_neg_index, all_label_ind, pos_index_concept_labels)

def Generate_multi_index(mainPath,CategoryListFile='Data/CUB200_Multi_csvnamelist.csv'):

    #CategoryListFile = '/media/phoenixqq/新加卷/Ubuntu_home/Caffe/ExcitationBP/ZhongangFinal_MainCUB/Data/CUB200_csvnamelist.csv'
    #CategoryListFile = 'Data/CUB200_Multi_csvnamelist.csv'                                                                    # 1 of 3

    with open(CategoryListFile, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader]

    all_pos_index = []
    #all_neg_index = []
    all_label_ind = []
    pos_len=0
    #neg_len=0
    for rowi in rows:

        #if rowi['Subfile']==None:
        #    continue

        print(rowi)
        Cname = rowi['Name']
        label_ind = rowi['No']
        label_ind = int(label_ind)
        #print(label_ind)

        with open( '../../../../DataSetforXAI/Results_newEBP/Model/thefilename.txt','r') as f1:
            linenum = 0
            pos_index = []
            #neg_index = []
            for line in f1.readlines():
                line = line.strip()
                print(line)
                if Cname in line:
                    pos_index.append(linenum)
                #else:
                #    neg_index.append(linenum)

                linenum = linenum + 1


        #print('linevec:', linevec)
        #p_start = min(linevec)-1  # line-1
        #p_end = max(linevec)  # line

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




def BIC(faithful_loss, MeanActivWeight, Neuron_n, parm):
    Base = np.sum(np.power(MeanActivWeight, 2))
    item0 = parm[0] * faithful_loss
    item1 = parm[1] * Neuron_n
    SortActivWeight = -np.sort(-MeanActivWeight)
    PositiveNum = np.sum(SortActivWeight > 0)
    item2 = - parm[2] * PositiveNum

    BIC_basic = Base +item0+item1
    BIC_plus = BIC_basic + item2
    return (BIC_basic, BIC_plus)

def pad(array, up, down, left, right):
    ########################
    # Create an array of zeros with the reference shape
    ####################################################
    newshapex = array.shape[0] + up + down
    newshapey = array.shape[1] + left + right
    result = np.zeros((newshapex, newshapey))

    result[up:array.shape[0] + up, left:array.shape[1] + left] = array
    return result


def showimgX(imgX, indexi, rowi):
    inx = int(imgX.size()[1])
    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    f, ax = plt.subplots(int(inx / rowi) + 1, rowi)

    for i in range(inx):
        ax[int(i / rowi), i % rowi].imshow(imgX[indexi, i].data.numpy(), cmap='gray')


def logpenaltyFC(x1, x2, Qq):
    ########################
    # loss function of log penalty for the fully-connected layer
    ####################################################
    nx2 = torch.mul(x2, -1)
    subx = torch.add(x1, nx2)

    powx = torch.pow(subx, 2)
    meanx = torch.mean(powx, 0)
    meanx = torch.mul(meanx, Qq)
    Qlog = torch.log1p(meanx)
    sploss = torch.mean(Qlog)
    return sploss


def PT_loss(feature_x, useGPU):
    ########################
    # pull-away term along the images
    ####################################################
    batch_size = feature_x.size(0) # 51
    denom = feature_x.norm(p=2, dim=1) # L2 norm of feature vector
    denom = torch.unsqueeze(denom, 1)
    denom = denom.expand_as(feature_x)
    feature_xNorm = torch.div(feature_x, denom)
    cosine = torch.mm(feature_xNorm, feature_xNorm.t())

    if useGPU == 1:
        mask = Variable((torch.ones(cosine.size()) - torch.diag(torch.ones(batch_size))).cuda())
    else:
        mask = Variable(torch.ones(cosine.size()) - torch.diag(torch.ones(batch_size)))

    pt_sum = torch.sum(torch.pow(torch.mul(cosine, mask), 2))
    pt_batch = batch_size * (batch_size - 1)
    pt_loss = torch.div(pt_sum, pt_batch)

    return pt_loss


def PT_loss_true(feature_x, useGPU):
    ########################
    # pull-away term along the features
    ####################################################
    feature_x = feature_x.t() #transpose (51x3) -> (3x51)
    batch_size = feature_x.size(0) # 3
    denom = feature_x.norm(p=2, dim=1) # L2 norm (3x51) -> (3,)
    denom = torch.unsqueeze(denom, 1) # (3x1)
    denom = denom.expand_as(feature_x) # (3x1x1) -> make 51 copies to get (3x51)

    feature_xNorm = torch.div(feature_x, denom) # elementwise division (3x51)

    cosine = torch.mm(feature_xNorm, feature_xNorm.t()) # matrix multiplication (3x3)

    if useGPU == 1:
        mask = Variable((torch.ones(cosine.size()) - torch.diag(torch.ones(batch_size))).cuda()) #(3x3) matrix of ones with zero diagonal
    else:
        mask = Variable(torch.ones(cosine.size()) - torch.diag(torch.ones(batch_size)))

    pt_sum = torch.sum(torch.pow(torch.mul(cosine, mask), 2)) # zero diagonal elements, square ans sum other elements (3x1)
    pt_batch = batch_size * (batch_size - 1) # 6
    pt_loss = torch.div(pt_sum, pt_batch)

    return pt_loss


def loadTags(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        data = list(reader)

    tagName = [r[0] for r in data]
    return tagName, dict(list(zip(tagName, list(range(len(tagName))))))


def getTagScore(scores, tags, tag2IDs):
    scores = np.exp(scores)
    scores /= scores.sum()
    tagScore = []
    for r in tags:
        tagScore.append((r, scores[tag2IDs[r]]))

    return tagScore


def showAttMap(img, attMaps, tagName, overlap=True, blur=False, sb=False):
    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    f, ax = plt.subplots(len(tagName) // 2 + 1, 2)
    ax[0, 0].imshow(img)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title("Original Image ", fontsize=10)

    for i in range(len(tagName)):
        attMap = attMaps[i].copy()

        if sb:
            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')
            ax[(i + 1) // 2, (i + 1) % 2].imshow(attMap)

        else:

            attMap -= attMap.min()
            if attMap.max() > 0:
                attMap /= attMap.max()

            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')

            if blur:
                attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
                attMap -= attMap.min()
                attMap /= attMap.max()

            cmap = plt.get_cmap('jet')
            attMapV = cmap(attMap)
            attMapV = np.delete(attMapV, 3, 2)
            if overlap:
                attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(
                    attMap.shape + (1,)) * attMapV;

            ax[(i + 1) // 2, (i + 1) % 2].imshow(attMap, interpolation='bicubic')

        ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d" % tagName[i], fontsize=10)
        ax[(i + 1) // 2, (i + 1) % 2].set_xticks([])
        ax[(i + 1) // 2, (i + 1) % 2].set_yticks([])


def showAttMap_padding(img, attMaps, tagName, tagweight, overlap=True, blur=False, sb=False):
    ########################
    # show attMaps for excitationBP with padding

    # Parameters:
    # -------------
    # img: the original image
    # attMaps: the attMaps for x-features in x-layer
    # tagName: the index of the x-features
    # tagweight: activation*weight for x-features
    # overlap: overlap or not
    # blur: blur or not
    # sb: if sb is True, only show the attmaps; else, show the attmaps combined with the original images
    ####################################################
    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    f, ax = plt.subplots(len(tagName) // 2 + 1, 2)
    ax[0, 0].imshow(img)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title("Original Image ", fontsize=10)

    for i in range(len(tagName)):
        attMap = attMaps[i].copy()

        if sb:
            attMap = pad(attMap, 30, 30, 30, 30)
            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')
            ax[(i + 1) // 2, (i + 1) % 2].imshow(attMap)

        else:

            attMap -= attMap.min()
            if attMap.max() > 0:
                attMap /= attMap.max()

            attMap = pad(attMap, 30, 10, 20, 20)
            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')

            if blur:
                attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
                attMap -= attMap.min()
                attMap /= attMap.max()

            cmap = plt.get_cmap('jet')
            attMapV = cmap(attMap)
            attMapV = np.delete(attMapV, 3, 2)
            if overlap:
                attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(
                    attMap.shape + (1,)) * attMapV;

            ax[(i + 1) // 2, (i + 1) % 2].imshow(attMap, interpolation='bicubic')

        ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[i]), fontsize=10)
        ax[(i + 1) // 2, (i + 1) % 2].set_xticks([])
        ax[(i + 1) // 2, (i + 1) % 2].set_yticks([])


def showAttMapNew(img, attMaps, tagName, tagweight, overlap=True, blur=False, sb=False):
    ########################
    # Show attMaps for excitationBP in pytorch

    # Parameters:
    # -------------
    # img: the original image
    # attMaps: the attMaps for x-features in x-layer
    # tagName: the index of the x-features: 0,1,2,...
    # tagweight: activation*weight for x-features
    # overlap: overlap or not
    # blur: blur or not
    # sb: if sb is True, only show the attmaps; else, show the attmaps combined with the original images
    ####################################################
    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    f, ax = plt.subplots(len(tagName) // 2 + 1, 2)
    ax[0, 0].imshow(img)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title("Original Image ", fontsize=15)

    for i in range(len(tagName)):
        attMap = attMaps[i].copy()

        # if sb is True, only show the attmaps
        # else, show the attmaps combined with the original images
        if sb:
            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')
            ax[(i + 1) // 2, (i + 1) % 2].imshow(attMap)

        else:

            attMap -= attMap.min()
            if attMap.max() > 0:
                attMap /= attMap.max()

            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')

            if blur:
                attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
                attMap -= attMap.min()
                attMap /= attMap.max()

            cmap = plt.get_cmap('jet')
            attMapV = cmap(attMap)
            attMapV = np.delete(attMapV, 3, 2)
            if overlap:
                attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(
                    attMap.shape + (1,)) * attMapV;

            ax[(i + 1) // 2, (i + 1) % 2].imshow(attMap, interpolation='bicubic')

        #tagweight[i] = tagweight[i] + 0 # because negative zero,we add 0
        tagweight = tagweight + 0  # because negative zero,we add 0
        #print(tagName.shape)
        print(tagweight.shape)
        print(str(tagweight[:,i]))
        tx = tagweight[:,i].copy()
        txstr = np.array2string(tx, formatter={'float_kind': lambda tx: "%.2f" % tx})
        print(txstr)
        #ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[i]), fontsize=15)
        #ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: " % tagName[i] +str(tagweight[:,i]), fontsize=11)
        ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: " % tagName[i] +txstr, fontsize=13)
        ax[(i + 1) // 2, (i + 1) % 2].set_xticks([])
        ax[(i + 1) // 2, (i + 1) % 2].set_yticks([])




def showAttMapSaeed(imgletter, img, attMaps, tagName, tagweight, overlap=True, blur=False, sb=False):
    ########################
    # Show attMaps for excitationBP in pytorch

    # Parameters:
    # -------------
    # img: the original image
    # attMaps: the attMaps for x-features in x-layer
    # tagName: the index of the x-features: 0,1,2,...
    # tagweight: activation*weight for x-features
    # overlap: overlap or not
    # blur: blur or not
    # sb: if sb is True, only show the attmaps; else, show the attmaps combined with the original images
    ####################################################
    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    col_num = len(tagName) // 2 + 1
    f, ax = plt.subplots(2, col_num)
    ax[0, 0].imshow(img)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title("Original Image ", fontsize=15)


    print('imgletter:', imgletter)
    if imgletter == 'a':
        img_j = 0
    elif imgletter == 'f':
        img_j = 1
    elif imgletter == 'h':
        img_j = 2
    elif imgletter == 'l':
        img_j = 3
    elif imgletter == 's':
        img_j = 4


    for i in range(len(tagName)):
        attMap = attMaps[i].copy()

        # if sb is True, only show the attmaps
        # else, show the attmaps combined with the original images
        if sb:
            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')
            ax[(i + 1) // col_num, (i + 1) % col_num].imshow(attMap)

        else:

            attMap -= attMap.min()
            if attMap.max() > 0:
                attMap /= attMap.max()

            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')

            if blur:
                attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
                attMap -= attMap.min()
                attMap /= attMap.max()

            cmap = plt.get_cmap('jet')
            attMapV = cmap(attMap)
            attMapV = np.delete(attMapV, 3, 2)
            if overlap:
                attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(
                    attMap.shape + (1,)) * attMapV;

            ax[(i + 1) // col_num, (i + 1) % col_num].imshow(attMap, interpolation='bicubic')

        #tagweight[i] = tagweight[i] + 0 # because negative zero,we add 0
        tagweight = tagweight + 0  # because negative zero,we add 0
        #print(tagName.shape)
        print(tagweight.shape)
        print(str(tagweight[:,i]))
        tx = tagweight[:,i].copy()
        txstr = np.array2string(tx, formatter={'float_kind': lambda tx: "%.2f" % tx})
        print(txstr)
        #ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[i]), fontsize=15)
        #ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: " % tagName[i] +str(tagweight[:,i]), fontsize=11)
        #ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: " % tagName[i] +txstr, fontsize=13)
        ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[img_j, i]), fontsize=13)
        ax[(i + 1) // col_num, (i + 1) % col_num].set_xticks([])
        ax[(i + 1) // col_num, (i + 1) % col_num].set_yticks([])



def showAttMap_xnn(img, attMap, tagName, overlap = True, blur = False):
    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    plt.imshow(img)
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'edge')

    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()

    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1*(1-attMap**0.8).reshape(attMap.shape + (1,))*img + (attMap**0.8).reshape(attMap.shape+(1,)) * attMapV;

    fig=plt.imshow(attMap, interpolation = 'bicubic')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
#    plt.set_title(tagName)


def showAttMapCUB(img_j, img, attMaps, tagName, tagweight, overlap=True, blur=False, sb=False, onevsall=0):
    ########################
    # Show attMaps for excitationBP in pytorch

    # Parameters:
    # -------------
    # img: the original image
    # attMaps: the attMaps for x-features in x-layer
    # tagName: the index of the x-features: 0,1,2,...
    # tagweight: activation*weight for x-features
    # overlap: overlap or not
    # blur: blur or not
    # sb: if sb is True, only show the attmaps; else, show the attmaps combined with the original images
    ####################################################
    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    col_num = len(tagName) // 2 + 1
    f, ax = plt.subplots(2, col_num)
    ax[0, 0].imshow(img)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title("Original Image ", fontsize=15)


    '''print('imgletter:', imgletter)
    if imgletter == 'a':
        img_j = 0
    elif imgletter == 'f':
        img_j = 1
    elif imgletter == 'h':
        img_j = 2
    elif imgletter == 'l':
        img_j = 3
    elif imgletter == 's':
        img_j = 4
    '''


    for i in range(len(tagName)):
        attMap = attMaps[i].copy()

        # if sb is True, only show the attmaps
        # else, show the attmaps combined with the original images
        if sb:
            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')
            ax[(i + 1) // col_num, (i + 1) % col_num].imshow(attMap)

        else:

            attMap -= attMap.min()
            if attMap.max() > 0:
                attMap /= attMap.max()

            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')

            if blur:
                attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
                attMap -= attMap.min()
                attMap /= attMap.max()

            cmap = plt.get_cmap('jet')
            attMapV = cmap(attMap)
            attMapV = np.delete(attMapV, 3, 2)
            if overlap:
                attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(
                    attMap.shape + (1,)) * attMapV;

            ax[(i + 1) // col_num, (i + 1) % col_num].imshow(attMap, interpolation='bicubic')

        #tagweight[i] = tagweight[i] + 0 # because negative zero,we add 0
        tagweight = tagweight + 0  # because negative zero,we add 0
        #print(tagName.shape)
        print(tagweight.shape)
        if onevsall == 1:
            Vtagweight = tagweight
        else:
            Vtagweight = tagweight[:,i]

        print(str(Vtagweight))
        tx = Vtagweight.copy()
        txstr = np.array2string(tx, formatter={'float_kind': lambda tx: "%.2f" % tx})
        print(txstr)
        #ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[i]), fontsize=15)
        #ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: " % tagName[i] +str(tagweight[:,i]), fontsize=11)
        #ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: " % tagName[i] +txstr, fontsize=13)
        #ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[img_j, i]), fontsize=13)
        ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: %.4f" % (tagName[i], Vtagweight[i]),
                                                            fontsize=13)
        ax[(i + 1) // col_num, (i + 1) % col_num].set_xticks([])
        ax[(i + 1) // col_num, (i + 1) % col_num].set_yticks([])



def showAttMapColor(imgletter, img, attMaps, tagName, tagweight, overlap=True, blur=False, sb=False):
    ########################
    # Show attMaps for excitationBP in pytorch

    # Parameters:
    # -------------
    # img: the original image
    # attMaps: the attMaps for x-features in x-layer
    # tagName: the index of the x-features: 0,1,2,...
    # tagweight: activation*weight for x-features
    # overlap: overlap or not
    # blur: blur or not
    # sb: if sb is True, only show the attmaps; else, show the attmaps combined with the original images
    ####################################################
    pylab.rcParams['figure.figsize'] = (12.0, 5.0)
    #pylab.rcParams['figure.figsize'] = (5.0, 5.0)
    col_num = len(tagName) // 2 + 1
    f, ax = plt.subplots(2, col_num)
    f.tight_layout()  # 调整整体空白
    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.93,
                wspace=0.05, hspace=0.25)  # 调整子图间距

    ax[0, 0].imshow(img)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title("Original Image ", fontsize=13)


    print('imgletter:', imgletter)
    if imgletter == 'a':
        img_j = 0
    elif imgletter == 'f':
        img_j = 1
    elif imgletter == 'h':
        img_j = 2
    elif imgletter == 'l':
        img_j = 3
    elif imgletter == 's':
        img_j = 4


    tagindex = np.argsort(-tagweight)
    print('tagweight:', tagweight)
    print('tagindex:', tagindex)

    for i in range(len(tagName)):
        attMap = attMaps[i].copy()

        # if sb is True, only show the attmaps
        # else, show the attmaps combined with the original images
        if sb:
            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')
            ax[(i + 1) // col_num, (i + 1) % col_num].imshow(attMap)


        else:

            attMap -= attMap.min()
            if attMap.max() > 0:
                attMap /= attMap.max()

            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')

            if blur:
                attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
                attMap -= attMap.min()
                attMap /= attMap.max()

            cmap = plt.get_cmap('jet')
            attMapV = cmap(attMap)
            attMapV = np.delete(attMapV, 3, 2)
            if overlap:
                attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(
                    attMap.shape + (1,)) * attMapV;

            ax[(i + 1) // col_num, (i + 1) % col_num].imshow(attMap, interpolation='bicubic')


        #tagweight[i] = tagweight[i] + 0 # because negative zero,we add 0
        tagweight = tagweight + 0  # because negative zero,we add 0
        #print(tagName.shape)
        print(tagweight.shape)
        print(str(tagweight[:,i]))
        tx = tagweight[:,i].copy()
        txstr = np.array2string(tx, formatter={'float_kind': lambda tx: "%.2f" % tx})
        print(txstr)
        #ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[i]), fontsize=15)
        #ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: " % tagName[i] +str(tagweight[:,i]), fontsize=11, fontweight='bold')
        #ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: " % tagName[i] +txstr, fontsize=13, fontweight='black')
        if i==tagindex[img_j, 0]:
            ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[img_j, i]),
                                                                fontsize=13, color=(0.54, 0, 0), fontweight='bold')
        #color = (0.69, 0.09, 0.12)
        #color=(1,0,0)
        #color = (0.94, 0.5, 0.5)
        #color = (1, 0.39, 0.28)
        #elif i!=tagindex[img_j, 0] and tagweight[img_j, i]>=3:
        elif i==tagindex[img_j,1] and tagweight[img_j, i] > 0:
            ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[img_j, i]),
                                                                fontsize=13, color='red', fontweight ='bold')
        elif tagweight[img_j, i]>0:
            ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[img_j, i]),
                                                                fontsize=13, color=(1, 0.39, 0.28))
        elif i==tagindex[img_j,-1] and tagweight[img_j, i] < 0:
            ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[img_j, i]),
                                                                fontsize=13, color=(0, 0, 0.54), fontweight ='bold')
        elif i==tagindex[img_j,-2] and tagweight[img_j, i] < 0:
            ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[img_j, i]),
                                                                fontsize=13, color='blue', fontweight ='bold')
        elif tagweight[img_j, i]<0:
            ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[img_j, i]),
                                                                fontsize=13, color=(0, 0.75, 1))
        else:
            ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[img_j, i]),
                                                                fontsize=13)
        ax[(i + 1) // col_num, (i + 1) % col_num].set_xticks([])
        ax[(i + 1) // col_num, (i + 1) % col_num].set_yticks([])




def showAttMapColorAdapt(imgletter, img, attMaps, tagName, tagweight, overlap=True, blur=False, sb=False):
    ########################
    # Show attMaps for excitationBP in pytorch

    # Parameters:
    # -------------
    # img: the original image
    # attMaps: the attMaps for x-features in x-layer
    # tagName: the index of the x-features: 0,1,2,...
    # tagweight: activation*weight for x-features
    # overlap: overlap or not
    # blur: blur or not
    # sb: if sb is True, only show the attmaps; else, show the attmaps combined with the original images
    ####################################################
    pylab.rcParams['figure.figsize'] = (12.0, 5.0)
    #pylab.rcParams['figure.figsize'] = (5.0, 5.0)
    col_num = len(tagName) // 2 + 1
    f, ax = plt.subplots(2, col_num)
    f.tight_layout()  # 调整整体空白
    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.93,
                wspace=0.05, hspace=0.25)  # 调整子图间距

    ax[0, 0].imshow(img)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title("Original Image ", fontsize=13)


    print('imgletter:', imgletter)
    if imgletter == 'a':
        img_j = 0
    elif imgletter == 'f':
        img_j = 1
    elif imgletter == 'h':
        img_j = 2
    elif imgletter == 'l':
        img_j = 3
    elif imgletter == 's':
        img_j = 4


    tagindex = np.argsort(-tagweight)
    print('tagweight:', tagweight)
    print('tagindex:', tagindex)

    for i in range(len(tagName)):
        attMap = attMaps[i].copy()

        # if sb is True, only show the attmaps
        # else, show the attmaps combined with the original images
        if sb:
            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')
            ax[(i + 1) // col_num, (i + 1) % col_num].imshow(attMap)


        else:

            attMap -= attMap.min()
            if attMap.max() > 0:
                attMap /= attMap.max()

            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')

            if blur:
                attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
                attMap -= attMap.min()
                attMap /= attMap.max()

            cmap = plt.get_cmap('jet')
            attMapV = cmap(attMap)
            attMapV = np.delete(attMapV, 3, 2)
            if overlap:
                attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(
                    attMap.shape + (1,)) * attMapV;

            ax[(i + 1) // col_num, (i + 1) % col_num].imshow(attMap, interpolation='bicubic')


        #tagweight[i] = tagweight[i] + 0 # because negative zero,we add 0
        tagweight = tagweight + 0  # because negative zero,we add 0
        #print(tagName.shape)
        #print(tagweight.shape)
        #print(str(tagweight[:,i]))
        tx = tagweight[:,i].copy()
        txstr = np.array2string(tx, formatter={'float_kind': lambda tx: "%.2f" % tx})
        #print(txstr)
        #ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[i]), fontsize=15)
        #ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: " % tagName[i] +str(tagweight[:,i]), fontsize=11, fontweight='bold')
        #ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: " % tagName[i] +txstr, fontsize=13, fontweight='black')

        tagmax = tagweight[img_j, tagindex[img_j, 0]]
        tagmin = tagweight[img_j, tagindex[img_j,-1]]
        #print('tagmax:',tagmax, 'tagmin:', tagmin)

        if tagweight[img_j, i] > 0:
            #print('check:', tagweight[img_j, i]/tagmax)
            colorx = (1- tagweight[img_j, i]/tagmax) *200.0/255.0
            ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Heatmap %d: %.4f" % (tagName[i], tagweight[img_j, i]),
                                                                fontsize=13, color=(1, colorx, colorx), fontweight='bold')
        elif tagweight[img_j, i] < 0:
            colorx = (1-tagweight[img_j, i]/tagmin) *200.0/255.0
            ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Heatmap %d: %.4f" % (tagName[i], tagweight[img_j, i]),
                                                                fontsize=13, color=(colorx, colorx, 1), fontweight='bold')
        else:
            ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Heatmap %d: %.4f" % (tagName[i], tagweight[img_j, i]),
                                                                fontsize=13, color=(0.8, 0.8, 1), fontweight='bold')

        ax[(i + 1) // col_num, (i + 1) % col_num].set_xticks([])
        ax[(i + 1) // col_num, (i + 1) % col_num].set_yticks([])



def showAttMapColorAdaptOneline(imgletter, img, attMaps, tagName, tagweight, overlap=True, blur=False, sb=False):
    ########################
    # Show attMaps for excitationBP in pytorch

    # Parameters:
    # -------------
    # img: the original image
    # attMaps: the attMaps for x-features in x-layer
    # tagName: the index of the x-features: 0,1,2,...
    # tagweight: activation*weight for x-features
    # overlap: overlap or not
    # blur: blur or not
    # sb: if sb is True, only show the attmaps; else, show the attmaps combined with the original images
    ####################################################
    pylab.rcParams['figure.figsize'] = (22.0, 2.5)
    #pylab.rcParams['figure.figsize'] = (5.0, 5.0)
    col_num = len(tagName) + 1
    f, ax = plt.subplots(1, col_num)
    f.tight_layout()  # 调整整体空白
    plt.subplots_adjust(left=0.005, bottom=0.02, right=0.995, top=0.93,
                wspace=0.05, hspace=0.25)  # 调整子图间距

    ax[0].imshow(img)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("Original Image ", fontsize=13)


    print('imgletter:', imgletter)
    if imgletter == 'a':
        img_j = 0
    elif imgletter == 'f':
        img_j = 1
    elif imgletter == 'h':
        img_j = 2
    elif imgletter == 'l':
        img_j = 3
    elif imgletter == 's':
        img_j = 4


    tagindex = np.argsort(-tagweight)
    tagabsmax = np.sort(np.abs(tagweight[img_j, :]))[-1]
    print('tagabsmax:',tagabsmax)
    print('tagweight:', tagweight)
    print('tagindex:', tagindex)

    for i in range(len(tagName)):
        attMap = attMaps[i].copy()

        # if sb is True, only show the attmaps
        # else, show the attmaps combined with the original images
        if sb:
            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')
            ax[i+1].imshow(attMap)


        else:

            attMap -= attMap.min()
            if attMap.max() > 0:
                attMap /= attMap.max()

            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')

            if blur:
                attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
                attMap -= attMap.min()
                attMap /= attMap.max()

            cmap = plt.get_cmap('jet')
            attMapV = cmap(attMap)
            attMapV = np.delete(attMapV, 3, 2)
            if overlap:
                attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(
                    attMap.shape + (1,)) * attMapV;

            ax[i+1].imshow(attMap, interpolation='bicubic')


        #tagweight[i] = tagweight[i] + 0 # because negative zero,we add 0
        tagweight = tagweight + 0  # because negative zero,we add 0
        #print(tagName.shape)
        #print(tagweight.shape)
        #print(str(tagweight[:,i]))
        tx = tagweight[:,i].copy()
        txstr = np.array2string(tx, formatter={'float_kind': lambda tx: "%.2f" % tx})
        #print(txstr)
        #ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[i]), fontsize=15)
        #ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: " % tagName[i] +str(tagweight[:,i]), fontsize=11, fontweight='bold')
        #ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: " % tagName[i] +txstr, fontsize=13, fontweight='black')

        tagmax = tagweight[img_j, tagindex[img_j, 0]]
        tagmin = tagweight[img_j, tagindex[img_j,-1]]
        #print('tagmax:',tagmax, 'tagmin:', tagmin)

        titlevalue = tagweight[img_j, i]/tagabsmax

        if tagweight[img_j, i] > 0:
            #print('check:', tagweight[img_j, i]/tagmax)
            colorx = (1- tagweight[img_j, i]/tagmax) *200.0/255.0
            ax[i+1].set_title("Heatmap %d: %.4f" % (tagName[i], titlevalue),
                                                                fontsize=13, color=(1, colorx, colorx), fontweight='bold')
        elif tagweight[img_j, i] < 0:
            colorx = (1-tagweight[img_j, i]/tagmin) *200.0/255.0
            ax[i+1].set_title("Heatmap %d: %.4f" % (tagName[i], titlevalue),
                                                                fontsize=13, color=(colorx, colorx, 1), fontweight='bold')
        else:
            ax[i+1].set_title("Heatmap %d: %.4f" % (tagName[i], titlevalue),
                                                                fontsize=13, color=(0.8, 0.8, 1), fontweight='bold')

        ax[i+1].set_xticks([])
        ax[i+1].set_yticks([])

def showAttMapColorAdaptFinal(imgletter, img, attMaps, tagName, tagweight, tagweight_index, target_path, imgName, overlap=True, blur=False, sb=False):
    ########################
    # Show attMaps for excitationBP in pytorch

    # Parameters:
    # -------------
    # img: the original image
    # attMaps: the attMaps for x-features in x-layer
    # tagName: the index of the x-features: 0,1,2,...
    # tagweight: activation*weight for x-features
    # overlap: overlap or not
    # blur: blur or not
    # sb: if sb is True, only show the attmaps; else, show the attmaps combined with the original images
    ####################################################



    print('imgletter:', imgletter)
    if imgletter == 'a':
        img_j = 0
    elif imgletter == 'f':
        img_j = 1
    elif imgletter == 'h':
        img_j = 2
    elif imgletter == 'l':
        img_j = 3
    elif imgletter == 's':
        img_j = 4


    tagindex = np.argsort(-tagweight)
    tagabsmax = np.sort(np.abs(tagweight[img_j, :]))[-1]
    print('tagabsmax:',tagabsmax)
    print('tagweight:', tagweight)
    print('tagindex:', tagindex)



    for i in range(len(tagName)):
        real_i = tagweight_index[i]
        print(real_i)
        attMap = attMaps[real_i].copy()

        # if sb is True, only show the attmaps
        # else, show the attmaps combined with the original images
        if sb:
            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')
            fig = plt.imshow(attMap)



        else:

            attMap -= attMap.min()
            if attMap.max() > 0:
                attMap /= attMap.max()

            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')

            if blur:
                attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
                attMap -= attMap.min()
                attMap /= attMap.max()

            cmap = plt.get_cmap('jet')
            attMapV = cmap(attMap)
            attMapV = np.delete(attMapV, 3, 2)
            if overlap:
                attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(
                    attMap.shape + (1,)) * attMapV;

            fig = plt.imshow(attMap, interpolation='bicubic')




        tagweight = tagweight + 0  # because negative zero,we add 0
        tx = tagweight[:,real_i].copy()
        txstr = np.array2string(tx, formatter={'float_kind': lambda tx: "%.2f" % tx})

        tagmax = tagweight[img_j, tagindex[img_j, 0]]
        tagmin = tagweight[img_j, tagindex[img_j,-1]]
        #print('tagmax:',tagmax, 'tagmin:', tagmin)

        titlevalue = tagweight[img_j, real_i]/tagabsmax

        if tagweight[img_j, real_i] > 0:
            #print('check:', tagweight[img_j, i]/tagmax)
            colorx = (1- tagweight[img_j, real_i]/tagmax) *200.0/255.0
            plt.title("Heatmap %d: %.4f" % (tagName[i], titlevalue),
                                                                fontsize=13, color=(1, colorx, colorx), fontweight='bold')
        elif tagweight[img_j, real_i] < 0:
            colorx = (1-tagweight[img_j, real_i]/tagmin) *200.0/255.0
            plt.title("Heatmap %d: %.4f" % (tagName[i], titlevalue),
                                                                fontsize=13, color=(colorx, colorx, 1), fontweight='bold')
        else:
            plt.title("Heatmap %d: %.4f" % (tagName[i], titlevalue),
                                                                fontsize=13, color=(0.8, 0.8, 1), fontweight='bold')

        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.savefig(target_path + 'Reslut_' + imgName[:-4] + str(tagName[i]) +'_X'+str(tagName[real_i])+'.png', bbox_inches='tight', pad_inches=0)
        #plt.savefig(target_path + 'Reslut_' + imgName, bbox_inches='tight', pad_inches=0)





def showAttMapColorAdaptFinal2(imgletter, img, attMaps, tagName, tagweight, tagweight_index, target_path, imgName, overlap=True, blur=False, sb=False):
    ########################
    # Show attMaps for excitationBP in pytorch

    # Parameters:
    # -------------
    # img: the original image
    # attMaps: the attMaps for x-features in x-layer
    # tagName: the index of the x-features: 0,1,2,...
    # tagweight: activation*weight for x-features
    # overlap: overlap or not
    # blur: blur or not
    # sb: if sb is True, only show the attmaps; else, show the attmaps combined with the original images
    ####################################################



    print('imgletter:', imgletter)
    if imgletter == 'a':
        img_j = 0
    elif imgletter == 'f':
        img_j = 1
    elif imgletter == 'h':
        img_j = 2
    elif imgletter == 'l':
        img_j = 3
    elif imgletter == 's':
        img_j = 4


    tagindex = np.argsort(-tagweight)
    tagabsmax = np.sort(np.abs(tagweight[img_j, :]))[-1]
    print('tagabsmax:',tagabsmax)
    print('tagweight:', tagweight)
    print('tagindex:', tagindex)

    titlenumber = []
    tagweight_sorted = []

    for i in range(len(tagName)):
        real_i = tagweight_index[i]
        print(real_i)
        attMap = attMaps[real_i].copy()

        # if sb is True, only show the attmaps
        # else, show the attmaps combined with the original images
        if sb:
            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')
            fig = plt.imshow(attMap)



        else:

            attMap -= attMap.min()
            if attMap.max() > 0:
                attMap /= attMap.max()

            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')

            if blur:
                attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
                attMap -= attMap.min()
                attMap /= attMap.max()

            cmap = plt.get_cmap('jet')
            attMapV = cmap(attMap)
            attMapV = np.delete(attMapV, 3, 2)
            if overlap:
                attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(
                    attMap.shape + (1,)) * attMapV;

            #fig = plt.imshow(attMap, interpolation='bicubic')




        tagweight = tagweight + 0  # because negative zero,we add 0
        tx = tagweight[:,real_i].copy()
        txstr = np.array2string(tx, formatter={'float_kind': lambda tx: "%.2f" % tx})

        tagmax = tagweight[img_j, tagindex[img_j, 0]]
        tagmin = tagweight[img_j, tagindex[img_j,-1]]
        print('tagmax:',tagmax, 'tagmin:', tagmin)

        titlevalue = tagweight[img_j, real_i]/tagabsmax

        if tagweight[img_j, real_i] > 0:
            #print('check:', tagweight[img_j, i]/tagmax)
            #print('note:', np.abs(tagweight[img_j, real_i]/tagabsmax))
            attMap = attMap*(0.3 + 0.7*np.abs(tagweight[img_j, real_i]/tagabsmax))
            fig = plt.imshow(attMap, interpolation='bicubic')

            colorx = (1- tagweight[img_j, real_i]/tagmax) *200.0/255.0
            #plt.title("Heatmap %d: %.2f" % (tagName[i]+1, titlevalue),
            #                                                    fontsize=22, color=(1, colorx, colorx), fontweight='bold')
            plt.title("Contribution: %.2f" %titlevalue,
                      fontsize=22, color=(1, colorx, colorx), fontweight='bold')
        elif tagweight[img_j, real_i] < 0:
            #print('note:', np.abs(tagweight[img_j, real_i] / tagabsmax))
            attMap = attMap * (0.3 + 0.7* np.abs(tagweight[img_j, real_i] / tagabsmax))
            fig = plt.imshow(attMap, interpolation='bicubic')

            colorx = (1-tagweight[img_j, real_i]/tagmin) *200.0/255.0
            #plt.title("Heatmap %d: %.2f" % (tagName[i]+1, titlevalue),
            #                                                    fontsize=22, color=(colorx, colorx, 1), fontweight='bold')
            plt.title("Contribution: %.2f" %titlevalue,
                      fontsize=22, color=(colorx, colorx, 1), fontweight='bold')
        else:

            attMap = attMap * 0.3
            fig = plt.imshow(attMap, interpolation='bicubic')

            #plt.title("Heatmap %d: %.2f" % (tagName[i]+1, titlevalue),
            #                                                    fontsize=22, color=(0.8, 0.8, 1), fontweight='bold')
            plt.title("Contribution: %.2f" %titlevalue,
                      fontsize=22, color=(0.8, 0.8, 1), fontweight='bold')

        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.savefig(target_path + 'Reslut_' + imgName[:-4] + '_' + str(tagName[i]) +'_X'+str(tagName[real_i])+'.png', bbox_inches='tight', pad_inches=0)
        #plt.savefig(target_path + 'Reslut_' + imgName, bbox_inches='tight', pad_inches=0)

        titlenumber.append(titlevalue)
        tagweight_sorted.append(tagweight[img_j, real_i])

    print('titlenumber:', titlenumber)
    print('tagweight_sorted:', tagweight_sorted)
    scio.savemat(target_path + 'Xweight_' + imgName[:-4] + '.mat',
                 mdict={'titlenumber': titlenumber, 'tagweight_sorted': tagweight_sorted}, oned_as='column')







def showAttMapColorAdaptFinalInterface(img_j, img, attMaps, tagName, tagweight, tagweight_index, target_path, imgName, overlap=True, blur=False, sb=False):
    ########################
    # Show attMaps for excitationBP in pytorch

    # Parameters:
    # -------------
    # img: the original image
    # attMaps: the attMaps for x-features in x-layer
    # tagName: the index of the x-features: 0,1,2,...
    # tagweight: activation*weight for x-features
    # overlap: overlap or not
    # blur: blur or not
    # sb: if sb is True, only show the attmaps; else, show the attmaps combined with the original images
    ####################################################


    tagindex = np.argsort(-tagweight)
    tagabsmax = np.sort(np.abs(tagweight[img_j, :]))[-1]
    print('tagabsmax:',tagabsmax)
    print('tagweight:', tagweight)
    print('tagindex:', tagindex)

    titlenumber = []
    tagweight_sorted = []

    for i in range(len(tagName)):
        real_i = tagweight_index[i]
        print(real_i)
        attMap = attMaps[real_i].copy()

        # if sb is True, only show the attmaps
        # else, show the attmaps combined with the original images
        if sb:
            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')
            fig = plt.imshow(attMap)



        else:

            attMap -= attMap.min()
            if attMap.max() > 0:
                attMap /= attMap.max()

            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')

            if blur:
                attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
                attMap -= attMap.min()
                attMap /= attMap.max()

            cmap = plt.get_cmap('jet')
            attMapV = cmap(attMap)
            attMapV = np.delete(attMapV, 3, 2)
            if overlap:
                attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(
                    attMap.shape + (1,)) * attMapV;

            #fig = plt.imshow(attMap, interpolation='bicubic')




        tagweight = tagweight + 0  # because negative zero,we add 0
        tx = tagweight[:,real_i].copy()
        txstr = np.array2string(tx, formatter={'float_kind': lambda tx: "%.2f" % tx})

        tagmax = tagweight[img_j, tagindex[img_j, 0]]
        tagmin = tagweight[img_j, tagindex[img_j,-1]]
        print('tagmax:',tagmax, 'tagmin:', tagmin)

        titlevalue = tagweight[img_j, real_i]/tagabsmax

        if tagweight[img_j, real_i] > 0:
            #print('check:', tagweight[img_j, i]/tagmax)
            #print('note:', np.abs(tagweight[img_j, real_i]/tagabsmax))
            attMap = attMap*(0.3 + 0.7*np.abs(tagweight[img_j, real_i]/tagabsmax))
            fig = plt.imshow(attMap, interpolation='bicubic')

            #colorx = (1- tagweight[img_j, real_i]/tagmax) *200.0/255.0
            colorx = (1 - np.abs(titlevalue)) * 200.0 / 255.0
            #plt.title("Heatmap %d: %.2f" % (tagName[i]+1, titlevalue),
            #                                                    fontsize=22, color=(1, colorx, colorx), fontweight='bold')
            plt.title("Contribution: %.2f" %titlevalue,
                      fontsize=22, color=(1, colorx, colorx), fontweight='bold')
        elif tagweight[img_j, real_i] < 0:
            #print('note:', np.abs(tagweight[img_j, real_i] / tagabsmax))
            attMap = attMap * (0.3 + 0.7* np.abs(tagweight[img_j, real_i] / tagabsmax))
            fig = plt.imshow(attMap, interpolation='bicubic')

            #colorx = (1-tagweight[img_j, real_i]/tagmin) *200.0/255.0
            colorx = (1 - np.abs(titlevalue)) * 200.0 / 255.0
            #plt.title("Heatmap %d: %.2f" % (tagName[i]+1, titlevalue),
            #                                                    fontsize=22, color=(colorx, colorx, 1), fontweight='bold')
            plt.title("Contribution: %.2f" %titlevalue,
                      fontsize=22, color=(colorx, colorx, 1), fontweight='bold')
        else:

            attMap = attMap * 0.3
            fig = plt.imshow(attMap, interpolation='bicubic')

            #plt.title("Heatmap %d: %.2f" % (tagName[i]+1, titlevalue),
            #                                                    fontsize=22, color=(0.8, 0.8, 1), fontweight='bold')
            plt.title("Contribution: %.2f" %titlevalue,
                      fontsize=22, color=(0.8, 0.8, 1), fontweight='bold')

        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.savefig(target_path + 'Reslut_' + imgName[:-4] + '_' + str(tagName[i]) +'_X'+str(tagName[real_i])+'.png', bbox_inches='tight', pad_inches=0)
        #plt.savefig(target_path + 'Reslut_' + imgName, bbox_inches='tight', pad_inches=0)

        titlenumber.append(titlevalue)
        tagweight_sorted.append(tagweight[img_j, real_i])

    print('titlenumber:', titlenumber)
    print('tagweight_sorted:', tagweight_sorted)
    scio.savemat(target_path + 'Xweight_' + imgName[:-4] + '.mat',
                 mdict={'titlenumber': titlenumber, 'tagweight_sorted': tagweight_sorted}, oned_as='column')



def showAttMapColorPositive(imgletter, img, attMaps, tagName, tagweight, overlap=True, blur=False, sb=False):
    ########################
    # Show attMaps for excitationBP in pytorch

    # Parameters:
    # -------------
    # img: the original image
    # attMaps: the attMaps for x-features in x-layer
    # tagName: the index of the x-features: 0,1,2,...
    # tagweight: activation*weight for x-features
    # overlap: overlap or not
    # blur: blur or not
    # sb: if sb is True, only show the attmaps; else, show the attmaps combined with the original images
    ####################################################
    print('imgletter:', imgletter)
    if imgletter == 'a':
        img_j = 0
    elif imgletter == 'f':
        img_j = 1
    elif imgletter == 'h':
        img_j = 2
    elif imgletter == 'l':
        img_j = 3
    elif imgletter == 's':
        img_j = 4


    posnumber = len((np.where(tagweight[img_j,:]>0))[0])
    print('posnumber:', posnumber, np.where(tagweight[img_j,:]>0)[0], tagweight[img_j,:])


    pylab.rcParams['figure.figsize'] = (12.0, 3.5)
    #pylab.rcParams['figure.figsize'] = (5.0, 5.0)
    #col_num = len(tagName) // 2 + 1
    f, ax = plt.subplots(1, posnumber+1)
    f.tight_layout()  # 调整整体空白
    #plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.93,
    #            wspace=0.05, hspace=0.25)  # 调整子图间距
    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.93,
                        wspace=0.05, hspace=0)  # 调整子图间距

    ax[0].imshow(img)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("Original Image ", fontsize=15)





    tagindex = np.argsort(-tagweight)
    print('tagweight:', tagweight)
    print('tagindex:', tagindex)

    pos_sum = 0

    for i in range(len(tagName)):
        if tagweight[img_j, i] > 0:
            pos_sum = pos_sum + tagweight[img_j, i]

    print('pos_sum:', pos_sum)


    pos_i = 1
    for i in range(len(tagName)):
        if tagweight[img_j, i] > 0:
            attMap = attMaps[i].copy()

            attMap -= attMap.min()
            if attMap.max() > 0:
                attMap /= attMap.max()

            attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')

            if blur:
                attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
                attMap -= attMap.min()
                attMap /= attMap.max()

            cmap = plt.get_cmap('jet')
            attMapV = cmap(attMap)
            attMapV = np.delete(attMapV, 3, 2)
            if overlap:
                attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(
                    attMap.shape + (1,)) * attMapV;



            #tagweight[i] = tagweight[i] + 0 # because negative zero,we add 0
            tagweight = tagweight + 0  # because negative zero,we add 0
            #print(tagName.shape)
            #print(tagweight.shape)
            #print(str(tagweight[:,i]))
            tx = tagweight[:,i].copy()
            txstr = np.array2string(tx, formatter={'float_kind': lambda tx: "%.2f" % tx})
            #print(txstr)
            #ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: %.4f" % (tagName[i], tagweight[i]), fontsize=15)
            #ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron %d: " % tagName[i] +str(tagweight[:,i]), fontsize=11, fontweight='bold')
            #ax[(i + 1) // col_num, (i + 1) % col_num].set_title("Neuron %d: " % tagName[i] +txstr, fontsize=13, fontweight='black')

            tagmax = tagweight[img_j, tagindex[img_j, 0]]
            tagmin = tagweight[img_j, tagindex[img_j,-1]]
            #print('tagmax:',tagmax, 'tagmin:', tagmin)


            #print('check:', tagweight[img_j, i]/tagmax)
            ax[pos_i].imshow(attMap, interpolation='bicubic')
            colorx = (1- tagweight[img_j, i]/tagmax) *200.0/255.0
            #ax[pos_i].set_title("Heatmap %d: %.4f" % (tagName[i], tagweight[img_j, i]),
            #                                                    fontsize=13, color=(1, colorx, colorx), fontweight='bold')

            title_value = tagweight[img_j, i] /pos_sum *100
            ax[pos_i].set_title("%.2f%%" %title_value,
                                fontsize=15, color=(1, colorx, colorx), fontweight='bold')
            ax[pos_i].set_xticks([])
            ax[pos_i].set_yticks([])

            pos_i = pos_i + 1





def showAttMapEXBP(img, attMaps, overlap=True, blur=False, sb=False):
    ########################
    # Show attMaps for excitationBP in pytorch

    # Parameters:
    # -------------
    # img: the original image
    # attMaps: the attMaps for x-features in x-layer
    # tagName: the index of the x-features: 0,1,2,...
    # tagweight: activation*weight for x-features
    # overlap: overlap or not
    # blur: blur or not
    # sb: if sb is True, only show the attmaps; else, show the attmaps combined with the original images
    ####################################################
    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("Original Image ", fontsize=15)






    attMap = attMaps.copy()

    # if sb is True, only show the attmaps
    # else, show the attmaps combined with the original images
    if sb:
        attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')
        ax[0, 1].imshow(attMap)

    else:

        attMap -= attMap.min()
        if attMap.max() > 0:
            attMap /= attMap.max()

        attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')

        if blur:
            attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
            attMap -= attMap.min()
            attMap /= attMap.max()

        cmap = plt.get_cmap('jet')
        attMapV = cmap(attMap)
        attMapV = np.delete(attMapV, 3, 2)
        if overlap:
            attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(
                attMap.shape + (1,)) * attMapV;

        ax[1].imshow(attMap, interpolation='bicubic')


    ax[1].set_title("ExcitationBP", fontsize=15)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

def showAttMapEXBP_One(img, attMaps, overlap=True, blur=False, sb=False):
    ########################
    # Show attMaps for excitationBP in pytorch

    # Parameters:
    # -------------
    # img: the original image
    # attMaps: the attMaps for x-features in x-layer
    # tagName: the index of the x-features: 0,1,2,...
    # tagweight: activation*weight for x-features
    # overlap: overlap or not
    # blur: blur or not
    # sb: if sb is True, only show the attmaps; else, show the attmaps combined with the original images
    ####################################################
    #pylab.rcParams['figure.figsize'] = (3.5, 3.5)

    #ax = plt.plot
    #ax[0].imshow(img)
    #ax[0].set_xticks([])
    #ax[0].set_yticks([])
    #ax[0].set_title("Original Image ", fontsize=15)






    attMap = attMaps.copy()

    # if sb is True, only show the attmaps
    # else, show the attmaps combined with the original images
    if sb:
        attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')
        fig = plt.imshow(attMap)


    else:

        attMap -= attMap.min()
        if attMap.max() > 0:
            attMap /= attMap.max()

        attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')

        if blur:
            attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
            attMap -= attMap.min()
            attMap /= attMap.max()

        cmap = plt.get_cmap('jet')
        attMapV = cmap(attMap)
        attMapV = np.delete(attMapV, 3, 2)
        if overlap:
            attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(
                attMap.shape + (1,)) * attMapV;

        fig = plt.imshow(attMap, interpolation='bicubic')
    #plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #plt.margins(0, 0)
    #fig.savefig(out_png_path, format='png', transparent=True, dpi=300, pad_inches=0)



    #ax.set_title("ExcitationBP", fontsize=15)
    #plt.title("ExcitationBP", fontsize=15)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    #plt.tight_layout()
    #ax = plt.gca()
    #ax.set_xticks([])
    #ax.set_yticks([])


def showAttMapcaffe(img, attMaps, tagName, scores, overlap=True, blur=False):
    ########################
    # show attMaps for excitationBP in caffe

    # Parameters:
    # -------------
    # img: the original image
    # attMaps: the attMaps for x-features in x-layer
    # tagName: the index of the x-features: 1,2,...
    # scores: activation*weight for x-features
    # overlap: overlap or not
    # blur: blur or not
    ####################################################
    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    f, ax = plt.subplots(len(tagName) // 2 + 1, 2)
    ax[0, 0].imshow(img)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title("Original Image ", fontsize=15)

    for i in range(len(tagName)):
        attMap = attMaps[i].copy()
        attMap -= attMap.min()
        if attMap.max() > 0:
            attMap /= attMap.max()

        attMap = transform.resize(attMap, (img.shape[:2]), order=3, mode='edge')
        if blur:
            attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
            attMap -= attMap.min()
            attMap /= attMap.max()

        cmap = plt.get_cmap('jet')
        attMapV = cmap(attMap)
        attMapV = np.delete(attMapV, 3, 2)
        if overlap:
            attMap = 1 * (1 - attMap ** 0.8).reshape(attMap.shape + (1,)) * img + (attMap ** 0.8).reshape(
                attMap.shape + (1,)) * attMapV;

        ax[(i + 1) // 2, (i + 1) % 2].imshow(attMap, interpolation='bicubic')

        # because negative zero,we add 0
        sindex = scores[i] + 0
        ax[(i + 1) // 2, (i + 1) % 2].set_title("Neuron " + tagName[i] + ": %.4f" % sindex, fontsize=15)
        ax[(i + 1) // 2, (i + 1) % 2].set_xticks([])
        ax[(i + 1) // 2, (i + 1) % 2].set_yticks([])
