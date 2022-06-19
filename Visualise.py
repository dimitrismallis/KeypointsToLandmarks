import argparse
import yaml
from configuration import Configuration
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2   
from Database import Database
# import Test
import numpy as np
import random
from utils import *



def ShowTestExamples(keypoints,log_path,experiment_name,number_of_clusters,dataset_name,metadata,imagefile_name='Stage2.jpg',sortimages=True,mycolors=None,pointsize=400,showgt=False):

    fig = plt.figure(figsize=(34,55))
    gs1 = gridspec.GridSpec(13, 8)
    gs1.update(wspace=0.0, hspace=0.0)

    filenames=[k for k in keypoints.keys() if keypoints[k]['is_it_test_sample']]

    if(sortimages):
        filenames.sort()
    else:
        permute=np.arange(len(filenames))
        np.random.seed(42) 
        np.random.shuffle(permute)

        filenames=np.array(filenames)[permute].tolist()

    filenames=filenames[:13*8]
    dataset = Database( dataset_name,metadata,test=True)
    if(mycolors is None):
        if(showgt is False):
            colors = [colorlist[int(i)] for i in np.arange(number_of_clusters)]
        else:
            colors = [colorlist[int(i)] for i in np.arange(80)]
    else:
        colors = [mycolors[int(i)] for i in np.arange(number_of_clusters)]
    for i in range(len(filenames)):

        ax = plt.subplot(gs1[i])
        plt.axis('off')

        if(showgt is False):
            pointstoshow = keypoints[filenames[i]]['prediction']
        else:
            pointstoshow = keypoints[filenames[i]]['groundtruth']

        image ,_= dataset.Datasource.getimage_FAN( filenames[i],is_it_test_sample=keypoints[filenames[i]]['is_it_test_sample'])
        ax.imshow(image)
        if(pointstoshow.shape[1]==2):
            pointstoshow=np.append(pointstoshow,np.arange(len(pointstoshow)).reshape(-1,1),axis=1)
        ax.scatter(pointstoshow[:, 0], pointstoshow[:, 1], s=pointsize, c=np.array(colors)[pointstoshow[:, 2].astype(int)].tolist(), marker='P',edgecolors='black', linewidths=0.3)
    fig.show()

    filename=get_logs_path(experiment_name,log_path) / imagefile_name
    fig.savefig(filename)
    plt.close("all")



def ShowTrainExamples(keypoints,log_path,experiment_name,dataset_name,metadata,imagefile_name):

    fig = plt.figure(figsize=(34,55))
    gs1 = gridspec.GridSpec(13, 8)
    gs1.update(wspace=0.0, hspace=0.0)

    filenames=[k for k in keypoints.keys()]
    filenames.sort()


    filenames=filenames[:13*8]
    dataset = Database( dataset_name,metadata,test=False)

    colors = [colorlist[int(i)] for i in np.arange(80)]

    for i in range(len(filenames)):

        ax = plt.subplot(gs1[i])
        plt.axis('off')

        pointstoshow = keypoints[filenames[i]]
        image ,_= dataset.Datasource.getimage_FAN( filenames[i],is_it_test_sample=False)
        ax.imshow(image)
        if(pointstoshow.shape[1]==2):
            pointstoshow=np.append(pointstoshow,np.arange(len(pointstoshow)).reshape(-1,1),axis=1)
        ax.scatter(4*pointstoshow[:, 0], 4*pointstoshow[:, 1], s=400, c=np.array(colors)[pointstoshow[:, 2].astype(int)].tolist(), marker='P',edgecolors='black', linewidths=0.3)
    fig.show()

    filename=get_logs_path(experiment_name,log_path) / imagefile_name
    fig.savefig(filename)
    plt.close("all")




