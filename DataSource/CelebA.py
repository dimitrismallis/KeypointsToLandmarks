from torch.utils.data import Dataset
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import numpy as np
from utils import *
import imgaug.augmenters as iaa
import imgaug.augmentables.kps 
import cv2
import random
import yaml
import torchfile
import scipy.io
from pathlib import Path
import itertools
from scipy.spatial import distance

class CelebA():
    def __init__(self,test,metadata):
        self.metadata=metadata
        self.test=test
        self.H=218
        self.W=178



        with open('paths.yml') as file:
            paths = yaml.load(file, Loader=yaml.FullLoader)
    
        self.datapath = paths['CelebA_datapath']

        assert self.datapath!=None, "Path missing!! Update 'CelebA_datapath' on paths/main.yaml with path to CelebA images."
        assert Path(self.datapath).exists(), f'Specified path to CelebA images does not exists {self.datapath}'

        with open(Path(self.metadata,'CelebA/list_eval_partition.txt'), 'r') as f:
            CelebAImages = f.read().splitlines()
        assert len(list(Path(self.datapath).glob('*.jpg')))==len(CelebAImages), f"There are missing CelebA images from {self.datapath}. Please specify a path that includes all CelebA images"
        
        self.boxes=load_keypoints(Path(self.metadata,'CelebA/CelebABoundingBoxes.pickle'))
        self.groundtruth=load_keypoints(Path(self.metadata,'CelebA/MaflGroundtruthLandmarks.pickle'))


        #load 5 point gt
        with open(Path(self.metadata,'CelebA/list_landmarks_align_celeba.txt'), 'r') as f:
            gt_5points = f.read().splitlines()
        gt_5points=gt_5points[2:]
        names = [l.split()[0] for l in gt_5points]
        coords = [np.array(l.split()[1::]).reshape(5,2).astype(int) for l in gt_5points]
        self.gt_5points = dict(zip(names,coords))


        #augmentations for first stage
        self.augmentations = iaa.Sequential([
        iaa.Sometimes(0.3,
                    iaa.GaussianBlur(sigma=(0, 0.5))
                    ),
        iaa.ContrastNormalization((0.85, 1.3)),
        iaa.Sometimes(0.5,
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
            )
        ,
        iaa.Multiply((0.9, 1.1), per_channel=0.2),
        iaa.Sometimes(0.3,
                    iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
                    ),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-40, 40),
        ),
        
    ])



    def init(self):           
        if (self.test):

            with open(Path(self.metadata,'CelebA/mafl_testing.txt'), 'r') as f:
                TestImages = f.read().splitlines()
            with open(Path(self.metadata,'CelebA/mafl_training.txt'), 'r') as f:
                TrainImages = f.read().splitlines()
            
            
            self.files = TrainImages + TestImages
            self.is_test_sample = np.ones(len(self.files))
            self.is_test_sample[:len(TrainImages)]=0
            return self.files, self.is_test_sample

            

        else:

            with open(Path(self.metadata,'CelebA/list_eval_partition.txt'), 'r') as f:
                CelebAImages = f.read().splitlines()

            CelebATrainImages=[f[:-2] for f in CelebAImages if f[-1]=='0']
            with open(Path(self.metadata,'CelebA/mafl_testing.txt'), 'r') as f:
                MaflTestImages = f.read().splitlines()


            CelebATrainImages=list(set(CelebATrainImages)-set(MaflTestImages))
            self.files = CelebATrainImages

            return self.files


    def getGroundtruth(self,imagefile,is_test_sample):


            
        groundtruthpoints_68=self.groundtruth[imagefile].copy()
        groundtruthpoints_5=self.gt_5points[imagefile].copy()

        groundtruthpoints= np.concatenate((groundtruthpoints_68, groundtruthpoints_5),axis=0)

        groundtruthpoints=self.keypointsToFANResolution(imagefile,groundtruthpoints,self.W,self.H)
        return groundtruthpoints
    

    def getFANBox(self,imagefile,W,H,is_it_test_sample=False):
        bbox = self.getbox(imagefile)
        delta_x=1.2*bbox[2]-bbox[0]
        delta_y=2*bbox[3]-bbox[1]
        delta=0.25*(delta_x+delta_y)

        if(delta<20): 
            tight_aux=8
        else: tight_aux=int(8*delta/100)

        minx=int(max(bbox[0]-tight_aux,0))
        miny=int(max(bbox[1]-tight_aux,0))
        maxx=int(min(bbox[2]+tight_aux,W-1))
        maxy=int(min(bbox[3]+tight_aux,H-1))

        return minx,miny,maxx,maxy

    def keypointsToFANResolution(self,imagefile,keypoints,W=None,H=None,is_test_sample=False):
                
        if(W is None or H is None):
            W=self.W
            H=self.H
        minx,miny,maxx,maxy=self.getFANBox(imagefile,W,H)

        keypoints[:,0]=keypoints[:,0]-minx
        keypoints[:,1]=keypoints[:,1]-miny
        keypoints[:,0]=keypoints[:,0]*(256/(maxx-minx))
        keypoints[:,1]=keypoints[:,1]*(256/(maxy-miny))
        return keypoints


    def keypointsToOriginalResolution(self,imagefile,keypoints,is_it_test_sample=False):

        minx,miny,maxx,maxy=self.getFANBox(imagefile,self.W,self.H,is_it_test_sample)
        
        keypoints[:,0]=keypoints[:,0]*((maxx-minx)/256)
        keypoints[:,1]=keypoints[:,1]*((maxy-miny)/256)
        keypoints[:,0]=keypoints[:,0]+minx
        keypoints[:,1]=keypoints[:,1]+miny
        return keypoints

    def getbox(self,imagefile):
        bbox = self.boxes[imagefile].copy()
        return bbox
            

    def getimage_superpoint(self,imagefile):
        image = cv2.cvtColor(cv2.imread(str(Path(self.datapath,imagefile))), cv2.COLOR_BGR2RGB)
        return image


    def getimage_FAN(self,imagefile, augmentations=None, keypoints=None,is_it_test_sample=False):
        image = cv2.cvtColor(cv2.imread(str(Path(self.datapath,imagefile))), cv2.COLOR_BGR2RGB)
        
        W=image.shape[1]
        H=image.shape[0]
        if(augmentations is not None):
            keypoints_originalres=self.keypointsToOriginalResolution(imagefile,keypoints)
            imgaug_keypoints = []
            for i in range(len(keypoints)):
                imgaug_keypoints.append(Keypoint(x=keypoints_originalres[i, 0], y=keypoints_originalres[i, 1]))
            kpsoi = KeypointsOnImage(imgaug_keypoints, shape=image.shape)
            image, keypoitns_aug = augmentations(image=image, keypoints=kpsoi)

            keypoints_originalres = np.column_stack((keypoitns_aug.to_xy_array(), keypoints_originalres[:, 2:]))


        minx,miny,maxx,maxy=self.getFANBox(imagefile,image.shape[1],image.shape[0])

        image=image[miny:maxy,minx:maxx,:]
        scaledImage=cv2.resize(image,dsize=(256,256))

        if(keypoints is not None):
            augmentedkeypoints=self.keypointsToFANResolution(imagefile,keypoints_originalres,self.W,self.H)


            return scaledImage,augmentedkeypoints
        
        return scaledImage,np.array([W,H])


