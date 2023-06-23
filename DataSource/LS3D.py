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

class LS3D():
    def __init__(self,test,metadata):
        self.metadata=metadata
        self.test=test
        self.H=450
        self.W=450   

        self.boxes=load_keypoints(Path(self.metadata,'LS3D/300W_LPBoundingBoxes.pickle'))


        with open('paths.yml') as file:
            paths = yaml.load(file, Loader=yaml.FullLoader)
        self.datapath = paths['300WLP_datapath']

        assert self.datapath!=None, "Path missing!! Update '300WLP_datapath' on paths/main.yaml with path to 300WLP images."
        self.path_to_LS3Dbalanced=paths['LS3Dbalanced_datapath']



        self.weight=0.25

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

            assert self.path_to_LS3Dbalanced!=None, "Path missing!! Update 'LS3Dbalanced_datapath' on paths/main.yaml with path to LS3Dbalanced images."
            self.getbox=self.getbox_fromlandmarks_ls3d_eval
            testfiles = glob.glob(self.path_to_LS3Dbalanced + '/**/*.jpg', recursive=True)
            trainfiles= list(self.boxes.keys())
            
            self.files=trainfiles[:1000]+testfiles
            self.is_test_sample = np.ones(len(self.files))
            self.is_test_sample[:1000]=0
            return self.files, self.is_test_sample
        else:
            self.files = list(self.boxes.keys())
            return self.files


    def GetFullImagePath(self,imagefile,istestsample=False):
        if(istestsample):
            return  imagefile
        return self.datapath+imagefile


    def getbox_fromlandmarks_ls3d_eval(self,imagefile,is_test_sample=False):
    
        try:
            if(is_test_sample):  
                gt=torchfile.load(imagefile[:-4]+'.t7')
            else:
                gt_filename=self.GetFullImagePath(imagefile,is_test_sample)[:-4]+'.t7'
                tempstring=gt_filename.split('/')
                tempstring.insert(-2,'landmarks')
                tempstring='/'.join(tempstring)
                gt_filename=tempstring[:-3]+'_pts.mat'
                gt=scipy.io.loadmat(gt_filename)['pts_3d']
        except:
            pass         

        
        bbox=[0,0,0,0]
        bbox[0]=int(min(gt[:,0]))
        bbox[1]=int(min(gt[:,1]))
        bbox[2]=int(max(gt[:,0]))
        bbox[3]=int(max(gt[:,1]))

        bbox[1]=bbox[1]-(bbox[3]-bbox[1])/3
        return bbox
        

    def getGroundtruth(self,imagefile,is_test_sample):
        image = cv2.cvtColor(cv2.imread(self.GetFullImagePath(imagefile,is_test_sample)), cv2.COLOR_BGR2RGB)
        if(is_test_sample):  
            groundtruthpoints=torchfile.load(imagefile[:-4]+'.t7')
        else:
            gt_filename=self.GetFullImagePath(imagefile,is_test_sample)[:-4]+'.tz'
            isflip=False
            if("_Flip" in gt_filename):
                gt_filename=gt_filename.replace('_Flip','')
                isflip=True
            tempstring=gt_filename.split('/')
            tempstring.insert(-2,'landmarks')
            tempstring='/'.join(tempstring)
            gt_filename=tempstring[:-3]+'_pts.mat'
            groundtruthpoints=scipy.io.loadmat(gt_filename)['pts_3d']
            if(isflip):
                groundtruthpoints[:,0]=self.W-groundtruthpoints[:,0]


        groundtruthpoints=self.keypointsToFANResolution(imagefile,groundtruthpoints,image.shape[1],image.shape[0],is_test_sample)
        return groundtruthpoints
    

    def getFANBox(self,imagefile,W,H,is_test_sample=False,weight=0.25):
        bbox = self.getbox(imagefile,is_test_sample)
        delta_x=1.2*bbox[2]-bbox[0]
        delta_y=2*bbox[3]-bbox[1]
        delta=weight*(delta_x+delta_y)


        if(delta<20): tight_aux=8
        else: 
            tight_aux=int(8*delta/100)
             

        minx=int(max(bbox[0]-tight_aux,0))
        miny=int(max(bbox[1]-tight_aux,0))
        maxx=int(min(bbox[2]+tight_aux,W-1))
        maxy=int(min(bbox[3]+tight_aux,H-1))

        return minx,miny,maxx,maxy


    def keypointsToFANResolution(self,imagefile,keypoints,W=None,H=None,is_test_sample=False,weight=0.25):  
        if(W is None or H is None):
            W=self.W
            H=self.H
        minx,miny,maxx,maxy=self.getFANBox(imagefile,W,H,is_test_sample,weight)

        keypoints[:,0]=keypoints[:,0]-minx
        keypoints[:,1]=keypoints[:,1]-miny
        keypoints[:,0]=keypoints[:,0]*(256/(maxx-minx))
        keypoints[:,1]=keypoints[:,1]*(256/(maxy-miny))
        return keypoints


    def keypointsToOriginalResolution(self,imagefile,keypoints,W=None,H=None,is_it_test_sample=False,weight=0.25):
        
        if H is None:
            H=self.H
        if W is None:
            W=self.W
        minx,miny,maxx,maxy=self.getFANBox(imagefile,W,H,is_it_test_sample,weight)
        
        keypoints[:,0]=keypoints[:,0]*((maxx-minx)/256)
        keypoints[:,1]=keypoints[:,1]*((maxy-miny)/256)
        keypoints[:,0]=keypoints[:,0]+minx
        keypoints[:,1]=keypoints[:,1]+miny
        return keypoints

    def getbox(self,imagefile,is_test_sample=False):
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
            keypoints_originalres=self.keypointsToOriginalResolution(imagefile,keypoints,is_it_test_sample=is_it_test_sample)

            imgaug_keypoints = []
            for i in range(len(keypoints)):
                imgaug_keypoints.append(Keypoint(x=keypoints_originalres[i, 0], y=keypoints_originalres[i, 1]))
            kpsoi = KeypointsOnImage(imgaug_keypoints, shape=image.shape)
            image, keypoitns_aug = augmentations(image=image, keypoints=kpsoi)

            keypoints_originalres = np.column_stack((keypoitns_aug.to_xy_array(), keypoints_originalres[:, 2:]))


        minx,miny,maxx,maxy=self.getFANBox(imagefile,image.shape[1],image.shape[0],is_it_test_sample,self.weight)

        image=image[miny:maxy,minx:maxx,:]
        scaledImage=cv2.resize(image,dsize=(256,256))

        if(keypoints is not None):
            augmentedkeypoints=self.keypointsToFANResolution(imagefile,keypoints_originalres,self.W,self.H)



            return scaledImage,augmentedkeypoints
        
        return scaledImage,np.array([W,H])


