from torch.utils.data import Dataset
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import numpy as np
from utils import *
import imgaug.augmenters as iaa
import cv2
import random
import itertools
from DataSource.CelebA import CelebA


class Database(Dataset):
    def __init__(self,
                    dataset_name,
                    metadata,
                    test=False,
                    image_keypoints=None, 
                    function_for_dataloading=None,
                    useflip=False,
                    flipppingCorrespondance=None,
                    number_of_channels=30):
        
        
        self.image_keypoints = image_keypoints
        self.number_of_channels=number_of_channels
        self.test=test

        self.dataset_name=dataset_name
        self.metadata=metadata
        self.function_for_dataloading = function_for_dataloading
        self.useflip=useflip
        self.flipppingCorrespondance=flipppingCorrespondance

        if(dataset_name =='CelebA'):
            self.Datasource=CelebA(self.test,self.metadata)
        
        if (self.image_keypoints is not None):
            self.files = list(self.image_keypoints.keys())
        else:
            if(test):
                self.files,self.is_test_sample=self.Datasource.init()
            else:
                self.files=self.Datasource.init()

        self.heatmapsize=64
        self.heatmapsize_scale1=80
        self.heatmapsize_scale2=100
        self.scale1=self.heatmapsize/self.heatmapsize_scale1
        self.scale2=self.heatmapsize/self.heatmapsize_scale2

        self.ScaleDistill1 = iaa.Affine(scale={"x": self.scale1, "y": self.scale1},mode='edge')
        self.ScaleDistill2 = iaa.Affine(scale={"x": self.scale2, "y": self.scale2},mode='edge')

        self.cutout=cutout(80, 0.5, True)

        # try:
        #     self.files=self.files[:2000]  
        # except:
        #     pass

        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.function_for_dataloading(self,idx)


    def update_keypoints(self,keypoints):
        self.image_keypoints=keypoints
        self.files = list(self.image_keypoints.keys())

    def get_image_superpoint(self,idx):
        name = self.files[idx]

        image =self.Datasource.getimage_superpoint(name)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = torch.from_numpy(np.expand_dims(image_gray, 0) / 255.0).float()


        
        bbox=self.Datasource.getbox(name)
            


        bbox = torch.tensor(bbox)
        sample={'image_gray': image_gray, 'filename': name,'bounding_box':bbox}

            
        return sample



    def get_FAN_inference(self,idx):
    
        name = self.files[idx]
        image ,_=self.Datasource.getimage_FAN(name)
        image =torch.from_numpy(image / 255.0).permute(2, 0, 1).float()

        sample = {'image': image, 'filename': name}
        return sample




    def get_FAN_evaluation(self,idx):     
        name = self.files[idx]
        is_it_test_sample=bool(self.is_test_sample[idx])
        image ,size=self.Datasource.getimage_FAN(name,is_it_test_sample=is_it_test_sample)
        image =torch.from_numpy(image / 255.0).permute(2, 0, 1).float()
        
        groundtruth=torch.from_numpy(self.Datasource.getGroundtruth(name,is_it_test_sample)).float()
        sample = {'image': image, 'filename': name,'groundtruth':groundtruth,'is_it_test_sample':is_it_test_sample,'originalsize':size}
        return sample



    def get_FAN_secondStage_train(self, idx):        

        name = self.files[idx]
        keypoints = self.image_keypoints[name].copy()
        keypoints[:,:2]=keypoints[:,:2]*4
        keypoints=keypoints.round()

        image,keypoints =self.Datasource.getimage_FAN(name,self.Datasource.augmentations, keypoints)

        if(self.useflip and random.random() <0.5 and self.flipppingCorrespondance is not None): 
                image=np.fliplr(image)
                keypoints=keypoints.copy()
                keypoints[:,0]=(image.shape[0]-1) - keypoints[:,0]
                keypoints[:,2]=self.flipppingCorrespondance[keypoints[:,2].astype(int)]

        image=self.cutout(image)



        keypoints[:,:2]=keypoints[:,:2]/4
        keypoints=keypoints.round()

        image = torch.from_numpy(image / 255.0).permute(2, 0, 1).float()
        
        heatmaps_with_keypoints = torch.zeros(self.number_of_channels)
        
        indeces = torch.from_numpy(keypoints[:, 2]).int().tolist()
        heatmaps_with_keypoints[indeces] = 1
        heatmaps_with_keypoints=heatmaps_with_keypoints==1
        heatmaps = BuildMultiChannelGaussians(self.number_of_channels, keypoints.round())

        sample = {'image': image, 'heatmaps': heatmaps, 'heatmaps_with_keypoints': heatmaps_with_keypoints}
        return sample



    def get_FAN_firstStage_train(self, idx):     
        
        name1 = self.files[idx]
        keypoints1 = self.image_keypoints[name1].copy()

        keypoints1[:,:2]=keypoints1[:,:2]*4
        image1 ,keypoints1=self.Datasource.getimage_FAN(name1,self.Datasource.augmentations,keypoints1)


        if(self.useflip and random.random() <0.5 and keypoints1.shape[1]==4): 
                image1=np.fliplr(image1)
                keypoints1=keypoints1.copy()
                keypoints1[:,0]=(image1.shape[0]-1) - keypoints1[:,0]
                keypoints1[:,2]=keypoints1[:,3]

        keypoints1[:,:2]=keypoints1[:,:2]/4
        keypoints1=keypoints1.round()

        

        if (random.random() <0.5 and sum(keypoints1[:,2]==-1)==0):     
            idx2 = random.randint(0, len(self.files) - 1)
            name2 = self.files[idx2]
            keypoints2 = self.image_keypoints[name2].copy()        
            
            keypoints2[:,:2]=keypoints2[:,:2]*4
            image2 ,keypoints2=self.Datasource.getimage_FAN(name2,self.Datasource.augmentations,keypoints2)
    

        else:
  
            name2 = name1
            keypoints2 = self.image_keypoints[name2].copy()

            keypoints2[:,:2]=keypoints2[:,:2]*4
            image2 ,keypoints2=self.Datasource.getimage_FAN(name2,self.Datasource.augmentations,keypoints2)


        if(self.useflip and random.random() <0.5 and keypoints2.shape[1]==4): 
                image2=np.fliplr(image2)
                keypoints2=keypoints2.copy()
                keypoints2[:,0]=(image2.shape[0]-1) - keypoints2[:,0]
                keypoints2[:,2]=keypoints2[:,3]


        keypoints2[:,:2]=keypoints2[:,:2]/4
        keypoints2=keypoints2.round()



        image1 = torch.from_numpy(image1 / 255.0).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2 / 255.0).permute(2, 0, 1).float()

        image = torch.cat((image1, image2))


        gaussian1 = BuildGaussians(keypoints1)
        gaussian2 = BuildGaussians(keypoints2)

        gaussian=torch.cat((gaussian1.unsqueeze(0),gaussian2.unsqueeze(0)))

        if(sum(keypoints1[:,2]==-1)>0):
            keypoints1[:,2]=np.arange(len(keypoints1))
            keypoints2[:,2]=np.arange(len(keypoints2))

        number_of_pairs=5000
        pairs_positives = -1*np.ones((number_of_pairs, 4))
        pair_index = 0

        pairs_negatives1 = -1*np.ones((number_of_pairs, 4))
        pairs_negatives2 = -1*np.ones((number_of_pairs, 4))


        # positive pairs
        for i in range(len(keypoints1)):
            if(keypoints1[i, 2]==-1):continue
            indxes = keypoints2[:, 2] == keypoints1[i, 2]
            coord1 = keypoints1[i, :2]
            coord2 = keypoints2[indxes, :2]

            if (len(coord2) == 0): continue

            coord2 = coord2[0]
            # check that not of the coordinates are out of range cause of the augmentations
            if (sum(coord1 > self.heatmapsize-1) == 0 and sum(coord1 < 0) == 0) and (sum(coord2 > self.heatmapsize-1) == 0 and sum(coord2 < 0) == 0):
                if (pair_index >= number_of_pairs - 1): break
                pairs_positives[pair_index, :2] = coord1
                pairs_positives[pair_index, 2:4] = coord2
                pair_index += 1


        # negative pairs
        keypoints1=keypoints1[np.logical_and(keypoints1[:,0]<self.heatmapsize-1, keypoints1[:,1]<self.heatmapsize-1)]
        keypoints1=keypoints1[np.logical_and(keypoints1[:,0]>0, keypoints1[:,1]>0)]
        keypoints2=keypoints2[np.logical_and(keypoints2[:,0]<self.heatmapsize-1, keypoints2[:,1]<self.heatmapsize-1)]
        keypoints2=keypoints2[np.logical_and(keypoints2[:,0]>0, keypoints2[:,1]>0)]

        if(len(keypoints1)>2):
            combinationskeypoints1=np.array( list( itertools.combinations(np.arange(len(keypoints1)),2) ) )
            pairs_negatives1[:len(combinationskeypoints1), :2] = keypoints1[combinationskeypoints1[:,0],:2]
            pairs_negatives1[:len(combinationskeypoints1), 2:4] = keypoints1[combinationskeypoints1[:,1],:2]


            randomNegatives=number_of_pairs-len(combinationskeypoints1)
            negatives=np.concatenate((np.random.randint(low=1,high=self.heatmapsize, size=(randomNegatives,1)),np.random.randint(low=1,high=self.heatmapsize, size=(randomNegatives,1))),axis=1)
            indexes_for_keypoints=np.random.randint(low=0,high=len(keypoints1),size=randomNegatives)
            pairs_negatives1[len(combinationskeypoints1):, :2] = keypoints1[indexes_for_keypoints,:2]
            pairs_negatives1[len(combinationskeypoints1):, 2:4] = negatives


            pairDistance = np.linalg.norm(pairs_negatives1[:,:2]-pairs_negatives1[:,2:4],axis=1)
            #5.7 corresponds to nms of size 1 on the 64x64 dimension
            pairs_negatives1[pairDistance<5.7]=-1
        
        if(len(keypoints2)>2):
            combinationskeypoints2=np.array( list( itertools.combinations(np.arange(len(keypoints2)),2) ) )
            pairs_negatives2[:len(combinationskeypoints2), :2] = keypoints2[combinationskeypoints2[:,0],:2]
            pairs_negatives2[:len(combinationskeypoints2), 2:4] = keypoints2[combinationskeypoints2[:,1],:2]

            
            randomNegatives=number_of_pairs-len(combinationskeypoints2)
            negatives=np.concatenate((np.random.randint(low=1,high=self.heatmapsize, size=(randomNegatives,1)),np.random.randint(low=1,high=self.heatmapsize, size=(randomNegatives,1))),axis=1)
            indexes_for_keypoints=np.random.randint(low=0,high=len(keypoints2),size=randomNegatives)
            pairs_negatives2[len(combinationskeypoints2):, :2] = keypoints2[indexes_for_keypoints,:2]
            pairs_negatives2[len(combinationskeypoints2):, 2:4] = negatives


            pairDistance = np.linalg.norm(pairs_negatives2[:,:2]-pairs_negatives2[:,2:4],axis=1)
            pairs_negatives2[pairDistance<5.7]=-1

        pairs_positives=torch.from_numpy(pairs_positives)
        pairs_negatives1=torch.from_numpy(pairs_negatives1)
        pairs_negatives2=torch.from_numpy(pairs_negatives2)


        sample = {'image': image, 'positive_pairs': pairs_positives,'pairs_negatives1': pairs_negatives1, 'pairs_negatives2': pairs_negatives2, 'keypointHeatmaps': gaussian}
        
        return sample




