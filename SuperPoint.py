import torch
from utils import *
import torchvision
import math
import numpy as np
import faiss
import clustering
from scipy.optimize import linear_sum_assignment
import imgaug.augmenters as iaa
import imgaug.augmentables.kps 


class SuperPoint():
    def __init__(self, 
                       confidence_thres_superpoint,
                       nms_thres_superpoint,
                       path_to_pretrained_superpoint,
                       experiment_name,
                       log_path,
                       remove_superpoint_outliers_percentage,
                       ):

        self.path_to_pretrained_superpoint=path_to_pretrained_superpoint

        self.confidence_thres_superpoint=confidence_thres_superpoint
        self.nms_thres_superpoint=nms_thres_superpoint
        self.log_path=log_path
        self.remove_superpoint_outliers_percentage=remove_superpoint_outliers_percentage
        self.experiment_name=experiment_name



        self.model = my_cuda(SuperPointNet())

        try:
            checkpoint = torch.load(path_to_pretrained_superpoint, map_location='cpu')
            self.model.load_state_dict(checkpoint)
            log_text(f"Superpoint Network from checkpoint {path_to_pretrained_superpoint}", self.experiment_name, self.log_path)
        except:
            raise Exception(f"Superpoint weights from {path_to_pretrained_superpoint} failed to load.")

        self.softmax = torch.nn.Softmax(dim=1)
        self.pixelSuffle = torch.nn.PixelShuffle(8)
        self.model.eval()




    def CreateInitialPseudoGroundtruth(self, dataloader):

        log_text(f"Extraction of initial Superpoint pseudo groundtruth", self.experiment_name,self.log_path)
        
        imagesize=256
        heatmapsize=64
        numberoffeatures=256
        buffersize=500000

        #this is used for calculating a consistent NMS theshold for datasets that include images of 
        #various sizes. For datasets with images of consistent size it does not have any effect
        referenceResolution=218

        #allocation of 2 buffers for temporal storing of keypoints and descriptors.
        Keypoint_buffer = torch.zeros(buffersize, 3)
        Descriptor__buffer = torch.zeros(buffersize, numberoffeatures)

        #arrays on which we save buffer content periodically. Corresponding files are temporal and
        #will be deleted after the completion of the process
        CreateFileArray(str(get_checkpoints_path(self.experiment_name,self.log_path) / 'keypoints'),3)
        CreateFileArray(str(get_checkpoints_path(self.experiment_name,self.log_path) / 'descriptors'), numberoffeatures)

        #intermediate variables
        first_index = 0
        last_index = 0
        buffer_first_index = 0
        buffer_last_index = 0
        keypoint_indexes = {}

        log_text(f"Inference of Keypoints begins", self.experiment_name, self.log_path)
        for i_batch, sample in enumerate(dataloader):
            input = my_cuda(sample['image_gray'])

            
            names = sample['filename']
            bsize=input.size(0)
            
            with torch.no_grad():
                detectorOutput,descriptorOutput=self.GetSuperpointOutput(input)
            detectorOutput=detectorOutput
            descriptorOutput=descriptorOutput
            for i in range(0, bsize):
            
                input_H=input.shape[2]
                NMSThres=int((input_H/referenceResolution)*self.nms_thres_superpoint)
                keypoints,_ = self.GetPointsFromHeatmap(detectorOutput[i].unsqueeze(0), self.confidence_thres_superpoint, NMSThres)
                

                #remove keypoints outside the bouding box
                bounding_box=sample['bounding_box'][i]
                pointsinbox = torch.ones(len(keypoints))
                pointsinbox[(keypoints[:, 0] < int(bounding_box[0]))] = -1
                pointsinbox[(keypoints[:, 1] < int(bounding_box[1]))] = -1
                pointsinbox[(keypoints[:, 0] > int(bounding_box[2]))] = -1
                pointsinbox[(keypoints[:, 1] > int(bounding_box[3]))] = -1
                keypoints=keypoints[pointsinbox==1]

                descriptors = GetDescriptors(descriptorOutput[i], keypoints, input.shape[3], input.shape[2])


                keypoints=keypoints.detach().cpu()

                #scale image keypoints to FAN resolution
                keypoints=dataloader.dataset.Datasource.keypointsToFANResolution(names[i],keypoints,input.shape[3], input.shape[2])


                keypoints = ((heatmapsize/imagesize)*keypoints).round()


                last_index += len(keypoints)
                buffer_last_index += len(keypoints)

                Keypoint_buffer[buffer_first_index:buffer_last_index, :2] = keypoints
                Descriptor__buffer[buffer_first_index:buffer_last_index] = descriptors.detach().cpu()

  

                keypoint_indexes[names[i]] = [first_index, last_index]
                first_index += len(keypoints)
                buffer_first_index += len(keypoints)

            #periodically we store the buffer in file
            if buffer_last_index>int(buffersize*0.8):
                AppendFileArray(np.array(Keypoint_buffer[:buffer_last_index]),str(get_checkpoints_path(self.experiment_name,self.log_path) / 'keypoints'))
                AppendFileArray(np.array(Descriptor__buffer[:buffer_last_index]), str(get_checkpoints_path(self.experiment_name,self.log_path) / 'descriptors'))

                Keypoint_buffer = torch.zeros(buffersize, 3)
                Descriptor__buffer = torch.zeros(buffersize, numberoffeatures)
                buffer_first_index = 0
                buffer_last_index = 0


        del self.model 
        torch.cuda.empty_cache()
        
        log_text(f"Inference of Keypoints completed", self.experiment_name, self.log_path)
        #store any keypoints left on the buffers
        AppendFileArray(np.array(Keypoint_buffer[:buffer_last_index]), str(get_checkpoints_path(self.experiment_name,self.log_path) / 'keypoints'))
        AppendFileArray(np.array(Descriptor__buffer[:buffer_last_index]), str(get_checkpoints_path(self.experiment_name,self.log_path) / 'descriptors'))

        #load handlers to the Keypoints and Descriptor files
        Descriptors,fileHandler1=OpenreadFileArray(str(get_checkpoints_path(self.experiment_name,self.log_path) / 'descriptors'))
        Keypoints, fileHandler2 = OpenreadFileArray( str(get_checkpoints_path(self.experiment_name,self.log_path) / 'keypoints'))
        Keypoints = Keypoints[:, :]
        log_text(f"Keypoints Detected per image {len(Keypoints)/len(keypoint_indexes)}", self.experiment_name, self.log_path)

        #perform outlier detection
        inliersindexes=np.ones(len(Keypoints))==1
        if(self.remove_superpoint_outliers_percentage>0):
            inliersindexes=self.Indexes_of_inliers(Keypoints,Descriptors,buffersize)


        Image_Keypoints={}
        avgKeypoints=0
        for k, v in keypoint_indexes.items():
            keypoints=Keypoints[v[0]:v[1]]

            inliersinimage=inliersindexes[v[0]:v[1]]
            keypoints=keypoints[inliersinimage]
            
            keypoints[:,2]=-1

            if(len(keypoints)<2):
                continue

            #remove points that lay outsize the  resized image
            pointsinImage = torch.ones(len(keypoints))
            pointsinImage[(keypoints[:, 0] < 0)] = -1
            pointsinImage[(keypoints[:, 1] < 0)] = -1
            pointsinImage[(keypoints[:, 0] > heatmapsize-1)] = -1
            pointsinImage[(keypoints[:, 1] > heatmapsize-1)] = -1
            keypoints=keypoints[pointsinImage==1]

            Image_Keypoints[k] = keypoints
            
            avgKeypoints+=len(keypoints)
        avgKeypoints=avgKeypoints/len(keypoint_indexes)

        ClosereadFileArray(fileHandler1,str(get_checkpoints_path(self.experiment_name,self.log_path) / 'descriptors'))
        ClosereadFileArray(fileHandler2,str(get_checkpoints_path(self.experiment_name,self.log_path) / 'keypoints'))

        self.save_keypoints(Image_Keypoints,"SuperPointKeypoints.pickle")
        log_text(f"Keypoints Detected per image {avgKeypoints}", self.experiment_name, self.log_path)
        log_text(f"Extraction of Initial pseudoGroundtruth completed", self.experiment_name, self.log_path)
        return Image_Keypoints



    def Indexes_of_inliers(self,Keypoints,Descriptors,buffersize):
        res = faiss.StandardGpuResources()
        nlist = 100
        quantizer = faiss.IndexFlatL2(256)
        index = faiss.IndexIVFFlat(quantizer, 256, nlist)

        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)

        gpu_index_flat.train(clustering.preprocess_features(Descriptors[:buffersize]))
        gpu_index_flat.add(clustering.preprocess_features(Descriptors[:buffersize]))

        #we process the descriptors in batches of 10000 vectors
        rg = np.linspace(0, len(Descriptors), math.ceil(len(Descriptors) / 10000) + 1, dtype=int)
        keypoints_outlier_score=np.zeros(len(Keypoints))
        for i in range(len(rg) - 1):
            descr = clustering.preprocess_features(Descriptors[rg[i]:rg[i + 1], :])
            distance_to_closest_points, _ = gpu_index_flat.search(descr, 100)
            outlierscore = np.median(distance_to_closest_points, axis=1)
            keypoints_outlier_score[rg[i]:rg[i + 1]] = outlierscore

        inliers = keypoints_outlier_score.copy()
        inliers = np.sort(inliers)

        threshold = inliers[int((1-self.remove_superpoint_outliers_percentage) * (len(inliers) - 1))]
        inliers = keypoints_outlier_score < threshold
        return inliers



    def GetPointsFromHeatmap(self,confidenceMap, threshold, NMSthes):
        mask = confidenceMap > threshold
        prob = confidenceMap[mask]
        value, indices = prob.sort(descending=True)
        pred = torch.nonzero(mask)
        prob = prob[indices]
        pred = pred[indices]
        points = pred[:, 2:4]
        points = points.flip(1)
        nmsPoints = torch.cat((points.float(), prob.unsqueeze(1)), 1).transpose(0, 1)
        thres = math.ceil(NMSthes / 2)  
        newpoints = torch.cat((nmsPoints[0:1, :] - thres, nmsPoints[1:2, :] - thres, nmsPoints[0:1, :] + thres,
                               nmsPoints[1:2, :] + thres, nmsPoints[2:3, :]), 0).transpose(0, 1)
        res = torchvision.ops.nms(newpoints[:, 0:4], newpoints[:, 4], 0.01)

        points = nmsPoints[:, res].transpose(0, 1)
        returnPoints = points[:, 0:2]
        prob = points[:, 2]
        return returnPoints,prob



    def GetSuperpointOutput(self,input):
        keypoints_volume, descriptors_volume = self.model(input)
        keypoints_volume = keypoints_volume.detach()
        keypoints_volume = self.softmax(keypoints_volume)
        volumeNoDustbin = keypoints_volume[:, :-1, :, :]
        spaceTensor = self.pixelSuffle(volumeNoDustbin)
        return spaceTensor,descriptors_volume



    def save_keypoints(self,Image_Keypoints,filename):
        checkPointdir = get_checkpoints_path(self.experiment_name,self.log_path)
        checkPointFile=checkPointdir /filename
        with open(checkPointFile, 'wb') as handle:
            pickle.dump(Image_Keypoints, handle, protocol=pickle.HIGHEST_PROTOCOL)



# ----------------------------------------------------------------------    
#  https://github.com/magicleap/SuperPointPretrainedNetwork/
#
# --------------------------------------------------------------------*/
#

class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.numberOfClasses=1
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.git c
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)
    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return semi, desc
