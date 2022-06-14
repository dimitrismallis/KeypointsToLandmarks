from utils import  *
from model import FAN
import torch.nn as nn
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np
import clustering
import faiss
from scipy.spatial.distance import pdist,squareform,cdist
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
import random
import numpy.ma as ma
import sklearn.cluster
import sklearn.utils.extmath
import torch.nn.parallel
from model import ConvBlock
import torch.nn.functional as F

class FAN_Model():
    def __init__(self,criterion,experiment_name,confidence_thres_FAN,log_path,stage):
        
        self.model = FAN(stage)
        

        self.criterion=criterion
        self.log_path=log_path
        self.experiment_name=experiment_name
        self.log_path=log_path
        self.confidence_thres_FAN=confidence_thres_FAN


    def init_firststage(self,lr,weight_decay,M,bootstrapping_iterations,iterations_per_round,K,nms_thres_FAN,lr_step_schedual_stage1):

        log_text(f"Training model initiated", self.experiment_name, self.log_path)
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_step_schedual_stage1=lr_step_schedual_stage1
        self.nms_thres_FAN=nms_thres_FAN
        self.bootstrapping_iterations=bootstrapping_iterations
        self.M=M
        self.iterations_per_round=iterations_per_round
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr,  weight_decay=self.weight_decay)
        self.schedualer = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        self.K=K
        self.centroid= None
        self.margin = 0.8
        self.eps = 1e-9
        self.iterations=0
        self.model = torch.nn.DataParallel(self.model).cuda()


    def init_secondstage(self,lr,weight_decay,K,lr_step_schedual_stage2,roundIterations,checkpoint_filename=None,flipppingCorrespondance=None):
        self.iterations = 0
        self.weight_decay=weight_decay
        self.lr = lr
        self.lr_step_schedual_stage2=lr_step_schedual_stage2
        self.roundIterations=roundIterations
        self.flipppingCorrespondance=flipppingCorrespondance
        if(checkpoint_filename is not None):
            log_text(f"Pretrained First Stage model loaded from  : {checkpoint_filename}", self.experiment_name,self.log_path)

            checkpoint = torch.load(checkpoint_filename, map_location='cpu')
            try:
                self.model.load_state_dict(checkpoint['state_dict'])
            except:
                # create new OrderedDict that does contain `module.`
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)


        self.K=K
        self.model._modules['l1'] = nn.Conv2d(256, self.K, kernel_size=1, stride=1, padding=0)

        basemodel=nn.ModuleList()
        detector=nn.ModuleList()
        for child in self.model._modules:
            if(child in ['l1','bn_end1','conv_last1','top_m_1']):
                detector.append(self.model._modules[child])
            else:
                basemodel.append(self.model._modules[child])


        self.optimizer = torch.optim.RMSprop([{"params":basemodel.parameters(),'lr':self.lr/10},
                                              {"params":detector.parameters(), 'lr':self.lr }],weight_decay=self.weight_decay)

        self.model = torch.nn.DataParallel(self.model).cuda()


        self.schedualer = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)


    def load_trained_secondstage_model(self,checkpoint_filename):

                
        log_text(f"Pretrained Second Stage model loaded from  : {checkpoint_filename}", self.experiment_name,self.log_path)

        try:
            checkpoint = torch.load(checkpoint_filename, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
        except:
            # create new OrderedDict that does contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = 'module.'+k # remove `module.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
            # raise Exception(f'Loading weights for FAN from {checkpoint_filename} failed.')

        self.iterations = checkpoint['iteration']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.schedualer.load_state_dict(checkpoint['schedualer'])
        self.flipppingCorrespondance = checkpoint['flipppingCorrespondance']



    def load_trained_fiststage_model(self,checkpoint_filename):
        log_text(f"Pretrained First Stage model loaded from  : {checkpoint_filename}", self.experiment_name,self.log_path)

        try:
            checkpoint = torch.load(checkpoint_filename, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
        except:
            # create new OrderedDict that does contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = 'module.'+k # remove `module.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
            # raise Exception(f'Loading weights for FAN from {checkpoint_filename} failed.')

        self.iterations = checkpoint['iteration']
        self.centroid= checkpoint['centroid']

        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.schedualer =  torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)




    def Train_stage1(self, dataloader):

        log_text(f"Training Begins", self.experiment_name, self.log_path)
        self.model.train()

        log_text('Current LR ' + str(self.optimizer.param_groups[0]['lr']),self.experiment_name,self.log_path)
        while(True):
            for i_batch, sample in enumerate(dataloader):

                self.optimizer.zero_grad()

                #reduce the learning rate a few times during training
                if (self.iterations in self.lr_step_schedual_stage1):
                    self.schedualer.step()
                    log_text('LR ' + str(self.optimizer.param_groups[0]['lr']),self.experiment_name,self.log_path)

                #save model weights during warm - start
                if (self.iterations >0 and self.iterations % 15000 == 0 and self.iterations < self.bootstrapping_iterations):
                    log_text(f"Iterations : {self.iterations}", self.experiment_name, self.log_path)
                    self.save_stage1() 
                
                # end of warm start
                if( self.iterations == self.bootstrapping_iterations):
                    log_text(f"Warm Start Completed", self.experiment_name, self.log_path)
                    self.iterations+=1
                    self.save_stage1()
                    return

                # Training round completed
                if (self.iterations > self.bootstrapping_iterations and self.iterations % self.iterations_per_round == 0):
                    log_text(f"Iterations : {self.iterations}", self.experiment_name, self.log_path)
                    self.iterations+=1
                    self.save_stage1()    
                    return


                input = my_cuda(sample['image'])
                descriptorpairs = my_cuda(sample['positive_pairs'])
                keypointHeatmaps = (my_cuda(sample['keypointHeatmaps']))
                descriptorpairs_negatives1 = my_cuda(sample['pairs_negatives1'])
                descriptorpairs_negatives2 = my_cuda(sample['pairs_negatives2'])

                bsize=input.size(0)
                number_of_pairs=descriptorpairs.size(1)


                batchid = my_cuda(
                    torch.arange(bsize)
                        .repeat(number_of_pairs)
                        .reshape(number_of_pairs,bsize)
                        .transpose(1, 0))


                output1_detector, output1_descriptor = self.model(input[:, 0:3, :, :])
                output2_detector, output2_descriptor = self.model(input[:, 3:, :, :])

                loss_detector1 = self.criterion(output1_detector, keypointHeatmaps[:, 0:1, :, :])
                loss_detector2 = self.criterion(output2_detector, keypointHeatmaps[:, 1:2, :, :])

                output1features_positives = output1_descriptor[
                                  batchid.reshape(-1).long(),
                                  :,
                                  descriptorpairs[:, :, 1].reshape(-1).long(),
                                  descriptorpairs[:, :, 0].reshape(-1).long()]
                output1features_positives=output1features_positives[descriptorpairs[:, :, 0].reshape(-1) != -1]

                output2features_positives  = output2_descriptor[
                                  batchid.reshape(-1).long(),
                                  :,
                                  descriptorpairs[:, :, 3].reshape(-1).long(),
                                  descriptorpairs[:, :, 2].reshape(-1).long()]
                output2features_positives=output2features_positives[descriptorpairs[:, :, 0].reshape(-1) != -1]



                output1features_negatives1 = output1_descriptor[
                                  batchid.reshape(-1).long(),
                                  :,
                                  descriptorpairs_negatives1[:, :, 1].reshape(-1).long(),
                                  descriptorpairs_negatives1[:, :, 0].reshape(-1).long()]
                output1features_negatives1=output1features_negatives1[descriptorpairs_negatives1[:, :, 0].reshape(-1) != -1]

                output2features_negatives1 = output1_descriptor[
                                  batchid.reshape(-1).long(),
                                  :,
                                  descriptorpairs_negatives1[:, :, 3].reshape(-1).long(),
                                  descriptorpairs_negatives1[:, :, 2].reshape(-1).long()]
                output2features_negatives1=output2features_negatives1[descriptorpairs_negatives1[:, :, 0].reshape(-1) != -1]



                output1features_negatives2 = output2_descriptor[
                                  batchid.reshape(-1).long(),
                                  :,
                                  descriptorpairs_negatives2[:, :, 1].reshape(-1).long(),
                                  descriptorpairs_negatives2[:, :, 0].reshape(-1).long()]
                output1features_negatives2=output1features_negatives2[descriptorpairs_negatives2[:, :, 0].reshape(-1) != -1]

                output2features_negatives2 = output2_descriptor[
                                  batchid.reshape(-1).long(),
                                  :,
                                  descriptorpairs_negatives2[:, :, 3].reshape(-1).long(),
                                  descriptorpairs_negatives2[:, :, 2].reshape(-1).long()]
                output2features_negatives2=output2features_negatives2[descriptorpairs_negatives2[:, :, 0].reshape(-1) != -1]



                distances_positives = (output1features_positives - output2features_positives).pow(2).sum(1)

                distances_negatives1 = (output1features_negatives1 - output2features_negatives1).pow(2).sum(1)

                distances_negatives2 = (output1features_negatives2 - output2features_negatives2).pow(2).sum(1)


                descriptor_losses = (distances_positives.sum()
                        +torch.nn.functional.relu(self.margin - (distances_negatives1 + self.eps).sqrt()).pow(2).sum()
                        +torch.nn.functional.relu(self.margin - (distances_negatives2 + self.eps).sqrt()).pow(2).sum())


                descriptor_losses = descriptor_losses/(len(distances_positives)+len(distances_negatives1)+len(distances_negatives2))

                loss = 10 * descriptor_losses + loss_detector1 + loss_detector2
                loss.backward()
                self.optimizer.step()
                self.iterations+=1




    def Train_stage2(self,dataloader):

        self.model.train()
        count = 0
        log_text(f"Training Begins", self.experiment_name,self.log_path)
        while(True):
            for i_batch, sample in enumerate(dataloader):
                
                if (self.iterations>0  and self.iterations in self.lr_step_schedual_stage2):
                    self.schedualer.step()
                    log_text('LR ' + str(self.optimizer.param_groups[0]['lr']),self.experiment_name,self.log_path)


                if (self.iterations>0  and self.iterations%self.roundIterations==0):
                    log_text(f"Round Completed, iteration {self.iterations}", self.experiment_name, self.log_path)
                    self.save_stage2()
                    return


                self.optimizer.zero_grad()
                
                input = my_cuda(sample['image'])
                heatmaps = my_cuda(sample['heatmaps'])

                heatmaps_with_keypoints = my_cuda(sample['heatmaps_with_keypoints'])

                predictions = self.model(input) 
                
                loss =self.criterion(predictions, heatmaps,heatmaps_with_keypoints)

                loss.backward()
                self.optimizer.step()
                self.iterations += 1
                
                if(self.iterations%50==0):
                    log_text(f"Stats, Iteration:{self.iterations} Loss:{loss.item()}, max:{torch.max(predictions)}", self.experiment_name, self.log_path)


    def Update_pseudoLabels(self,dataloader):

        log_text(f"Clustering stage for iteration {self.iterations}", self.experiment_name, self.log_path)
        self.model.eval()

        imagesize=256
        heatmapsize=64
        numberoffeatures = 256
        buffersize = 500000
        # allocation of 2 buffers for temporal storing of keypoints and descriptors.
        Keypoint_buffer = torch.zeros(buffersize,3)
        Descriptor__buffer = torch.zeros(buffersize, numberoffeatures)
        Descriptor__buffer_flipped = torch.zeros(buffersize, numberoffeatures)

        # arrays on which we save buffer content periodically. Corresponding files are temporal and
        # will be deleted after the completion of the process
        CreateFileArray(str(get_checkpoints_path(self.experiment_name,self.log_path) / 'keypoints'), 3)
        CreateFileArray(str(get_checkpoints_path(self.experiment_name,self.log_path) / 'descriptors'), numberoffeatures)
        CreateFileArray(str(get_checkpoints_path(self.experiment_name,self.log_path) / 'descriptors_flipped'), numberoffeatures)

        # intermediate variables
        first_index = 0
        last_index = 0
        buffer_first_index = 0
        buffer_last_index = 0
        keypoint_indexes = {}

        pointsperimage=0
        log_text(f"Inference of keypoints and descriptors begins", self.experiment_name, self.log_path)


        for i_batch, sample in enumerate(dataloader):

  
            input = my_cuda(sample['image'])
            names = sample['filename']

            with torch.no_grad():
                output = self.model.forward(input)
            outputHeatmap = output[0]
            descriptors_volume = output[1]

            input_flipped = input.flip(3)
            with torch.no_grad():
                output_flipped = self.model.forward(input_flipped)
            descriptors_volume_flipped = output_flipped[1]


            batch_keypoints = GetBatchMultipleHeatmap(outputHeatmap, self.confidence_thres_FAN,self.nms_thres_FAN)

            for i in range(input.size(0)):

                indexes = batch_keypoints[:, 0] == i

                sample_keypoints = batch_keypoints[indexes, 1:][:,:3]
                

                for n in range(len(sample_keypoints)):
                    px = int(math.floor(sample_keypoints[n][0] + 0.5))
                    py = int(math.floor(sample_keypoints[n][1] + 0.5))
                    if (1 < px < heatmapsize-1 and 1 < py < heatmapsize-1):
                        diff = torch.tensor([outputHeatmap[i,0][py][px+1] - outputHeatmap[i,0][py][px-1],outputHeatmap[i,0][py+1][px]-outputHeatmap[i,0][py-1][px]])
                        sample_keypoints[n][:2] += torch.sign(diff).cuda() * .25

                pointsperimage+=len(sample_keypoints)

                sample_keypoints_flipped=sample_keypoints.clone()
                sample_keypoints_flipped[:,0]=(heatmapsize-1) - sample_keypoints_flipped[:,0]


                descriptors = GetDescriptors(descriptors_volume[i], sample_keypoints[:, :2],
                                             heatmapsize,
                                             heatmapsize)

                descriptors_flipped = GetDescriptors(descriptors_volume_flipped[i], sample_keypoints_flipped[:, :2],
                                heatmapsize,
                                heatmapsize)





                numofpoints = sample_keypoints.shape[0]
                last_index += numofpoints
                buffer_last_index += numofpoints

                Keypoint_buffer[buffer_first_index: buffer_last_index, :2] = sample_keypoints.detach().cpu()[:,:2]
                Descriptor__buffer[buffer_first_index: buffer_last_index, :] = descriptors.detach().cpu()
                Descriptor__buffer_flipped[buffer_first_index: buffer_last_index, :] = descriptors_flipped.detach().cpu()

                keypoint_indexes[names[i]] = [first_index, last_index]
                first_index += numofpoints
                buffer_first_index += numofpoints

              
            # periodically we store the buffer in file
            if buffer_last_index > int(buffersize * 0.8):
                AppendFileArray(np.array(Keypoint_buffer[:buffer_last_index]),
                                str(get_checkpoints_path(self.experiment_name,self.log_path) / 'keypoints'))
                AppendFileArray(np.array(Descriptor__buffer[:buffer_last_index]),
                                str(get_checkpoints_path(self.experiment_name,self.log_path) / 'descriptors'))
                AppendFileArray(np.array(Descriptor__buffer_flipped[:buffer_last_index]),
                                str(get_checkpoints_path(self.experiment_name,self.log_path) / 'descriptors_flipped'))

                Keypoint_buffer = torch.zeros(buffersize, 3)
                Descriptor__buffer = torch.zeros(buffersize, numberoffeatures)
                Descriptor__buffer_flipped = torch.zeros(buffersize, numberoffeatures)
                buffer_first_index = 0
                buffer_last_index = 0

        # store any keypoints left on the buffers
        AppendFileArray(np.array(Keypoint_buffer[:buffer_last_index]),str(get_checkpoints_path(self.experiment_name,self.log_path) / 'keypoints'))
        AppendFileArray(np.array(Descriptor__buffer[:buffer_last_index]),str(get_checkpoints_path(self.experiment_name,self.log_path) / 'descriptors'))
        AppendFileArray(np.array(Descriptor__buffer_flipped[:buffer_last_index]),str(get_checkpoints_path(self.experiment_name,self.log_path) / 'descriptors_flipped'))


        torch.cuda.empty_cache()
        
        # load handlers to the Keypoints and Descriptor files
        Descriptors, fileHandler1 = OpenreadFileArray(str(get_checkpoints_path(self.experiment_name,self.log_path) / 'descriptors'))
        Descriptors_flipped, fileHandler3 = OpenreadFileArray(str(get_checkpoints_path(self.experiment_name,self.log_path) / 'descriptors_flipped'))
        Keypoints, fileHandler2 = OpenreadFileArray(str(get_checkpoints_path(self.experiment_name,self.log_path) / 'keypoints'))
        Keypoints = Keypoints[:, :]
        log_text(f"Keypoints Detected per image Only detector {pointsperimage / len(keypoint_indexes)}", self.experiment_name,self.log_path)
        log_text(f"Inference of keypoints and descriptors completed", self.experiment_name, self.log_path)
        log_text(f"Keypoints Detected per image {len(Keypoints)/len(keypoint_indexes)}", self.experiment_name, self.log_path)

        
        Image_Keypoints, Image_Keypoints_inference,centroid , averagepointsperimage, averagepointsperimage_inference,flipppingCorrespondance_inference = self.RecoverCorrespondance(Keypoints,Descriptors,Descriptors_flipped,keypoint_indexes)
        self.centroid=centroid


        log_text(f"Keypoints Detected per image(inference) {averagepointsperimage_inference}", self.experiment_name, self.log_path)
        log_text(f"Keypoints Detected per image {averagepointsperimage}", self.experiment_name, self.log_path)


        ClosereadFileArray(fileHandler1, str(get_checkpoints_path(self.experiment_name,self.log_path) / 'keypoints'))
        ClosereadFileArray(fileHandler2, str(get_checkpoints_path(self.experiment_name,self.log_path) / 'descriptors'))
        ClosereadFileArray(fileHandler3, str(get_checkpoints_path(self.experiment_name,self.log_path) / 'descriptors_flipped'))

        log_text(f"Clustering stage completed", self.experiment_name, self.log_path)
        return Image_Keypoints , Image_Keypoints_inference,flipppingCorrespondance_inference

    

    def RecoverCorrespondance(self,Keypoints,Descriptors,Descriptors_flipped,keypoint_indexes):
        # we use a subset of all the descriptors for clustering based on the recomendation of the Faiss repository
        numberOfPointsForClustering = 800000
        
        descriptors = clustering.preprocess_features(Descriptors[:numberOfPointsForClustering])
        descriptors_flipped = clustering.preprocess_features(Descriptors_flipped[:numberOfPointsForClustering])
        clusteringDescriptors=np.concatenate((descriptors[:numberOfPointsForClustering],descriptors_flipped[:numberOfPointsForClustering]),axis=0)

        KthCluster_num_of_elements=[]
        for k in range(self.K,self.K+2):
            # precompute squared norms of data points
            x_squared_norms = sklearn.utils.extmath.row_norms(clusteringDescriptors[:50000], squared=True)
            centroids,_=sklearn.cluster.kmeans_plusplus(clusteringDescriptors[:50000],k,x_squared_norms=x_squared_norms)

            # we use a subset of all the descriptors for clustering based on the recomendation of the Faiss repository
            centroids=np.array(centroids)


            KmeansClustering=clustering.Kmeans(k)
            I,centroids,distanceToCentroid=KmeansClustering.cluster(clusteringDescriptors[:50000], centroids=centroids,verbose=False)

            KthCluster_num_of_elements.append((k,np.sort(np.unique(I,return_counts=True)[1])[-self.K]))


        k=max(KthCluster_num_of_elements,key=lambda item:item[1])[0]


        x_squared_norms = sklearn.utils.extmath.row_norms(clusteringDescriptors, squared=True)
        centroids,_=sklearn.cluster.kmeans_plusplus(clusteringDescriptors,k,x_squared_norms=x_squared_norms)

        # we use a subset of all the descriptors for clustering based on the recomendation of the Faiss repository
        centroids=np.array(centroids)

        KmeansClustering=clustering.Kmeans(k)
        I,centroids,_=KmeansClustering.cluster(clusteringDescriptors, centroids=centroids,verbose=False)


        log_text(f"Points per cluster {np.sort(np.unique(I,return_counts=True)[1])}", self.experiment_name, self.log_path)
        log_text(f"Actual Cluster {len(np.unique(I,return_counts=True)[1])}", self.experiment_name, self.log_path)

        counts=np.unique(I,return_counts=True)
        bigger_clusters_indeces=np.argsort(counts[1])[-self.K:]
        recover_indeces=-1*np.ones(k)
        recover_indeces[bigger_clusters_indeces]=np.arange(self.K)
        centroids=centroids[bigger_clusters_indeces]

        flipppingCorrespondance_inference=np.zeros((self.K,self.K))

        Image_Keypoints_inference={}
        averagepointsperimage_inference=0
        Image_keypointsToKeep={}
        PointsToKeep=np.zeros(len(Descriptors))==1
        for image in keypoint_indexes:
            start, end = keypoint_indexes[image]
            detectorkeypoints = Keypoints[start:end, :]
            image_descriptors=clustering.preprocess_features(Descriptors[start:end, :])
            keypoint_distanceToCentroid,clustering_assingments=KmeansClustering.index.search(image_descriptors,1)
            

            image_descriptors_flipped=clustering.preprocess_features(Descriptors_flipped[start:end, :])
            _,clustering_assingments_flipped=KmeansClustering.index.search(image_descriptors_flipped,1)


            tokeep_indexes=np.arange(len(detectorkeypoints))
            image_keypointsToKeep=np.zeros(len(detectorkeypoints))

            x_ind=np.in1d(clustering_assingments[:,0],bigger_clusters_indeces)
            detectorkeypoints=detectorkeypoints[x_ind]
            clustering_assingments=clustering_assingments[x_ind]
            clustering_assingments_flipped=clustering_assingments_flipped[x_ind]
            keypoint_distanceToCentroid=keypoint_distanceToCentroid[x_ind]
            tokeep_indexes=tokeep_indexes[x_ind]

            x_ind=np.in1d(clustering_assingments_flipped[:,0],bigger_clusters_indeces)
            detectorkeypoints=detectorkeypoints[x_ind]
            clustering_assingments=clustering_assingments[x_ind]
            clustering_assingments_flipped=clustering_assingments_flipped[x_ind]
            keypoint_distanceToCentroid=keypoint_distanceToCentroid[x_ind]
            tokeep_indexes=tokeep_indexes[x_ind]

            clustering_assingments=recover_indeces[clustering_assingments].astype(int)
            clustering_assingments_flipped=recover_indeces[clustering_assingments_flipped].astype(int)
            

            flipppingCorrespondance_inference[clustering_assingments[:,0],clustering_assingments_flipped[:,0]]+=1

            keypoints=np.zeros((len(detectorkeypoints),4))
            keypoints[:,:2]=detectorkeypoints[:,:2]
            keypoints[:,2]=clustering_assingments[:,0]
            keypoints[:,3]=clustering_assingments_flipped[:,0]


            sort_indexes=np.argsort(keypoint_distanceToCentroid.reshape(-1))
            keypoints=keypoints[sort_indexes]
            tokeep_indexes=tokeep_indexes[sort_indexes]

            _,closestPointIndex=np.unique(keypoints[:,2],return_index=True)

            newkeypoints=keypoints[closestPointIndex]
            tokeep_indexes=tokeep_indexes[closestPointIndex]

            image_keypointsToKeep[tokeep_indexes]=1
            image_keypointsToKeep=image_keypointsToKeep==1


            Image_Keypoints_inference[image]= newkeypoints
            Image_keypointsToKeep[image]=image_keypointsToKeep
            PointsToKeep[start:end]=image_keypointsToKeep
            averagepointsperimage_inference+=len(newkeypoints)



        averagepointsperimage_inference=averagepointsperimage_inference/len(keypoint_indexes)
        
        flipppingCorrespondance_inference=np.argmax(flipppingCorrespondance_inference,axis=1)

        KmeansClustering=clustering.Kmeans(self.M)

        clusteringDescriptors=np.concatenate((descriptors[PointsToKeep[:numberOfPointsForClustering]],descriptors_flipped[PointsToKeep[:numberOfPointsForClustering]]),axis=0)
        I,self.traincentroids,_=KmeansClustering.cluster(clusteringDescriptors,verbose=False)

        Image_Keypoints={}
        averagepointsperimage=0

        PointsToKeep=np.zeros(len(Descriptors))
        for image in keypoint_indexes:
            start, end = keypoint_indexes[image]

            image_keypointsToKeep=Image_keypointsToKeep[image]
            detectorkeypoints = Keypoints[start:end, :][image_keypointsToKeep]

            if(len(detectorkeypoints)<2):
                continue

            image_descriptors=clustering.preprocess_features(Descriptors[start:end, :])[image_keypointsToKeep]
            keypoint_distanceToCentroid,clustering_assingments=KmeansClustering.index.search(image_descriptors,1)


            image_descriptors_flipped=clustering.preprocess_features(Descriptors_flipped[start:end, :])[image_keypointsToKeep]
            _,clustering_assingments_flipped=KmeansClustering.index.search(image_descriptors_flipped,1)

            keypoints=np.zeros((len(detectorkeypoints),4))
            keypoints[:,:2]=detectorkeypoints[:,:2]
            keypoints[:,2]=clustering_assingments[:,0]
            keypoints[:,3]=clustering_assingments_flipped[:,0]



            sort_indexes=np.argsort(keypoint_distanceToCentroid.reshape(-1))
            keypoints=keypoints[sort_indexes]

            _,closestPointIndex=np.unique(keypoints[:,2],return_index=True)

            newkeypoints=keypoints[closestPointIndex]

            Image_Keypoints[image]= newkeypoints

            averagepointsperimage+=len(newkeypoints)

        averagepointsperimage=averagepointsperimage/len(Image_Keypoints)
        

        return Image_Keypoints ,Image_Keypoints_inference,centroids ,averagepointsperimage ,averagepointsperimage_inference,flipppingCorrespondance_inference


    def Get_labels_for_evaluation_firstStage(self,dataloader):
        log_text('Predictions for evaluation FAN',self.experiment_name,self.log_path)
        self.model.eval()
        
        heatmapsize=64
        keypoints={}


        for i_batch, sample in enumerate(dataloader):

            input = my_cuda(sample['image'])
            bsize = input.size(0)
            name = sample['filename']
            groundtruth=sample['groundtruth']
            is_test_sample=sample['is_it_test_sample']

            with torch.no_grad():
                output = self.model.forward(input)
            outputHeatmap = output[0].detach()
            descriptors_volume = output[1].detach()

            batch_keypoints = GetBatchMultipleHeatmap(outputHeatmap, self.confidence_thres_FAN,self.nms_thres_FAN)


            for i in range(input.size(0)):

                indexes = batch_keypoints[:, 0] == i
                sample_keypoints = batch_keypoints[indexes, 1:][:,:3]

                samplegroundtruth=groundtruth[i].detach().cpu().numpy()
                descriptors = GetDescriptors(descriptors_volume[i], sample_keypoints[:, :2],heatmapsize,heatmapsize)
                descriptors = clustering.preprocess_features(descriptors.cpu().detach().numpy())

                sample_keypoints=sample_keypoints.detach().cpu().numpy()
                keypoint_distanceToCentroid=cdist(self.centroid,descriptors)
                clustering_assingments=np.argmin(keypoint_distanceToCentroid,axis=0)
                keypoint_distanceToCentroid=np.min(keypoint_distanceToCentroid,axis=0)
                

                sample_keypoints[:,2]=clustering_assingments

                sort_indexes=np.argsort(keypoint_distanceToCentroid.reshape(-1))
                sample_keypoints=sample_keypoints[sort_indexes]


                _,closestPointIndex=np.unique(sample_keypoints[:,2],return_index=True)

                sample_keypoints=sample_keypoints[closestPointIndex]
                sample_keypoints[:,:2]=4*sample_keypoints[:,:2]


                sampleKeypoints=np.empty((self.K,2,))
                sampleKeypoints[:] = np.nan
                sampleKeypoints[sample_keypoints[:,2].astype(int)]=sample_keypoints[:,:2]


                keypoints[name[i]]={'prediction':sampleKeypoints,'groundtruth':samplegroundtruth,'is_it_test_sample':is_test_sample[i]}

        return keypoints





    def Get_labels_for_evaluation_test(self,dataloader ,useflip=True):
        log_text('Predictions for evaluation FAN',self.experiment_name,self.log_path)
        self.model.eval()
        
        
        keypoints={}
        for i_batch, sample in enumerate(dataloader):

            input = my_cuda(sample['image'])
            bsize = input.size(0)
            name = sample['filename']
            groundtruth=sample['groundtruth']
            is_test_sample=sample['is_it_test_sample']
            originalsize=sample['originalsize']
            with torch.no_grad():
                output = self.model(input)

            if(useflip):
                input_flipped = input.flip(3)
                output_flipped = self.model(input_flipped)
                output_flipped = output_flipped.flip(3)
                output_flipped=output_flipped[:,self.flipppingCorrespondance,:,:]
                output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5


            for i in range(bsize):
                sampleKeypoints=GetPointsFromHeatmaps(output[i])[:,:3].detach().cpu().numpy()
                sampleKeypoints=sampleKeypoints[:,:2]
                samplegroundtruth=groundtruth[i].detach().cpu().numpy()


                keypoints[name[i]]={'prediction':sampleKeypoints,'groundtruth':samplegroundtruth,'is_it_test_sample':is_test_sample[i].item()}

        return keypoints





    def save_stage1(self):

        checkPointDirectory = get_checkpoints_path(self.experiment_name,self.log_path)
        checkPointFileName=f'{self.experiment_name}FirstStageIteration{self.iterations}' + '.pth'

        checkPointFileName = checkPointDirectory / checkPointFileName
        save_parameters = {
        'state_dict': self.model.state_dict(),
        'optimizer':  self.optimizer.state_dict(),
        'iteration':  self.iterations,
        'centroid': self.centroid
         }
        torch.save(save_parameters, checkPointFileName)


    def save_stage2(self):

        checkPointDirectory = get_checkpoints_path(self.experiment_name,self.log_path)
        checkPointFileName=f'{self.experiment_name}SecondStageIter{self.iterations}' + '.pth'

        checkPointFileName = checkPointDirectory / checkPointFileName

        save_parameters = {
        'state_dict': self.model.state_dict(),
        'optimizer':  self.optimizer.state_dict(),
        'iteration':  self.iterations,
        'schedualer':self.schedualer.state_dict(),
        'flipppingCorrespondance':self.flipppingCorrespondance
         }
        torch.save(save_parameters, checkPointFileName)



    def save_keypoints(self,Image_Keypoints,filename):
        checkPointDirectory = get_checkpoints_path(self.experiment_name,self.log_path)
        checkPointFileName = checkPointDirectory / filename

        with open(checkPointFileName, 'wb') as handle:
            pickle.dump(Image_Keypoints, handle, protocol=pickle.HIGHEST_PROTOCOL)


