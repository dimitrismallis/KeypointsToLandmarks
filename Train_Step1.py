import torch
import numpy as np
import resource
from configuration import Configuration
import yaml
from utils import *
from SuperPoint import SuperPoint
from Database import Database
from torch.utils.data import Dataset, DataLoader
from FanClass import FAN_Model
from Visualise import ShowTrainExamples
from eval import test_stage1 as evalModel

def train(config):
    stage=1

    with open('paths.yml') as file:
        paths = yaml.load(file, Loader=yaml.FullLoader)
    check_paths(paths)
    log_path=paths['log_path']
    path_to_superpoint_checkpoint=paths['path_to_superpoint_checkpoint']
    metadata=paths['metadata']

    #This funcion will create the directories /Logs and a /CheckPoints at log_path
    initialize_log_dirs(config.experiment_name,log_path)

    log_text(
        f"Experiment Name: {config.experiment_name}\n"
        f"Database: {config.dataset_name}\n"
        "Training Parameters: \n"
        f"Number of discovered landmarks K: {config.K} \n"
        f"Number of clusters M: {config.M} \n"
        f"Batch size: {config.batchSize} \n"
        f"Learning rate: {config.lr} \n"
        f"Weight Decay: {config.weight_decay} \n"
        f"Flipping during Training: {config.useflip} \n"
        f"Detection threshold FAN: {config.confidence_thres_FAN} \n"
        f"Detection threshold SuperPoint: {config.confidence_thres_superpoint} \n"
        f"NMS threshold FAN: {config.nms_thres_FAN} pixels \n"
        f"NMS threshold SuperPoint: {config.nms_thres_superpoint} pixels \n"
        f"Number of Bootstrapping training iterations: {config.bootstrapping_iterations} \n"
        f"Total number of training iterations: {config.total_iterations_stage1} \n"
        f"Apply Clustering every {config.iterations_per_round} iterations \n"
        , config.experiment_name, log_path)


    log_text("Training of First Stage begins", config.experiment_name,log_path)
    
    criterion = nn.MSELoss().cuda()

    FAN = FAN_Model(criterion, 
                    config.experiment_name,
                    config.confidence_thres_FAN, 
                    log_path,
                    stage)


    FAN.init_firststage( config.lr,
                        config.weight_decay,
                        config.M,
                        config.bootstrapping_iterations,
                        config.iterations_per_round,
                        config.K,
                        config.nms_thres_FAN,
                        config.lr_step_schedual_stage1)

    cluster_dataset = Database( config.dataset_name, 
                                metadata,
                                function_for_dataloading=Database.get_FAN_inference )
                        
    cluster_dataloader = DataLoader(cluster_dataset, batch_size=config.batchSize, shuffle=False,num_workers=config.num_workers, drop_last=False)


    if(config.resume):
        path_to_checkpoint=config.path_to_checkpoint
        if(path_to_checkpoint is None ):
            path_to_checkpoint=GetPathsResumeFirstStage(config.experiment_name,log_path)

        FAN.load_trained_fiststage_model(path_to_checkpoint)


        keypoints,keypoints_val,_=FAN.Update_pseudoLabels(cluster_dataloader)

        ShowTrainExamples(keypoints_val,log_path,config.experiment_name,config.dataset_name,metadata,f'TrainIteration{FAN.iterations}.jpg')

    else:
        superpoint= SuperPoint( config.confidence_thres_superpoint,
                                config.nms_thres_superpoint,
                                path_to_superpoint_checkpoint,
                                config.experiment_name,
                                log_path,
                                config.remove_superpoint_outliers_percentage,                                                                                                    
                                )
        
        superpoint_dataset=Database( config.dataset_name, 
                                     metadata,
                                     function_for_dataloading=Database.get_image_superpoint)


        superpoint_dataloader = DataLoader(superpoint_dataset, 
                                batch_size=config.batchSize_superpoint, 
                                shuffle=False, 
                                num_workers=config.num_workers,
                                drop_last=True)

        keypoints=superpoint.CreateInitialPseudoGroundtruth(superpoint_dataloader)


        del superpoint
        del superpoint_dataset
        del superpoint_dataloader

    train_dataset = Database( config.dataset_name, 
                              metadata,
                              image_keypoints=keypoints,
                              function_for_dataloading=Database.get_FAN_firstStage_train,
                              useflip=config.useflip )

    train_dataloader = DataLoader(train_dataset, batch_size=config.batchSize, shuffle=True, num_workers=config.num_workers,drop_last=True)


    while (FAN.iterations<config.total_iterations_stage1):

        FAN.Train_stage1(train_dataloader)

        keypoints,keypoints_val,_=FAN.Update_pseudoLabels(cluster_dataloader)

        ShowTrainExamples(keypoints_val,log_path,config.experiment_name,config.dataset_name,metadata,f'TrainIteration{FAN.iterations}.jpg')

        train_dataloader.dataset.update_keypoints(keypoints)



if __name__=="__main__":
    torch.manual_seed(1993)
    torch.cuda.manual_seed_all(1993)
    np.random.seed(1993)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    config=Configuration().params
    train(config)
