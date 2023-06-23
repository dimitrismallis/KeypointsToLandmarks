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
from eval import test_stage2 as evalModel


def train(config):
    stage=2

    with open('paths.yml') as file:
        paths = yaml.load(file, Loader=yaml.FullLoader)
    check_paths(paths)
    log_path=paths['log_path']
    metadata=paths['metadata']

    #This funcion will create the directories /Logs and a /CheckPoints at log_path
    initialize_log_dirs(config.experiment_name,log_path)

    log_text(f"Experiment Name {config.experiment_name}\n"
        f"Database {config.dataset_name}\n"
        "Training Parameters: \n"
        f"Number of discovered landmarks K:  {config.K} \n"
        f"Batch size {config.batchSize} \n"
        f"Learning  rate {config.lr} \n"
        f"Weight Decay {config.weight_decay} \n"
        , config.experiment_name, log_path)


    log_text("Training of First Stage begins", config.experiment_name,log_path)
    
    criterion = JointsMSELoss(True).cuda()

    FAN = FAN_Model(criterion, 
                    config.experiment_name,
                    config.confidence_thres_FAN, 
                    log_path,
                    1)

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


    path_to_checkpoint=GetPathsResumeFirstStage(config.experiment_name,log_path)

    FAN.load_trained_fiststage_model(path_to_checkpoint)

    _,keypoints,flipppingCorrespondance=FAN.Update_pseudoLabels(cluster_dataloader)

   
    FAN = FAN_Model(criterion, 
                config.experiment_name,
                config.confidence_thres_FAN, 
                log_path,
                stage)

    FAN.init_secondstage(config.lr,
                        config.weight_decay,
                        config.K,
                        config.lr_step_schedual_stage2,
                        config.save_checkpoint_frequency,
                        path_to_checkpoint,
                        flipppingCorrespondance)

    train_dataset = Database( config.dataset_name, 
                              metadata,
                              image_keypoints=keypoints,
                              function_for_dataloading=Database.get_FAN_secondStage_train,
                              useflip=config.useflip,
                              flipppingCorrespondance=flipppingCorrespondance,
                              number_of_channels=config.K )


    train_dataloader = DataLoader(train_dataset, batch_size=config.batchSize, shuffle=True, num_workers=config.num_workers,drop_last=True)

    log_text(f'Dataset Number of Images:{len(train_dataset.files)}', config.experiment_name, log_path) 

    while FAN.iterations < config.total_iterations_stage2:

        log_text(f'Training for iteration {FAN.iterations} begins', config.experiment_name, log_path) 
        FAN.Train_stage2(train_dataloader)
        
    
        FAN.iterations+=1




if __name__=="__main__":
    torch.manual_seed(1993)
    torch.cuda.manual_seed_all(1993)
    np.random.seed(1993)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    config=Configuration().params
    train(config)
