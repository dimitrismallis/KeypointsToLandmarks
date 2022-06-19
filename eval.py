from Evaluators.evalCelebA import EvaluatorCelebA
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from Database import Database
from FanClass import FAN_Model
import imgaug.augmenters as iaa
import resource
import yaml
from utils import *
from configuration import Configuration
from Visualise import ShowTestExamples
import Evaluators.evalCelebA as evalCelebA



def test_stage1(FAN,config,log_path,metadata):



    evaluation_database = Database(config.dataset_name, 
                                   metadata, 
                                   test=True, 
                                   function_for_dataloading=Database.get_FAN_evaluation,
                                   number_of_channels=config.K)

    evaluation_dataloader = DataLoader(evaluation_database, batch_size=10, shuffle=False,num_workers=10, drop_last=False)
    
    keypoints=FAN.Get_labels_for_evaluation_firstStage(evaluation_dataloader)

    ShowTestExamples(keypoints,log_path,config.experiment_name,config.K,config.dataset_name,metadata,imagefile_name=f'Test_Stage1.jpg')

    evaluator=evalCelebA.EvaluatorCelebA(config.experiment_name,log_path)
    evaluator.Evaluate_Stage1(keypoints,config.K,evaluation_dataloader)




def test_stage2(FAN,config,log_path,metadata):



    evaluation_database = Database(config.dataset_name, 
                                   metadata, 
                                   test=True, 
                                   function_for_dataloading=Database.get_FAN_evaluation,
                                   number_of_channels=config.K)

    evaluation_dataloader = DataLoader(evaluation_database, batch_size=10, shuffle=False,num_workers=10, drop_last=False)


    ShowTestExamples(keypoints,log_path,config.experiment_name,config.K,config.dataset_name,metadata,imagefile_name=f'Test_Stage2.jpg')

    evaluator=evalCelebA.EvaluatorCelebA(config.experiment_name,log_path)
    evaluator.Evaluate_Stage2(keypoints,config.K,evaluation_dataloader)




if __name__=="__main__":
    torch.manual_seed(1993)
    torch.cuda.manual_seed_all(1993)
    np.random.seed(1993)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    config=Configuration().params

    with open('paths.yml') as file:
        paths = yaml.load(file, Loader=yaml.FullLoader)
    check_paths(paths)
    log_path=paths['log_path']
    path_to_superpoint_checkpoint=paths['path_to_superpoint_checkpoint']
    metadata=paths['metadata']


    stage=1
    initialize_log_dirs(config.experiment_name,log_path)

    path_to_checkpoint=config.path_to_checkpoint

    

    FAN = FAN_Model(None, 
                config.experiment_name,
                config.confidence_thres_FAN, 
                log_path,
                stage)



    if(stage == 1):

        FAN.init_firststage( config.lr,
                        config.weight_decay,
                        config.M,
                        config.bootstrapping_iterations,
                        config.iterations_per_round,
                        config.K,
                        config.nms_thres_FAN,
                        config.lr_step_schedual_stage1)

        if (path_to_checkpoint is None):
            path_to_checkpoint=GetPathsResumeFirstStage(config.experiment_name,log_path)

        FAN.load_trained_fiststage_model(path_to_checkpoint)

        keypoints=test_stage1(FAN,config,log_path,metadata)
    else:

        FAN.init_secondstage(config.lr,
                        config.weight_decay,
                        config.K,
                        config.lr_step_schedual_stage2,
                        config.roundIterations,
                        )
        if (path_to_checkpoint is None):
            path_to_checkpoint=GetPathsEval(config.experiment_name,log_path)

        FAN.load_trained_secondstage_model(path_to_checkpoint)

        keypoints=test_stage2(FAN,config,log_path,metadata)
    




