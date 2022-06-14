import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import random
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
import copy
import math
from utils import *
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from matrix_completion import svt_solve, calc_unobserved_rmse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

class EvaluatorCelebA():
    def __init__(self,experiment_name,log_path):
        self.experiment_name=experiment_name
        self.log_path=log_path


        def compute_iod(y):
            if y.shape[0] == 5:
                iod = np.sqrt( (y[0,0] - y[1,0])**2 + (y[0,1] - y[1,1])**2 )
            else:
                iod = np.sqrt( (y[41,0] - y[46,0])**2 + (y[41,1] - y[46,1])**2 )
            return iod
        
        self.compute_iod=compute_iod



    def Evaluate_Stage1(self,data,K,dataloader):

        model_predictions,groundtruth=self.ProcessEvalData(data)
        keypoints_array,groundtruth_array_68,groundtruth_array_5,is_test_sample=self.ProcessKeypoints(model_predictions,groundtruth,dataloader)

        fwd,_=fwd_withNan_calculation(keypoints_array,groundtruth_array_5,is_test_sample,reg_factor=0.001,npts=K,nimages=[19000],nrepeats=2,size=218,compute_iod=self.compute_iod)

        log_text(f'Forward NME: {fwd[0]}', self.experiment_name, self.log_path)


    def Evaluate_Stage2(self,data,K,dataloader):
        

        model_predictions,groundtruth=self.ProcessEvalData(data)
        keypoints_array,groundtruth_array_68,groundtruth_array_5,is_test_sample=self.ProcessKeypoints(model_predictions,groundtruth,dataloader)


        fwd5,_=fwd_calculation(keypoints_array,groundtruth_array_5,is_test_sample,reg_factor=0.001,npts=K,nimages=[19000],nrepeats=10,size=218,compute_iod=self.compute_iod)

        _,fwd_perlandmark_cumulative=fwd_calculation(keypoints_array,groundtruth_array_68,is_test_sample,reg_factor=0.001,npts=K,nimages=[300],nrepeats=10,size=218,compute_iod=self.compute_iod)
        _,bwd_perlandmark_cumulative=bwd_calculation(keypoints_array,groundtruth_array_68,is_test_sample,reg_factor=0.1,npts=K,nimages=[300],nrepeats=10,size=218,compute_iod=self.compute_iod)
        

        log_text(f'Forward NME: {fwd5[0]}', self.experiment_name, self.log_path)


        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.ticker as mtick
        matplotlib.use('Agg')
        fig = plt.figure(figsize=(6, 7))
        ax = fig.gca()
        ax.set_facecolor('#F8F8F8')
        plt.title(r"$\bf{MAFL}$, $\it{Forward}$", fontsize=24)
        plt.xlim(1, 67)
        plt.ylim(2.5, 9)
        ax.tick_params(labelsize=14)
        xrange = np.linspace(1, 68, 10).astype(int)
        yrange = np.arange(2.5, 10, 1)
        ax.set_xticks(xrange)
        ax.set_yticks(yrange)
        ax.tick_params(labelsize=15)
        plt.grid()
        plt.plot(np.arange(1, len(fwd_perlandmark_cumulative) + 1), 100 * fwd_perlandmark_cumulative, c='#b30000', label="Our Method $\it{(K=30)}$", linewidth=8)
        plt.ylabel('NME (%)', fontsize=20, fontstyle='italic')
        plt.xlabel('# of groundtruth landmarks', fontsize=20, fontstyle='italic')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.show()
        filename=get_logs_path(self.experiment_name, self.log_path) / 'ForwardNME.jpg'
        fig.savefig(filename,bbox_inches='tight')

        

        import matplotlib.ticker as mtick
        fig = plt.figure(figsize=(6, 7))
        ax = fig.gca()
        ax.set_facecolor('#F8F8F8')
        plt.title(r"$\bf{MAFL}$, $\it{Backward}$", fontsize=24)
        plt.xlim(1, 30)
        plt.ylim(3, 13)
        ax.tick_params(labelsize=14)
        xrange = np.linspace(1, 30, 10).astype(int)
        yrange = np.arange(2,15, 2)
        ax.set_xticks(xrange)
        ax.set_yticks(yrange)
        plt.grid()
        plt.plot(np.arange(1, len(bwd_perlandmark_cumulative) + 1), 100 * bwd_perlandmark_cumulative, c='#b30000', label="Our Method $\it{(K=30)}$", linewidth=8)
        plt.ylabel('NME (%)', fontsize=20, fontstyle='italic',labelpad=-7)
        plt.xlabel('# unsupervised landmarks', fontsize=20, fontstyle='italic')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.show()
        filename=get_logs_path(self.experiment_name, self.log_path) / 'BackwardNME.jpg'
        fig.savefig(filename,bbox_inches='tight')

   

    def ProcessEvalData(self,data):
        data=copy.deepcopy(data)

        predictions={}
        groundtruth={}
        
        for sample in data.keys():
            predictions[sample]=data[sample]['prediction']
            groundtruth[sample]={'groundtruth':data[sample]['groundtruth'],'is_it_test_sample':data[sample]['is_it_test_sample']}

        return predictions,groundtruth


    def ProcessKeypoints(self,predictions, groundtruth,dataloader):
        keypoints=copy.deepcopy(predictions)
        groundtruth=copy.deepcopy(groundtruth)
        Samples=[f for f in keypoints.keys() ]

        number_of_detected_keypoints = len(predictions[Samples[0]])

        keypoints_array = np.zeros((len(Samples), 2 * number_of_detected_keypoints))
        groundtruth_array_68 = np.zeros((len(Samples), 2 * 68))
        groundtruth_array_5 = np.zeros((len(Samples), 2 * 5))
        is_test_sample=np.zeros(len(Samples))

        for i in range(len(Samples)):
            sample_points=predictions[Samples[i]]
            is_test_sample[i]=groundtruth[Samples[i]]['is_it_test_sample']
            keypoints_array[i]=dataloader.dataset.Datasource.keypointsToOriginalResolution(Samples[i],sample_points,is_it_test_sample=is_test_sample[i].item()).reshape(-1)
            sample_gt = groundtruth[Samples[i]]['groundtruth']
            groundtruth_array_68[i]=dataloader.dataset.Datasource.keypointsToOriginalResolution(Samples[i],sample_gt[:-5],is_it_test_sample=is_test_sample[i].item()).reshape(-1)
            groundtruth_array_5[i]=dataloader.dataset.Datasource.keypointsToOriginalResolution(Samples[i],sample_gt[-5:],is_it_test_sample=is_test_sample[i].item()).reshape(-1)
        
        return keypoints_array,groundtruth_array_68,groundtruth_array_5,is_test_sample



