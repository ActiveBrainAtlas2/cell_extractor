import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import xgboost as xgb
import pandas as pd
from cell_extractor.CellDetectorIO import CellDetectorIO
from cell_extractor.AnnotationProximityTool import AnnotationProximityTool
import os
from numpy import *
from pylab import plot

class CellDetector(CellDetectorIO):
    """class for detecting cells after example and feature extraction   

    :param CellDetectorIO: base class for file IO
    :type CellDetectorIO: _type_
    """

    def __init__(self,animal,round = 2, *args, **kwargs):
        super().__init__(animal,round=round,*args, **kwargs)
        self.detector = self.load_detector()

    def print_version(self):
        print('version of xgboost is:',xgb.__version__,'should be at least 1.5.0')

    def get_detection_results(self):
        """calculates the detection results from features of all sections

        :return: data frame containg detection result predictions==2 is sure,predictions==0 is unsure and predictions==-2 is no detection
        :rtype: _type_
        """
        features = self.get_combined_features_for_detection()
        scores,labels,_mean,_std = self.detector.calculate_scores(features)
        predictions=self.detector.get_prediction(_mean,_std)
        detection_df = self.get_combined_features()
        detection_df['mean_score'],detection_df['std_score'] = _mean,_std
        detection_df['label'] = labels
        detection_df['predictions'] = predictions
        detection_df = detection_df[['animal', 'section', 'row', 'col','label', 'mean_score','std_score', 'predictions']]
        return detection_df
    
    def calculate_and_save_detection_results(self):
        detection_df = self.get_detection_results()
        detection_df.to_csv(self.DETECTION_RESULT_DIR,index=False)

class MultiThresholdDetector(CellDetector,AnnotationProximityTool):
    """detect cell using multiple segmentation threshold (depricated)"""
    def __init__(self,animal,round,thresholds = [2000,2100,2200,2300,2700]):
        super().__init__(animal = animal,round = round)
        self.thresholds=thresholds
        self.MULTI_THRESHOLD_DETECTION_RESULT_DIR = os.path.join(self.DETECTION,f'multithreshold_detections_{self.thresholds}_round_{str(self.round)}.csv')

    def get_detections_and_scores_for_all_threshold(self):
        non_detections = []
        detections = []
        scores = []
        non_detection_scores = []
        for threshold in self.thresholds:
            print(f'loading threshold {threshold}')
            detector = CellDetector(self.animal,self.round,segmentation_threshold=threshold)
            detection = detector.load_detections()
            sure = detection[detection.predictions == 2]
            null = detection[detection.predictions == -2]
            null = null.sample(int(len(null)*0.01))
            unsure = detection[detection.predictions == 0]
            sure_score = pd.DataFrame({'mean':sure.mean_score,'std':sure.std_score,'label':sure.label})
            unsure_score = pd.DataFrame({'mean':unsure.mean_score,'std':unsure.std_score,'label':unsure.label})
            null_score = pd.DataFrame({'mean':null.mean_score,'std':null.std_score,'label':null.label})
            sure = pd.DataFrame({'x':sure.col,'y':sure.row,'section':sure.section,'name':[f'{threshold}_sure' for _ in range(len(sure))]})
            unsure = pd.DataFrame({'x':unsure.col,'y':unsure.row,'section':unsure.section,'name':[f'{threshold}_unsure' for _ in range(len(unsure))]})
            null = pd.DataFrame({'x':null.col,'y':null.row,'section':null.section,'name':[f'{threshold}_null' for _ in range(len(null))]})
            detections.append(sure)
            detections.append(unsure)
            scores.append(sure_score)
            scores.append(unsure_score)
            non_detections.append(null)
            non_detection_scores.append(null_score)
        detections = pd.concat(detections)
        non_detections = pd.concat(non_detections)
        non_detection_scores = pd.concat(non_detection_scores)
        scores = pd.concat(scores)
        return detections,scores,non_detections,non_detection_scores

    def check_cells(self,scores,check_function,determination_function):
        cell =[i for i in self.pair_categories.values() if check_function(i)]
        cell_pairs = [self.pairs[id] for id,i in self.pair_categories.items() if check_function(i)]
        final_cell_detection = []
        for id,categories in enumerate(cell):
            pair = cell_pairs[id]
            coords = self.annotations_to_compare.iloc[pair]
            score = scores.iloc[pair]
            row = determination_function(score,coords,categories)
            final_cell_detection=final_cell_detection+row
        final_cell_detection = pd.concat(final_cell_detection,axis=1).T
        return final_cell_detection

    def plot_detector_threshold(self):
        detections = self.load_detections()
        plt.figure(figsize=[30,15])
        alpha = 0.8
        size = 15
        plt.scatter(detections['mean_score'].to_numpy(),detections['std_score'].to_numpy(),color='slategrey',s=1,alpha=0.3)
        plt.axvline(-1.5)
        plt.axvline(1.5)

    def determine_pure_detection(self,scores,type_to_exclude='sure'):
        does_not_have_cell_type = lambda i: self.check(i,exclude=[f'{threshold}_{type_to_exclude}' for threshold in self.thresholds])
        def find_max_mean_score_of_the_group(score,coords,categories):
            max_id = np.argmax(score.to_numpy()[:,0])
            max_threshold = categories[max_id].split('_')[0]
            is_max_threshold = [i.split('_')[0]==max_threshold for i in categories]
            row = coords.iloc[is_max_threshold].join(score.iloc[is_max_threshold])
            return [i for _,i in row.iterrows()]
        final_cell_detection = self.check_cells(scores,check_function = does_not_have_cell_type,determination_function = find_max_mean_score_of_the_group)
        return final_cell_detection

    def determine_mixed_detection(self,scores):
        def mixed_cell_type(categories):
            has_sure = np.any([i.split('_')[1]=='sure' for i in categories])
            has_unsure = np.any([i.split('_')[1]=='unsure' for i in categories])
            return has_sure and has_unsure
        def find_max_mean_score_of_sure_detections_in_group(score,coords,categories):
            is_sure = np.array([i.split('_')[1]=='sure' for i in categories ])
            score = score[is_sure]
            coords = coords[is_sure]
            categories = np.array(categories)[is_sure]
            max_id = np.argmax(score.to_numpy()[:,0])
            max_threshold = categories[max_id].split('_')[0]
            is_max_threshold = [i.split('_')[0]==max_threshold for i in categories]
            row = coords.iloc[is_max_threshold].join(score.iloc[is_max_threshold])
            return [i for _,i in row.iterrows()]
        final_cell_detection = self.check_cells(scores,check_function = mixed_cell_type,determination_function = find_max_mean_score_of_sure_detections_in_group)
        return final_cell_detection

    def calculate_and_save_detection_results(self):
        detections,scores,non_detections,non_detection_scores = self.get_detections_and_scores_for_all_threshold()
        self.set_annotations_to_compare(detections)
        self.find_equivalent_points()
        final_unsure_detection = self.determine_pure_detection(scores,type_to_exclude = 'sure')
        final_sure_detection = self.determine_pure_detection(scores,type_to_exclude = 'unsure')
        final_mixed_detection = self.determine_mixed_detection(scores)
        self.set_annotations_to_compare(non_detections)
        self.find_equivalent_points()
        final_non_detection = self.determine_pure_detection(non_detection_scores,type_to_exclude = '')
        final_detection = pd.concat([final_sure_detection,final_unsure_detection,final_mixed_detection,final_non_detection])
        final_detection.to_csv(self.MULTI_THRESHOLD_DETECTION_RESULT_DIR,index=False)
    
    def load_detections(self):
        return pd.read_csv(self.MULTI_THRESHOLD_DETECTION_RESULT_DIR)
    
    def get_sures(self):
        detections = self.load_detections()
        return detections[[string_to_prediction(i) ==2 for i in detections.name]]
    
    def get_unsures(self):
        detections = self.load_detections()
        return detections[[string_to_prediction(i) ==0 for i in detections.name]]
    
def string_to_prediction(string):
    if string.split('_')[1] == 'sure':
        return 2
    elif string.split('_')[1] == 'unsure':
        return 0
    elif string.split('_')[1] == 'null':
        return -2
        
def detect_cell(animal,round,*args,**kwargs):
    print(f'detecting {animal}')
    detector = CellDetector(*args,animal = animal,round=round,**kwargs)
    detector.calculate_and_save_detection_results()

def detect_cell_multithreshold(animal,round,thresholds = [2000,2100,2200,2300,2700]):
    print(f'detecting {animal} multithreshold')
    detector = MultiThresholdDetector(animal,round,thresholds)
    detector.calculate_and_save_detection_results()

class Predictor:
    """class used to assign cells to sure/unsure/no detection"""
    def __init__(self,std=1.5):
        self.std = std
    
    def decision(self,mean,std):
        """decision function, 2=sure,0=unsure,-2=no detection"""
        if mean <= -self.std:
            return -2
        elif mean>-self.std and mean <= self.std:
            return 0
        elif mean >self.std:
            return 2

class BetterPredictor:
    """a simpler predictors that sets detections with a mean score with a certain mutiple of std (of all 30 models) from zero as unsures"""
    def __init__(self,std=1.5):
        self.std = std
    
    def decision(self,mean,std):
        if mean<0:
            return -2
        elif mean-std<0 and mean >0:
            return 0
        elif mean-std>0:
            return 2

class GreedyPredictor:
    """a predictor that defines a dimond region of unsures using custom boundary points"""
    def __init__(self,boundary_points=[[0,3],[3,4.5],[1,6],[-3,4],[-10,7],[10,7]]):
        self.set_boundary_points(boundary_points)

    def set_boundary_points(self,boundary_points):
        self.boundary_points=boundary_points
        self.boundary_lines=[]
        self.boundary_lines.append(self.points2line(self.boundary_points[0],self.boundary_points[1],0))
        self.boundary_lines.append(self.points2line(self.boundary_points[1],self.boundary_points[2],1))
        self.boundary_lines.append(self.points2line(self.boundary_points[2],self.boundary_points[3],2))
        self.boundary_lines.append(self.points2line(self.boundary_points[3],self.boundary_points[0],3))
        self.boundary_lines.append(self.points2line(self.boundary_points[1],self.boundary_points[5],4))
        self.boundary_lines.append(self.points2line(self.boundary_points[3],self.boundary_points[4],5))

    def print_boundary_points(self):
        print(self.boundary_points)

    def points2line(self,p1,p2,i):
        x1,y1=p1
        x2,y2=p2
        a=(y1-y2)/(x1-x2)
        b=y1-a*x1
        return a,b

    def plotline(self,a,b,i):
        X=arange(-5,5,0.01)
        Y=a*X+b
        plot(X,Y,label=str(i))
        
    def aboveline(self,p,l):
        return l[0]*p[0]+l[1] < p[1]

    def decision(self,x,y):
        p=[x,y]
        if self.aboveline(p,self.boundary_lines[0]) and not self.aboveline(p,self.boundary_lines[1])\
        and not self.aboveline(p,self.boundary_lines[2]) and self.aboveline(p,self.boundary_lines[3]):
            return 0
        if (x<0 and not self.aboveline(p,self.boundary_lines[5])) or (x>0 and self.aboveline(p,self.boundary_lines[4])):
            return -2
        if (x>0 and not self.aboveline(p,self.boundary_lines[4])) or (x<0 and self.aboveline(p,self.boundary_lines[5])):
            return 2


class Detector():
    """parent class for stroing model and training information for a given detector """
    def __init__(self,model=None,predictor:GreedyPredictor=GreedyPredictor()):
        """

        :param model: model trained, defaults to None
        :type model: _type_, optional
        :param predictor: predictor used to separate result into sure, unsure and no detection, defaults to GreedyPredictor()
        :type predictor: GreedyPredictor, optional
        """
        self.model = model
        self.predictor = predictor
        self.depth = None
        self.niter = None

    def createDM(self,df):
        """create training DM"""
        labels=df['label']
        features=df.drop('label',axis=1)
        return xgb.DMatrix(features, label=labels)
    
    def calculate_scores(self,features):
        """Calculate predicted detection scores from features

        :param features: features to detect 
        :type features: _type_
        :return: scores, original labels, mean and std of 30 detectors
        :rtype: _type_
        """
        all=self.createDM(features)
        labels=all.get_label()
        scores=np.zeros([features.shape[0],len(self.model)])
        for i in range(len(self.model)):
            bst=self.model[i]
            scores[:,i] = bst.predict(all, iteration_range=[1,bst.best_ntree_limit], output_margin=True)
        mean=np.mean(scores,axis=1)
        std=np.std(scores,axis=1)
        return scores,labels,mean,std

    def get_prediction(self,mean,std):
        """sort cell into sure/unsure/no detection"""
        predictions=[]
        for mean,std in zip(mean,std):
            p=self.predictor.decision(float(mean),float(std))
            predictions.append(p)
        return np.array(predictions)
    
    def calculate_and_set_scores(self,df):
        """calculate scores if they do not exist"""
        if not hasattr(self,'mean') or not hasattr(self,'std') or not hasattr(self,'labels'):
            _,self.labels,self.mean,self.std = self.calculate_scores(df)

    def set_plot_limits(self,lower,higher):
        """seting xlim and ylim for diagnostic plots

        :param lower: _description_
        :type lower: _type_
        :param higher: _description_
        :type higher: _type_
        """
        if lower is not None and higher is not None:
            plt.ylim([lower,higher])

    def plot_score_scatter(self,df,lower_lim = None,upper_lim = None,alpha1 = 0.5,alpha2 = 0.5,color1='teal',color2 = 'orangered',size1=3,size2=3,title = None):
        """plot the mean and std of detection scores from 30 detector for each example

        :param df: data frame with prediction result
        :type df: _type_
        :param lower_lim: x and y upper lims, defaults to None
        :type lower_lim: _type_, optional
        :param upper_lim: x and y lower lims, defaults to None
        :type upper_lim: _type_, optional
        :param alpha1: alpha for cells with manual label, defaults to 0.5
        :type alpha1: float, optional
        :param alpha2: transparancy for cells without manual label, defaults to 0.5
        :type alpha2: float, optional
        :param color1: color for cells with manual label, defaults to 'teal'
        :type color1: str, optional
        :param color2: color for cells without manual label, defaults to 'orangered'
        :type color2: str, optional
        :param size1: size for cells with manual label, defaults to 3
        :type size1: int, optional
        :param size2: size for cells without manual label, defaults to 3
        :type size2: int, optional
        :param title: title of the plot, defaults to None
        :type title: _type_, optional
        """
        self.calculate_and_set_scores(df)
        plt.figure(figsize=[15,10])
        mean_has_label = self.mean[self.labels==1]
        mean_no_label = self.mean[self.labels==0]
        std_has_label = self.std[self.labels==1]
        std_no_label = self.std[self.labels==0]
        plt.scatter(mean_no_label,std_no_label,color=color2,s=size2,alpha=alpha2)
        plt.scatter(mean_has_label,std_has_label,color=color1,s=size1,alpha=alpha1)
        plt.title('mean and std of scores for 30 classifiers')
        plt.xlabel('mean')
        plt.ylabel('std')
        plt.grid()
        if title is not None:
            plt.title(title)
        self.set_plot_limits(lower_lim,upper_lim)


    def plot_decision_scatter(self,features,lower_lim = None,upper_lim = None,title = None):
        """plot the decision of sure and unsures

        :param features: _description_
        :type features: _type_
        :param lower_lim: _description_, defaults to None
        :type lower_lim: _type_, optional
        :param upper_lim: _description_, defaults to None
        :type upper_lim: _type_, optional
        :param title: _description_, defaults to None
        :type title: _type_, optional
        """
        self.calculate_and_set_scores(features)
        if not hasattr(self,'predictions'):
            self.predictions=self.get_prediction(self.mean,self.std)
        plt.figure(figsize=[15,10])
        plt.scatter(self.mean,self.std,c=self.predictions+self.labels,s=5)
        plt.title('mean and std of scores for 30 classifiers')
        plt.xlabel('mean')
        plt.ylabel('std')
        plt.grid()
        if title is not None:
            plt.title(title)
        self.set_plot_limits(lower_lim,upper_lim)
