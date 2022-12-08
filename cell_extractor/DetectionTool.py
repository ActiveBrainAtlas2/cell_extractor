import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from numpy import *
from pylab import plot
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