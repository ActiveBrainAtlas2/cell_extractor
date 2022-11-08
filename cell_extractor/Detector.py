from attr import has
from cell_extractor.Predictor import GreedyPredictor
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
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
