# ## Setting Parameters for XG Boost
# * Maximum Depth of the Tree = 3 _(maximum depth of each decision trees)_
# * Step size shrinkage used in update to prevents overfitting = 0.3 _(how to weigh trees in subsequent iterations)_
# * Maximum Number of Iterations = 1000 _(total number trees for boosting)_
# * Early Stop if score on Validation does not improve for 5 iterations
# 
# [Full description of options](https://xgboost.readthedocs.io/en/latest//parameter.html)

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from cell_extractor.CellDetectorIO import CellDetectorIO
print(xgb.__version__)
from cell_extractor.retraining.lib.logger  import logger
import pandas as pd
from cell_extractor.DetectorUsage import Detector,GreedyPredictor

class CellDetectorTrainer(Detector,CellDetectorIO):
    """class for training detectors"""
    def __init__(self,animal,round =2,segmentation_threshold=2000):
        """specifies detector to train

        :param animal: animal ID
        :type animal: str
        :param round: detection version, defaults to 2
        :type round: int, optional
        :param segmentation_threshold: threshold used for image segmentation, defaults to 2000
        :type segmentation_threshold: int, optional
        """
        CellDetectorIO.__init__(self,animal,round = round,segmentation_threshold=segmentation_threshold)
        self.last_round = CellDetectorIO(animal,round = round-1,segmentation_threshold=segmentation_threshold)
        self.init_parameter()
        self.predictor = GreedyPredictor()

    def gen_scale(self,n,reverse=False):
        s=np.arange(0,1,1/n)
        while s.shape[0] !=n:
            if s.shape[0]>n:
                s=s[:n]
            if s.shape[0]<n:
                s=np.arange(0,1,1/(n+0.1))
        if reverse:
            s=s[-1::-1]
        return s

    def get_train_and_test(self,df,frac=0.5):
        """split train and test set

        :param df: _description_
        :type df: _type_
        :param frac: _description_, defaults to 0.5
        :type frac: float, optional
        :return: _description_
        :rtype: _type_
        """
        train = pd.DataFrame(df.sample(frac=frac))
        test = df.drop(train.index,axis=0)
        print(train.shape,test.shape,train.index.shape,df.shape)
        train=self.createDM(train)
        test=self.createDM(test)
        all=self.createDM(df)
        return train,test,all

    def init_parameter(self):
        """initialize training parameter
        """
        self.default_param = {}
        shrinkage_parameter = 0.3
        self.default_param['eta'] =shrinkage_parameter
        self.default_param['objective'] = 'binary:logistic'
        self.default_param['nthread'] = 7 
        print(self.default_param)

    def train_classifier(self,features,niter,depth=None,models = None,**kwrds):
        """trains the classifier using given feature

        :param features: data frame containing detected features used for training
        :type features: _type_
        :param niter: number of iteration to train
        :type niter: _type_
        :param depth: depths of boosted trees, defaults to None
        :type depth: _type_, optional
        :param models: if given, retrain from a previous xgboost model, defaults to None
        :type models: _type_, optional
        :return: list of 30 xgboost models
        :rtype: _type_
        """
        param = self.default_param
        if depth is not None:
            param['max_depth'] = depth
        df = features
        train,test,all=self.get_train_and_test(df)
        evallist = [(train, 'train'), (test, 'eval')]
        bst_list=[]
        for i in range(30):
            train,test,all=self.get_train_and_test(df)
            if models is None:
                bst = xgb.train(param, train,niter,evallist, verbose_eval=False,**kwrds)
            else:
                bst = xgb.train(param, train,niter,evallist, verbose_eval=False,**kwrds,xgb_model=models[i])
            bst_list.append(bst)
            y_pred = bst.predict(test, iteration_range=[1,bst.best_ntree_limit], output_margin=True)
            y_test=test.get_label()
            pos_preds=y_pred[y_test==1]
            neg_preds=y_pred[y_test==0]
            pos_preds=np.sort(pos_preds)
            neg_preds=np.sort(neg_preds)
            plt.plot(pos_preds,self.gen_scale(pos_preds.shape[0]));
            plt.plot(neg_preds,self.gen_scale(neg_preds.shape[0],reverse=True))
        return bst_list
    
    def test_xgboost(self,df,depths = [1,3,5],num_round = 1000,**kwrds):
        """generate test diagnostic for a range of depth depth and a given max iterations.  Generated plot for comparing comparison through iterations for test and train dataset

        :param df: feature data frame
        :type df: _type_
        :param depths: depth of xgboost trees, defaults to [1,3,5]
        :type depths: list, optional
        :param num_round: max number of round to train, defaults to 1000
        :type num_round: int, optional
        """
        for depthi in depths:
            self.test_xgboost_at_depthi(df,depth = depthi,num_round=num_round,**kwrds)

    def test_xgboost_at_depthi(self,features,depth=1,num_round=1000,**kwrds):
        """generate diagnostics for one depths of boosted trees

        :param features: _description_
        :type features: _type_
        :param depth: _description_, defaults to 1
        :type depth: int, optional
        :param num_round: _description_, defaults to 1000
        :type num_round: int, optional
        :return: _description_
        :rtype: _type_
        """
        param = self.default_param
        param['max_depth']= depth
        train,test,_=self.get_train_and_test(features)
        evallist = [(train, 'train'), (test, 'eval')]
        _, axes = plt.subplots(1,2,figsize=(12,5))
        i=0
        for _eval in ['error','logloss']:
            Logger=logger()
            logall=Logger.get_logger()  
            param['eval_metric'] = _eval 
            bst = xgb.train(param, train, num_round, evallist, verbose_eval=False, callbacks=[logall],**kwrds)
            _=Logger.parse_log(ax=axes[i])
            i+=1
        plt.show()
        print(depth)
        return bst,Logger
    
    def save_predictions(self,features):
        """save prediction results as data frame

        :param features: _description_
        :type features: _type_
        """
        detection_df = self.load_new_features_with_coordinate()
        scores,labels,_mean,_std = self.calculate_scores(features)
        predictions=self.get_prediction(_mean,_std)
        detection_df['mean_score'],detection_df['std_score'] = _mean,_std
        detection_df['label'] = labels
        detection_df['predictions'] = predictions
        detection_df=detection_df[predictions!=-2]
        detection_df = detection_df[['animal', 'section', 'row', 'col','label', 'mean_score','std_score', 'predictions']]
        detection_df.to_csv(self.DETECTION_RESULT_DIR,index=False)
    
    def save_detector(self):
        """save the current detector"""
        detector = Detector(self.model,self.predictor)
        return super().save_detector(detector)
    
    def load_detector(self):
        """load the specified detector
        """
        detector = super().load_detector()
        self.model = detector.model
        self.predictor = detector.predictor


if __name__=='__main__':
    trainer = CellDetectorTrainer('DK55',round = 2)
    manual_df = trainer.get_combined_features_of_train_sections()
    trainer.test_xgboost(manual_df)