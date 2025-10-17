## Requirements

import warnings
from functools import partial

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.base import TransformerMixin, RegressorMixin, BaseEstimator

import lightgbm as lgb


def MirrorLog(X, c:float=(1/3)):

    '''
    X   :   The data that will be mirror-log transformed.
    c   :   Constant parameter. Default is 1/3, as suggested in the literature.
    '''

    # X = np.asarray(X)
    # X_new = np.empty_like(X)
    
    # FIX: avoids silent integer truncation if X comes in as integers
    X = np.asarray(X, dtype=float)
    X_new = np.empty_like(X, dtype=float)
    
    X_new[X != 0] = np.sign(X[X != 0]) * (np.log(np.abs(X[X != 0]) + (1/c)) + np.log(c))
    X_new[X == 0] = 0
    return X_new

def InverseMirrorLog(X, c:float=(1/3)):

    '''
    X   :   The data that will be inverse mirror-log transformed.
    c   :   Constant parameter. Default is 1/3, as suggested in the literature.
    '''

    # X = np.asarray(X)
    # X_inv = np.empty_like(X)
    
    # FIX: avoids silent integer truncation if X comes in as integers
    X = np.asarray(X, dtype=float)
    X_inv = np.empty_like(X, dtype=float)
    
    X_inv[X != 0] = np.sign(X[X != 0]) * (np.exp(np.abs(X[X != 0]) - np.log(c)) - (1/c))
    X_inv[X == 0] = 0
    return X_inv


class MirrorLogNormScaler(TransformerMixin, BaseEstimator):

    def __init__(self, mirrorlog_kwargs:dict=None, normalizer=None):

        '''
        mirrorlog_kwargs    :   Keyword arguments for the mirror-log transformation. Default is {'c': 1/3}.
        normalizer          :   Normalizer to use after mirror-log transformation. Default is MinMaxScaler.
        '''

        self.mirrorlog_kwargs = mirrorlog_kwargs
        self.normalizer = normalizer
        _normalizer = self.normalizer or MinMaxScaler()
        self._normalizer = _normalizer

        mirrorlog_kwargs_ = self.mirrorlog_kwargs or {'c': 1/3}

        self._mlog_scaler = FunctionTransformer(
            func=partial(MirrorLog, **mirrorlog_kwargs_),
            inverse_func=partial(InverseMirrorLog, **mirrorlog_kwargs_),
            check_inverse=False
        )

    def fit(self, X, y=None):

        '''
            X   :   The data that will be mirror-log transformed then used to compute the per-feature minimum and maximum used for later scaling along the features axis.
            y   :   Ignored.
        '''

        X_mlog = self._mlog_scaler.fit_transform(X)
        
        # stores the min/max in mirror-log space from the training data, making sure the inverse transform (exp) falls in the range, 
        # so inverse can be made safe
        self._mlog_min_ = np.nanmin(X_mlog, axis=0)     # NEW
        self._mlog_max_ = np.nanmax(X_mlog, axis=0)     # NEW
    
        self._normalizer.fit(X_mlog)
        return self

    def transform(self, X):

        '''
        X   :   The data that will be mirror-log transformed then min-max scaled.
        '''

        X_mlog = self._mlog_scaler.transform(X)
        X_new = self._normalizer.transform(X_mlog)
        return X_new
    
    def inverse_transform(self, X):

        '''
        X   :   The data that will be inverse min-max scaled then inverse mirror-log transformed.
        '''

        X_mlog = self._normalizer.inverse_transform(X)
        
        # predictions from the regressor can fall slightly outside the training [0,1] MinMax range 
        # without clipping, the subsequent InverseMirrorLog (which uses exp) can explode
        # if the model predicts something smaller(bigger) than self._mlog_min_(self._mlog_max_), it sets it to self._mlog_min_(self._mlog_max_)
        X_mlog = np.clip(X_mlog, self._mlog_min_, self._mlog_max_)   # NEW - prevents the explosion during inverse transformation
        
        X_inv = self._mlog_scaler.inverse_transform(X_mlog)
        return X_inv
    

class ProcessingPipeline(RegressorMixin, BaseEstimator):

    '''
    rfecv_kwargs        :   Keyword arguments for the RFECV feature selection step. Estimator for feature importance must be provided. If set to False, no RFECV is applied and input features are considered definitive. Default is a LGBM regressor with 3-fold CV and step size of 2.
    scaler              :   Scaler for feature normalization. If set to False, no scaler is applied. Default is MinMaxScaler.
    estimator           :   Estimator for the regression task. Default is a LGBM regressor.
    target_transformer  :   Transformer (or scaler) for the target variable. If set to False, no transformation is applied. Default is MirrorLogNormScaler. Transformer must implement fit, transform, and inverse_transform methods.
    '''

    def __init__(self, rfecv_kwargs=None, scaler=None, estimator=None, target_transformer=None):

        self.rfecv_kwargs = rfecv_kwargs
        self.scaler = scaler
        self.estimator = estimator
        self.target_transformer = target_transformer

        rfecv_kwargs_ = {'estimator':lgb.LGBMRegressor(verbose=-1), 'cv':TimeSeriesSplit(), 'step':2} if self.rfecv_kwargs is None else self.rfecv_kwargs
        scaler_ = MinMaxScaler() if self.scaler is None else self.scaler
        estimator_ = lgb.LGBMRegressor(verbose=-1) if self.estimator is None else self.estimator
        target_transformer_ = MirrorLogNormScaler() if self.target_transformer is None else self.target_transformer

        pipeline = []
        if self.rfecv_kwargs is not False:
            pipeline.append(('feature_elimination', RFECV(**rfecv_kwargs_)))
        if self.scaler is not False:
            pipeline.append(('normalization', scaler_))
        pipeline.append(('estimation', estimator_))
        pipe = Pipeline(pipeline)

        if self.target_transformer is not False:
            self.model = TransformedTargetRegressor(
                regressor=pipe,
                transformer=target_transformer_,
                check_inverse=False
            )
        else:
            self.model = pipe

    def fit(self, X, y):

        '''
        X   :   Training matrix.
        y   :   Target values.
        '''

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.model.fit(X, y)
        return self
    
    def predict(self, X):

        '''
        X   :   Samples.
        '''

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return self.model.predict(X)