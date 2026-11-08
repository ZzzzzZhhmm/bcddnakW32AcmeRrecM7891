"""Exploratory ridge probes with train-only fitting and explicit state holdouts."""
from __future__ import annotations
import numpy as np


def ridge_predict(train_x,train_y,test_x,*,relative_regularization=.1):
    x=np.asarray(train_x,dtype=np.float64);y=np.asarray(train_y,dtype=np.float64);test=np.asarray(test_x,dtype=np.float64)
    if x.ndim!=2 or y.ndim!=2 or test.ndim!=2 or len(x)!=len(y) or x.shape[1]!=test.shape[1] or len(x)<2 or not x.shape[1]:
        raise ValueError('matched nonempty training and test feature matrices required')
    if not all(np.isfinite(v).all() for v in (x,y,test)) or not relative_regularization>0:
        raise ValueError('finite data and positive fixed regularization required')
    mean=x.mean(0);std=x.std(0);active=std>1e-8
    target_mean=y.mean(0)
    if not active.any():return np.broadcast_to(target_mean,(len(test),len(target_mean))).copy(),dict(active_features=0,regularization=None)
    standardized=(x[:,active]-mean[active])/std[active]
    heldout=(test[:,active]-mean[active])/std[active]
    width=standardized.shape[1];kernel=standardized@standardized.T/width
    regularization=float(relative_regularization*np.trace(kernel)/len(x))
    coefficients=np.linalg.solve(kernel+regularization*np.eye(len(x)),y-target_mean)
    prediction=(heldout@standardized.T/width)@coefficients+target_mean
    return prediction,dict(active_features=int(active.sum()),regularization=regularization,
        training_rows=len(x),feature_scaling='Training rows only; no held-out features or outcomes fitted.')


def centered_error(predicted,observed):
    prediction=np.asarray(predicted,dtype=np.float64);truth=np.asarray(observed,dtype=np.float64)
    if prediction.ndim!=2 or prediction.shape!=truth.shape or len(truth)<2 or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError('at least two matched finite candidate effects required')
    p=prediction-prediction.mean(0);t=truth-truth.mean(0)
    variance=float(np.mean(t*t));error=float(np.mean((p-t)**2))
    return dict(centered_mse=error,observed_centered_variance=variance,
        centered_explained_variance=1-error/variance if variance>0 else None,
        complete_candidates=len(truth))
