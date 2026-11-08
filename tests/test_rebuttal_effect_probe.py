import numpy as np
import pytest
from fastwam.research.effect_probe import ridge_predict,centered_error


def test_train_only_ridge_is_finite_and_does_not_mutate_the_inputs():
    x=np.array([[-1.,0],[0,0],[1,0]]);y=np.array([[-2.],[0.],[2.]])
    original=x.copy();heldout=np.array([[-.5,99],[.5,-99]])
    prediction,receipt=ridge_predict(x,y,heldout)
    assert np.array_equal(x,original) and np.isfinite(prediction).all()
    assert receipt['active_features']==1 and prediction[0,0]<0<prediction[1,0]
    assert centered_error(prediction,np.array([[-1.],[1.]]))['centered_explained_variance']>.9


def test_constant_training_features_do_not_manufacture_candidate_variation():
    prediction,receipt=ridge_predict(np.ones((3,2)),np.arange(6).reshape(3,2),np.zeros((4,2)))
    assert np.array_equal(prediction,np.broadcast_to([2.,3.],(4,2))) and receipt['active_features']==0
    assert centered_error(prediction,np.zeros((4,2)))['centered_explained_variance'] is None


def test_centered_evaluation_removes_only_query_offsets():
    truth=np.array([[0.,1],[2,3]])
    assert centered_error(truth+20,truth)['centered_mse']==0
    assert centered_error(np.zeros_like(truth),truth)['centered_explained_variance']==0
    with pytest.raises(ValueError):centered_error(truth[:1],truth[:1])
