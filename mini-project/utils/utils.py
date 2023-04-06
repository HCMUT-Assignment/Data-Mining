import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler
from .preprocess import Preprocessing
import torch 
from .dataset import get_data_loader

def read_csv(path: str, train_cols, target_cols, return_np: bool):
    """
    Args:
        path: path to the .csv file
        col_name: names of columns to extract features
        return_np:  if true then return the numpy array of features
    Return:
        results: dict{
            "frame": original frame read by pandas given the .csv file
            "frame_cols": frame that just includes the given column names
            "feature":  if return_np is true then return the numpy array for the features
        }
        , col_name: name of the column
    """
    results = dict()
    df                          = pd.read_csv(path)
    feature_frame               = df[train_cols]
    target_frame                = df[target_cols]

    results['frame']            = df
    results['feature_frame']    = feature_frame
    results['target_frame']     = target_frame
    results['feature']          = None
    results['target']           = None
    
    if return_np:

        feature_arr = feature_frame.to_numpy()
        target_arr  = target_frame.to_numpy()

        if not isinstance(feature_arr, np.ndarray):
            raise TypeError("Feature array must be numpy array")
        results['feature']  = feature_arr
        results['target']   = target_arr

    return results

def visualize_trends(features: List, cls) -> None:
    """
    Plot lines to reflect trends of several indicators
    """
    assert len(features) == len(cls), "Size is not compatible between the number of features and indicators given"
    for idx, item in enumerate(features):
        print(cls[idx])
        plt.plot(item, label = cls[idx])
        plt.legend(cls[idx]) 
    plt.show()

def create_data_train(seq_feature, seq_target, n_steps):
    """
    Arg:
        seq_feature: the raw feature of data
    Return:
        (x_train, y_train): torch tensor
        x_train: (total - num_steps + 1, num_steps, num_features)
        y_train: (total - num_steps + 1, 1)
    """
    total, _ = seq_feature.shape
    
    div      = total % n_steps
    index    = (total // n_steps) * n_steps
    seq_feature = seq_feature[:index, :]
    curr_step = 0
    x_train, y_train = [], []
    
    while(curr_step < index):
        if curr_step + n_steps >= index:
            break
        x_train.append(seq_feature[curr_step:curr_step+n_steps, :])
        y_train.append(seq_target[curr_step + n_steps])

        curr_step += 1
    
    assert len(x_train) == len(y_train), "Number of samples must be the same of labels"
    x_train = torch.Tensor(np.stack(x_train, axis = 0))
    y_train = torch.Tensor(np.stack(y_train, axis = 0))

    return x_train, y_train
