from dataclasses import dataclass
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent.parent

@dataclass
class TSConfig:
    
    train_features  = [
        'high',
        'low',
        'MA',
        'MACD',
        'open',
        'close',
        'Volume',
        'Volume MA',
        'MA',
        'Histogram',
        'MACD',
        'Signal'
    ]
    pred_features   = [
        'high',
        'low',
        'MA',
        'MACD'
    ]

    n_steps         =   16
    n_classes       =   len(pred_features)
    n_features      =   len(train_features)
    path_csv        =   BASEDIR / 'dm-data-v1.csv'
    batch_size      =   64
@dataclass
class ModelConfig:

    bidirection     =   True
    batch_first     =   True
    num_layers      =   1
    lr              = 1e-3