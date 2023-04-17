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
        # 'MACD',
        'open',
        'close',
        # 'Volume',
        # 'Volume MA',
        # 'MA',
        # 'Histogram',
        # 'MACD',
        # 'Signal'
    ]

    n_steps         =   7
    n_classes       =   len(pred_features)
    n_features      =   len(train_features)
    path_csv        =   BASEDIR / 'data_preprocess.csv'
    batch_size      =   512
@dataclass
class ModelConfig:

    bidirection     =   True
    batch_first     =   True
    num_layers      =   2
    lr              =   1e-3