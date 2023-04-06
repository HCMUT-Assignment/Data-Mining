from utils.preprocess import Preprocessing
from utils.utils import read_csv, create_data_train
from utils.dataset import get_data_loader
from model import TSModel
from configs.config import *
import pytorch_lightning as pl
import torch
from tqdm import tqdm


TSCONFIG        =   TSConfig()
MODEL_CONFIG    =   ModelConfig()
DEVICE          =   torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    #   Read the csv file
    csv_stats       = read_csv(TSCONFIG.path_csv, TSCONFIG.train_features, TSCONFIG.pred_features, True)
    seq_features    = csv_stats['feature']
    seq_targets     = csv_stats['target']

    #   Pre-processing data

    
    #   Create training set and validation set
    #   Get dataloader for training
    x_train, y_train        = create_data_train(seq_feature=seq_features, seq_target=seq_targets, n_steps=TSCONFIG.n_steps)
    train_loader            = get_data_loader((x_train, y_train), TSCONFIG.batch_size, False)
    # val_loader              = get_data_loader((x_val, y_val), 16, False)
    val_loader              = None
    #   Prepare model
    model               =   TSModel(
                                n_steps     =   TSConfig.n_steps,
                                n_features  =   TSConfig.n_features,
                                n_classes   =   TSConfig.n_classes,
                                n_hidden    =   TSConfig.n_features,
                                bidirection =   ModelConfig.bidirection,
                                batch_first =   ModelConfig.batch_first,
                                num_layers  =   ModelConfig.num_layers,
                                lr          =   ModelConfig.lr
                            ).to(DEVICE)


    #   Train model
    trainer             =   pl.Trainer(accelerator = 'gpu', max_epochs=10)
    trainer.fit(model, train_loader, val_loader) 

    #   Save model

main()