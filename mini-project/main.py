from utils.preprocess import Preprocessing
from utils.utils import read_csv, create_data_train
from utils.dataset import get_data_loader
from model import TSModel
from configs.config import *
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np

LOGGER          =   SummaryWriter()
PROCESSOR       =   Preprocessing()
TSCONFIG        =   TSConfig()
MODEL_CONFIG    =   ModelConfig()
DEVICE          =   torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def training_loop(model, train_loader, lr, n_iters: int = 100):

    model.train()
    losses = []
    optimizer   = torch.optim.SGD(model.parameters(), lr = lr)
    for epoch in range(n_iters):
        
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = batch
            output       = model(data.to(DEVICE))
            loss         = F.mse_loss(output, target.to(DEVICE))

            loss.backward() 
            optimizer.step()
        # LOGGER.add_scalar('loss/train', loss.item(), epoch) 
        print(f'Loss = {loss.item()}')
        losses.append(loss.item()) 

    return model, loss
def main():
    #   Read the csv file
    csv_stats       = read_csv(TSCONFIG.path_csv, TSCONFIG.train_features, TSCONFIG.pred_features, True)
    df              = csv_stats['frame']
    breakpoint()
    seq_features    = csv_stats['feature']
    seq_targets     = csv_stats['target']
    # breakpoint()    y_train.
    seq_features, feature_scale = PROCESSOR.apply_minmax_scale(seq_features)
    seq_targets,  target_scale  = PROCESSOR.apply_minmax_scale(seq_targets)
    # breakpoint()

    #   Pre-processing data

    
    #   Create training set and validation set
    #   Get dataloader for training
    x_train, y_train        = create_data_train(seq_feature=seq_features, seq_target=seq_targets, n_steps=TSCONFIG.n_steps)
    # breakpoint()
    train_loader            = get_data_loader((x_train, y_train), TSCONFIG.batch_size, False)
    # val_loader              = get_data_loader((x_val, y_val), 16, False)
    val_loader              = None
    #   Prepare model
    model                   =   TSModel(
                                n_steps     =   TSConfig.n_steps,
                                n_features  =   TSConfig.n_features,
                                n_classes   =   TSConfig.n_classes,
                                n_hidden    =   TSConfig.n_features,
                                bidirection =   ModelConfig.bidirection,
                                batch_first =   ModelConfig.batch_first,
                                num_layers  =   ModelConfig.num_layers,
                                lr          =   ModelConfig.lr
                            ).to(DEVICE)

    model, losses           = training_loop(model, train_loader, 1e-3, 100)
    #   Train model
    # trainer             =   pl.Trainer(accelerator = 'gpu', max_epochs=10)
    # trainer.fit(model, train_loader, val_loader) 

    #   Save model

def open_pickle(pkl_file):
    import pickle

    result = None
    with open(pkl_file, mode = 'rb') as reader:

        result = pickle.load(reader)
    return result

def get_file_name(path):
    return str(path).split('/')[-1].split('.') 

if __name__ == '__main__':
    # main() 
    save = BASEDIR / 'save'
    
    pkl_files = [item for item in save.glob('**/*.pkl')]

    
    for path in pkl_files:
        filename = get_file_name(path)[0]
        res = open_pickle(path)

        print(f'Filename : {filename}')
        # breakpoint()

        for key, values in res.items():

            if isinstance(values, np.float64):
                print(f'{key}: {values: 0,.2f}')
            
        print('===========================================================================================================')