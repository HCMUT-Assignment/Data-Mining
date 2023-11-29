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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LOGGER          =   SummaryWriter()
PROCESSOR       =   Preprocessing()
TSCONFIG        =   TSConfig()
MODEL_CONFIG    =   ModelConfig()
DEVICE          =   torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_test_split(ratio, X, Y):
    
    split_idx = int(X.shape[0] * ratio)
    x_train, y_train = X[:split_idx, :, :], Y[:split_idx, :]
    x_test, y_test   = X[split_idx:, :, :], Y[split_idx:, :]

    return x_train, y_train, x_test, y_test 

def training_loop(model, train_loader, lr, n_iters: int = 100):

    for item in model.parameters():
        item.requires_grad = True

    model.train()
    losses = []
    optimizer   = torch.optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-7)
    for epoch in range(n_iters):
        
        for idx, batch in enumerate(train_loader):
            
            data, target = batch
            # print(data.shape)
            output       = model(data.to(DEVICE))
            loss         = F.mse_loss(output, target.to(DEVICE))

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
        # LOGGER.add_scalar('loss/train', loss.item(), epoch) 
        print(f'Loss = {loss.item()}')
        losses.append(loss.item()) 

    return model, losses
def main(is_eval: bool):
    #   Read the csv file
    csv_stats       = read_csv(TSCONFIG.path_csv, TSCONFIG.train_features, TSCONFIG.pred_features, True)
    epochs          = 100
    df              = csv_stats['frame']
    # breakpoint()
    seq_features    = csv_stats['feature']
    seq_targets     = csv_stats['target']
    # breakpoint()    y_train.
    seq_features, feature_scale = PROCESSOR.apply_minmax_scale(seq_features)
    seq_targets,  target_scale  = PROCESSOR.apply_minmax_scale(seq_targets)
    # breakpoint()
    ratio = 0.8
    #   Pre-processing data

    
    #   Create training set and validation set
    #   Get dataloader for training
    X, Y        = create_data_train(seq_feature=seq_features, seq_target=seq_targets, n_steps=TSCONFIG.n_steps)
    x_train, y_train, x_test, y_test = train_test_split(ratio, X, Y)

    # breakpoint()
    train_loader            = get_data_loader((x_train, y_train), TSCONFIG.batch_size, False)
    val_loader              = get_data_loader((x_test, y_test), 64, False)
    val_loader              = None
    #   Prepare model
    model                   =   TSModel(
                                n_steps     =   TSConfig.n_steps,
                                n_features  =   TSConfig.n_features,
                                n_classes   =   TSConfig.n_classes,
                                n_hidden    =   TSCONFIG.n_features, 
                                bidirection =   ModelConfig.bidirection,
                                batch_first =   ModelConfig.batch_first,
                                num_layers  =   ModelConfig.num_layers,
                                lr          =   ModelConfig.lr
                            ).to(DEVICE)

    if not is_eval:
        model, losses           = training_loop(model, train_loader, 1e-3, epochs)
        torch.save(model.state_dict(), f'save/ckpts/modelAdam_epoch_{epochs}_2') 
        plt.plot(losses)
        plt.savefig(f'save/losses_adam.png')
    else:
        model.load_state_dict(torch.load(f'save/ckpts/modelAdam_epoch_{epochs}_2'))
        eval(model, x_test, y_test,TSCONFIG, target_scale)
    #   Train model
    # trainer             =   pl.Trainer(accelerator = 'gpu', max_epochs=10)
    # trainer.fit(model, train_loader, val_loader) 

    #   Save model

#Evaluation
def eval(model, x_test, y_test, ts_config, target_scale):
    print(f'X test shape = {x_test.shape}\nY test shape = {y_test.shape}')
    model.eval()
    for item in model.parameters():
        item.requires_grad = False

    with torch.no_grad():
        x_test = x_test.to(DEVICE)
        test_predict = model(x_test)

        test_predict = test_predict.detach().cpu().numpy()
        y_test       = y_test.detach().cpu().numpy()
        
        target_max_scale, target_min_scale = target_scale
        test_pred_scale = (test_predict * (target_max_scale - target_min_scale)) + target_min_scale
        y_test_scale = (y_test * (target_max_scale - target_min_scale) + target_min_scale)

        for idx, category in enumerate(ts_config.pred_features):
            cate_mse = mean_squared_error(y_test_scale[:, idx], test_pred_scale[:, idx])
            cate_mae = mean_absolute_error(y_test_scale[:, idx], test_pred_scale[:, idx])
            cate_rsqr = r2_score(y_test_scale[:, idx], test_pred_scale[:, idx])

            print(f'MSE in {ts_config.pred_features[idx]} = {cate_mse}')
            print(f'MAE in {ts_config.pred_features[idx]} = {cate_mae}')
            print(f'R2 score in {ts_config.pred_features[idx]} = {cate_rsqr}')
    breakpoint()
    
    

def validate(model, val_loader):
    for item in model.parameters():
        item.requries_grad = False

    model.eval()
    losses = []
    count = 0
    for idx, batch in enumerate(val_loader):
        x, y = batch
        out = model(x)
        loss = F.mse_loss(out, y)
        losses.append(loss.item())
        print(f'Val_loss = {loss.item()}')

def open_pickle(pkl_file):
    import pickle

    result = None
    with open(pkl_file, mode = 'rb') as reader:

        result = pickle.load(reader)
    return result

def get_file_name(path):
    return str(path).split('/')[-1].split('.') 

if __name__ == '__main__':
    main(is_eval= True) 
    # save = BASEDIR / 'save'
    
    # pkl_files = [item for item in save.glob('**/*.pkl')]

    
    # for path in pkl_files:
    #     filename = get_file_name(path)[0]
    #     res = open_pickle(path)

    #     print(f'Filename : {filename}')
    #     # breakpoint()

    #     for key, values in res.items():

    #         if isinstance(values, np.float64):
    #             print(f'{key}: {values: 0,.2f}')
            
    #     print('===========================================================================================================')