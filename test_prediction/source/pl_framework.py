import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import os
import random

random_state = 54321
def set_seed(seed = random_state) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class nn_training:
    def __init__(self, model, X, y, device=torch.device('cuda:0')):

        self.X = X
        self.y = y
        self.model = model
        self.device = device

    def data_loaders(self, fold, X, y, batch_size):
        X_train = torch.from_numpy(X[fold[0]]).float().to(self.device)
        y_train = torch.from_numpy(y[fold[0]]).float().to(self.device)
        X_test = torch.from_numpy(X[fold[1]]).float().to(self.device)
        y_test = torch.from_numpy(y[fold[1]]).float().to(self.device)

        set_seed(seed=random_state)
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=False)

        val_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=y_test.shape[0], shuffle=False)
        del X_train, X_test, y_test, y_train
        torch.cuda.empty_cache()

        return (train_loader, val_loader)

    def train(self, min_epochs, max_epochs, model_params, fold, batch_size, val_fold=True):

        train_loader, val_loader = self.data_loaders(fold, self.X, self.y, batch_size)
        if val_fold == True:
            trainer = pl.Trainer(min_epochs=min_epochs,
                                 max_epochs=max_epochs,
                                 progress_bar_refresh_rate=0,
                                 callbacks=[EarlyStopping(min_delta=0.00001, patience=5, monitor='val_loss')],
                                 num_sanity_val_steps=0,
                                 gpus=1,
                                 logger=False,
                                 checkpoint_callback=False)
        else:
            trainer = pl.Trainer(min_epochs=min_epochs,
                                 max_epochs=max_epochs,
                                 progress_bar_refresh_rate=0,
                                 num_sanity_val_steps=0,
                                 gpus=1,
                                 logger=False,
                                 checkpoint_callback=False)

        my_model = self.model(**model_params)
        ## Тренирум модель
        trainer.fit(my_model, train_loader, val_loader)
        self.trained_model = my_model
        self.trainer = trainer