import torch
import pytorch_lightning as pl
from pytorch_forecasting.metrics import MAPE

class simple_torchpl(pl.LightningModule):
    ''' pytorch lightning модель
    '''

    def __init__(self,  n_h_1, activation1, optimizer, lr, weight_decay, p_1, p_2, loss):
        super(simple_torchpl, self).__init__()
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay

        if activation1 == 'Tanh':
            self.activ1 = torch.nn.Tanh()
        elif activation1 == 'Hardtanh':
            self.activ1 = torch.nn.Hardtanh()
        elif activation1 == 'Hardshrink':
            self.activ1 = torch.nn.Hardshrink()
        elif activation1 == 'ELU':
            self.activ1 = torch.nn.ELU()
        elif activation1 == 'SELU':
            self.activ1 = torch.nn.SELU()
        elif activation1 == 'ReLU':
            self.activ1 = torch.nn.ReLU()
        elif activation1 == 'Tanhshrink':
            self.activ1 = torch.nn.Tanhshrink()
        elif activation1 == 'CELU':
            self.activ1 = torch.nn.CELU()

        self.dropout1 = torch.nn.Dropout(p_1)
        self.dropout2 = torch.nn.Dropout(p_2)
        self.linear1 = torch.nn.Linear(10, n_h_1)
        self.linear_final = torch.nn.Linear(n_h_1, 1)
        if loss == 'MSE':
            self.loss = torch.nn.MSELoss()
        elif loss == 'MAE':
            self.loss = torch.nn.L1Loss()
        elif loss == 'MAPE':
            self.loss = MAPE()
        self.loss_sec = MAPE()

    def forward(self, x):
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.activ1(x)

        x = self.dropout2(x)
        predictions = self.linear_final(x)
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        train_loss = self.loss(y_pred, y)

        return (train_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        test_mape_loss = self.loss_sec(y_pred, y).item()
        self.log('val_loss', test_mape_loss)
        return (test_mape_loss)

    def configure_optimizers(self):
        if self.optimizer == 'AdamW':
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)