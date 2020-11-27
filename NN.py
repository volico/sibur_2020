import torch
import pytorch_lightning as pl
from pytorch_forecasting.metrics import MAPE

class simple_torchpl(pl.LightningModule):

    def __init__(self, n_in, n_h_1, n_h_2, n_out,
                 activation1, activation2,
                 optimizer, lr, weight_decay, p_1, p_2):
        super(simple_torchpl, self).__init__()
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_losses = []
        self.train_losses = []

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

        if activation2 == 'Tanh':
            self.activ2 = torch.nn.Tanh()
        elif activation2 == 'Hardtanh':
            self.activ2 = torch.nn.Hardtanh()
        elif activation2 == 'Hardshrink':
            self.activ2 = torch.nn.Hardshrink()
        elif activation2 == 'ELU':
            self.activ2 = torch.nn.ELU()
        elif activation2 == 'SELU':
            self.activ2 = torch.nn.SELU()
        elif activation2 == 'ReLU':
            self.activ2 = torch.nn.ReLU()
        elif activation2 == 'Tanhshrink':
            self.activ2 = torch.nn.Tanhshrink()
        elif activation2 == 'CELU':
            self.activ2 = torch.nn.CELU()


        self.lstm = torch.nn.LSTM(n_in, n_h_1, batch_first=True)
        self.dropout1 = torch.nn.Dropout(p_1)
        self.dropout2 = torch.nn.Dropout(p_2)
        self.linear1 = torch.nn.Linear(n_h_1, n_h_2)
        self.linear_final = torch.nn.Linear(n_h_2, n_out)
        self.loss = torch.nn.L1Loss()
        self.loss_sec = MAPE()

    def forward(self, x):

        x, _ = self.lstm(x)
        del _
        torch.cuda.empty_cache()
        x = x[:, -1, :]
        x = self.activ1(x)

        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.activ2(x)

        x = self.dropout2(x)
        predictions = self.linear_final(x)
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        train_loss = self.loss(y_pred, y)
        self.train_losses.append(self.loss_sec(y_pred, y).item())
        print('train_loss:', self.loss_sec(y_pred, y).item())
        if train_loss.item()> 100_000:
            print('fuck')
            train_loss = train_loss*0

        return (train_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        test_loss = self.loss(y_pred, y)
        self.log('val_loss', self.loss_sec(y_pred, y).item())
        self.val_losses.append(self.loss_sec(y_pred, y).item())
        print('val_loss:', self.loss_sec(y_pred, y).item())
        return (self.loss_sec(y_pred, y).item())

    def configure_optimizers(self):
        if self.optimizer == 'AdamW':
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'Adadelta':
            return torch.optim.Adadelta(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'Adagrad':
            return torch.optim.Adagrad(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'Adamax':
            return torch.optim.Adamax(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'ASGD':
            return torch.optim.ASGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'LBFGS':
            return torch.optim.LBFGS(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'Rprop':
            return torch.optim.Rprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            return torch.optim.AdamW(self.parameters(), lr=self.lr)