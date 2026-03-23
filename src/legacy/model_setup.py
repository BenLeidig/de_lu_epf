import torch
import torch.nn as nn
import lightning.pytorch as pl
from pytorch_tcn import TCN

torch.set_float32_matmul_precision('high')
    

class TCN_LSTM_MHA(pl.LightningModule):
    def __init__(
        self,
        input_size:int,
        channel_sizes:list,
        kernel_size:int,
        hidden_sizes:list,
        tcn_dropout:float,
        lstm_dropouts:list,
        mha_dropout:float,
        mha_heads:int,
        lr_init:float
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = torch.nn.L1Loss()
        self.lr_init = lr_init

        #### model ####
        # ReLU
        self.relu = nn.ReLU()
        
        # TCN
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=channel_sizes,
            kernel_size=kernel_size,
            dropout=tcn_dropout,
            use_skip_connections=True,
            input_shape='NLC'   # (batch_size, time_steps, feature_channels)
        )

        # LSTM
        self.lstm0 = nn.LSTM(input_size=channel_sizes[-1], hidden_size=hidden_sizes[0], batch_first=True)
        self.dropout0 = nn.Dropout(lstm_dropouts[0])
        self.lstm1 = nn.LSTM(input_size=hidden_sizes[0], hidden_size=hidden_sizes[1], batch_first=True)
        self.dropout1 = nn.Dropout(lstm_dropouts[1])
        self.lstm2 = nn.LSTM(input_size=hidden_sizes[1], hidden_size=hidden_sizes[2], batch_first=True)
        
        # MHA
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_sizes[-1],
            num_heads=mha_heads,
            dropout=mha_dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_sizes[-1])

        # head
        self.fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        # TCN
        x = self.tcn(x)
        x = self.relu(x)
        
        # LSTM
        x, _ = self.lstm0(x)
        x = self.relu(x)
        x = self.dropout0(x)

        x, _ = self.lstm1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)

        # MHA
        mha_out, _ = self.mha(x, x, x)
        x = self.norm(x+mha_out)

        # head
        return self.fc(x[:, -24:, :]).squeeze(-1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        return val_loss
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_init)
        return optimizer