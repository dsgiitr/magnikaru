import torch
import lightning as L
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader
import os
from model import ChessDualDatasetNew

class LitCNN(L.LightningModule):
    def __init__(self, model, lr,K=0):
        super().__init__()
        self.model = model
        self.learning_rate = lr
        
        self.save_hyperparameters(ignore=["model"])
        self.criterion = nn.BCELoss()

        self.train_acc = torchmetrics.Accuracy(task='binary',threshold=0.5)
        self.test_acc = torchmetrics.Accuracy(task='binary',threshold=0.5)
        self.K = K
        self.current_K = 0
    
    def forward(self, board_batch, info_batch):
        return self.model(board_batch, info_batch)
    
    
    def _shared_step(self,batch):
        # Shared code between train val and test
        features, y_batch = batch
        board_batch, info_batch = features
        
        board_batch = board_batch
        info_batch  = info_batch
        y_batch      = y_batch
        out_probabilities = self(board_batch, info_batch)
        loss = self.criterion(out_probabilities, y_batch)
    
        return loss, y_batch, out_probabilities
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss, labels, out_probabilities = self._shared_step(batch=batch)
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.train_acc(out_probabilities, labels)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)    
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # not run during training so have to call it later manually
        loss, labels, out_probabilities = self._shared_step(batch=batch)
  
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.test_acc(out_probabilities, labels)
        self.log("accuracy", self.test_acc, prog_bar=True, on_epoch=True, on_step=False)    
        
        return loss 
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

class ChessDM(L.LightningDataModule):
    def __init__(self, data_dir= "./data", batch_size = 32, K:int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_csv = os.path.join("data","preprocessed","train.csv")
        self.test_csv = os.path.join("data","preprocessed","test.csv")
        self.K = K
           
    def setup(self, stage: str):
        sample_K = self.trainer.current_epoch
        self.chess_test = ChessDualDatasetNew(train=False, K=sample_K)

    def train_dataloader(self):
        print("Current epoch: ",self.trainer.current_epoch)
        sample_K = self.trainer.current_epoch
        self.chess_train = ChessDualDatasetNew(train=True, K=sample_K)
        return DataLoader(self.chess_train, batch_size=self.batch_size, shuffle=True) # There is also a drop_last parameters to drop the non-full batch

    def test_dataloader(self):
        sample_K = self.trainer.current_epoch
        self.chess_test = ChessDualDatasetNew(train=False, K=sample_K)
        return DataLoader(self.chess_test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        sample_K = self.trainer.current_epoch
        self.chess_predict = ChessDualDatasetNew(train=True, K=sample_K)
        return DataLoader(self.chess_predict, batch_size=self.batch_size, shuffle=False)

