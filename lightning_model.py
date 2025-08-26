
import torch
import lightning as L
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader
import os
import config as cf

from utils import ChessDataset

class LitCNN(L.LightningModule):
    def __init__(self, model, lr,K=0):
        super().__init__()
        self.model = model
        self.learning_rate = lr
        
        self.save_hyperparameters(ignore=["model"])
        self.criterion = nn.BCELoss()

        self.train_acc = torchmetrics.Accuracy(task='binary',threshold=0.5)
        self.val_correct = {}
        self.val_total = {}

        # self.val_acc = torchmetrics.Accuracy(task='binary',threshold=0.5)
        # self.K = cf.NUM_EPOCHS # Change this later
        # self.current_K = 0

        # self.seen = torch.zeros(self.K)
        # self.correct = torch.zeros(self.K)
    
    def forward(self, board_batch, info_batch):
        return self.model(board_batch, info_batch)
    
    def _shared_step(self,batch):
        # Shared code between train val and test
        k, features, y_batch = batch
        board_batch, info_batch = features
        
        out_probabilities = self(board_batch, info_batch)
        loss = self.criterion(out_probabilities, y_batch)
    
        return k, loss, y_batch, out_probabilities
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        _, loss, labels, out_probabilities = self._shared_step(batch=batch)
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        
        # Logging statement for Tensor Board
        # TODO: Figure out how to log the accuracies by K value
        self.train_acc(out_probabilities, labels)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=True)    
        
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        k, loss, labels, out_probabilities = self._shared_step(batch)

        preds = (out_probabilities > 0.5).float()
        correct = (preds == labels).float()

        # per k values
        for idx, val in enumerate(k):
            val_k = val.item()
            if val_k not in self.val_correct:
                self.val_correct[val_k] = 0.0
                self.val_total[val_k] = 0.0
            self.val_correct[val_k] += correct[idx].item()
            self.val_total[val_k] += 1

        #
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def on_validation_epoch_end(self):
        
        for k_val in self.val_correct.keys():
            acc = self.val_correct[k_val] /  self.val_total[k_val]
            self.log(f"val_acc/k={k_val}", acc, prog_bar=False, on_epoch=True)

       
        self.val_correct.clear()
        self.val_total.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# Datamodule using iterable dataset 
class ChessNewDM(L.LightningDataModule):
    def __init__(self, train_csv:str ="", test_csv:str="", sampling_probabilities = None, mode='train',batch_size:int=128,K:int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.K = K

    def train_dataloader(self):
        print("Current epoch: ",self.trainer.current_epoch)
        sample_K = self.trainer.current_epoch   
        self.chess_train = ChessDataset(end_steps=sample_K, train_csv=self.train_csv, test_csv=self.test_csv, sampling_probabilities = None, mode='train')
        return DataLoader(self.chess_train, batch_size=self.batch_size, num_workers=cf.NUM_WORKERS) # There is also a drop_last parameters to drop the non-full batch

    def val_dataloader(self):
        print("Current epoch: ",self.trainer.current_epoch)
        sample_K = cf.NUM_EPOCHS - 1
        self.chess_val = ChessDataset(end_steps=sample_K, train_csv=self.train_csv, test_csv=self.test_csv, sampling_probabilities = None, mode='test')
        return DataLoader(self.chess_val, batch_size=self.batch_size,num_workers=cf.NUM_WORKERS)
