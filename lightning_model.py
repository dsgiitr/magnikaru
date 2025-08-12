
import torch
import lightning as L
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader
import os
from model import ChessDualDatasetNew
import config as cf

from utils import GMDataset

class LitCNN(L.LightningModule):
    def __init__(self, model, lr,K=0):
        super().__init__()
        self.model = model
        self.learning_rate = lr
        
        self.save_hyperparameters(ignore=["model"])
        self.criterion = nn.BCELoss()

        self.train_acc = torchmetrics.Accuracy(task='binary',threshold=0.5)
        self.test_acc = torchmetrics.Accuracy(task='binary',threshold=0.5)
        self.K = cf.NUM_EPOCHS # Change this later
        self.current_K = 0

        self.seen = torch.zeros(self.K)
        self.correct = torch.zeros(self.K)
    
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
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        
        # Logging statement for Tensor Board
        # TODO: Figure out how to log the accuracies by K value
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)    
        self.train_acc(out_probabilities, labels)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # not run during training so have to call it later manually
        k, loss, labels, out_probabilities = self._shared_step(batch=batch)

        for idx, val in enumerate(k):
            self.seen[val] += 1
            if abs(labels[idx]-out_probabilities[idx]) < 0.5:
                self.correct[val]+=1

        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        print(f"TEST STEP: correct: {self.correct} \n seen: {self.seen} \n\n")

        return loss 
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def on_test_epoch_end(self):
        print("Inside epoch end")
        self.accuracies = torch.div(self.correct, self.seen)

        # Write code to log this multiclass accuracy to tensorboard
        print(f"Accuracy: {self.accuracies}")
        self.log("test_log_k", self.accuracies, prog_bar=True, on_epoch=True, on_step=False)
        return 

    

# Datamodule using pytorch dataset
class ChessDM(L.LightningDataModule):
    def __init__(self, data_dir= "./data", batch_size = 32, K:int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_csv = os.path.join("data","preprocessed","train.csv")
        self.test_csv = os.path.join("data","preprocessed","test.csv")
        self.K = K

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
        self.chess_train = GMDataset(end_steps=sample_K, train_csv=self.train_csv, test_csv=self.test_csv, sampling_probabilities = None, mode='train')
        return DataLoader(self.chess_train, batch_size=self.batch_size) # There is also a drop_last parameters to drop the non-full batch

    def test_dataloader(self):
        print("Current epoch: ",self.trainer.current_epoch)
        sample_K = cf.NUM_EPOCHS - 1
        self.chess_test = GMDataset(end_steps=sample_K, train_csv=self.train_csv, test_csv=self.test_csv, sampling_probabilities = None, mode='test')
        return DataLoader(self.chess_test, batch_size=self.batch_size)
