import torch
import lightning as L
import config as cf
import os

from model import ChessCNN
from lightning_model import LitCNN, ChessDM, ChessNewDM
from utils import GMDataset
from torch.utils.data import DataLoader


input_size = 28*28
hidden_size = 100
num_classes = 10
num_epochs = cf.NUM_EPOCHS
batch_size = cf.BATCH_SIZE
learning_rate = cf.LEARNING_RATE

torch.manual_seed(cf.SEED)
torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    
    # dm = ChessDM(batch_size=cf.BATCH_SIZE)
    dm = ChessNewDM(train_csv=cf.TRAIN_PATH,test_csv=cf.TEST_PATH,batch_size=cf.BATCH_SIZE)
    pytorchModel = ChessCNN()
    model = LitCNN(model=pytorchModel, lr=learning_rate)
    
    for iterations in range(1):
        print(f"\n---- ITERATION: {iterations} ----")
        trainer = L.Trainer(
            max_epochs=num_epochs,
            accelerator="auto",
            devices="auto",
            reload_dataloaders_every_n_epochs=1
        )
        
        trainer.fit(
            model=model,
            datamodule=dm
        )

        FILE_NAME = os.path.join("lightning_check",f"train2_it_{iterations}_epoch_{cf.NUM_EPOCHS}_lr_{cf.LEARNING_RATE}.ckpt")
        trainer.save_checkpoint(FILE_NAME)
    
    # Testing code
    acc = dict()
    for i in range(3):
        new_data_set = GMDataset(end_steps=i, train_csv=cf.TRAIN_PATH, test_csv=cf.TEST_PATH, sampling_probabilities = None, mode='test')
        new_test_dataloader = DataLoader(new_data_set, cf.BATCH_SIZE)
        test_acc = trainer.test(model=model, dataloaders=new_test_dataloader)[0]['accuracy']
        acc[i] = test_acc
        print(f"K={i}, Test accuracy: {test_acc}")    

    for k,v in acc.items():
        print(f"K: {k}, acc: {v}")
