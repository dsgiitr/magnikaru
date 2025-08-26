
import torch
import lightning as L
import config as cf
import os

from Transformer_cross_attention import ChessTransformerClassification
from lightning_model import LitCNN, ChessNewDM
from utils import ChessDataset
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
    pytorchModel = ChessTransformerClassification()
    # checkpoint_evaluation_check=LitCNN.load_from_checkpoint("Laabhanvi_CNN.ckpt", model=pytorchModel)
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
        # test_acc = trainer.test(model=model, dataloaders=dm.test_dataloader())
        # print(f"Test accuracy: {test_acc}")

        FILE_NAME = os.path.join("lightning_check",f"train2_it_{iterations}_epoch_{cf.NUM_EPOCHS}_lr_{cf.LEARNING_RATE}.ckpt")
        trainer.save_checkpoint(FILE_NAME)