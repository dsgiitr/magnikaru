import torch
from torch import nn
from model import ChessCNN, ChessDualDatasetNew
from torch import optim
from tqdm import tqdm
import config as cf
from preprocess import get_dataloader
import os
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NUM_EPOCHS   = cf.NUM_EPOCHS
BATCH_SIZE   = cf.BATCH_SIZE
LEARNING_RATE = cf.LEARNING_RATE


# train_loader, test_loader = get_dataloader(K=1)
train_ds = ChessDualDatasetNew(train=True, K=0)
test_ds = ChessDualDatasetNew(train=False, K=0)

train_loader = DataLoader(dataset=train_ds, batch_size=cf.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_ds, batch_size=cf.BATCH_SIZE, shuffle=False)

model = ChessCNN().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Training Loop
for epoch in tqdm(range(NUM_EPOCHS)):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        # board_batch: (B, 7, 8, 8)
        # info_batch:  (B, 13, 1)
        # y_batch:     (B, 1) after unsqueeze
        
        features, y_batch = batch
        board_batch, info_batch = features
        
        board_batch = board_batch.to(device)
        info_batch  = info_batch.to(device)
        y_batch      = y_batch.to(device)
        outputs = model(board_batch, info_batch)    # (B, 1)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * board_batch.size(0)
        preds = (outputs >= 0.5).float()
        train_correct += (preds == y_batch).sum().item()
        train_total += board_batch.size(0)

    epoch_train_loss = train_loss / train_total
    epoch_train_acc  = train_correct / train_total

    print(
        f"Epoch {epoch+1:02d}/{NUM_EPOCHS}  "
        f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.3f}  "
    )

FILE_NAME = os.path.join("check","test1.pth")
torch.save(model.state_dict(), FILE_NAME)
print(f"Model saved to {FILE_NAME}")

model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for (board_batch, info_batch), y_batch in test_loader:
        board_batch = board_batch.to(device)
        info_batch  = info_batch.to(device)
        y_batch      = y_batch.to(device)

        test_outputs = model(board_batch, info_batch)
        tpreds = (test_outputs >= 0.5).float()
        test_correct += (tpreds == y_batch).sum().item()
        test_total += board_batch.size(0)

test_acc = test_correct / test_total
print(f"Test Accuracy: {test_acc:.3f}")