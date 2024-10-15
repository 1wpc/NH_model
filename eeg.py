import os
import time
import torch
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms
from torcheeg.model_selection import KFoldGroupbyTrial
from torch.utils.data import DataLoader
from torcheeg.models import CCNN
from net import Net
from torcheeg.models import TSCeption
from torcheeg.model_selection import train_test_split
from torcheeg.trainers import ClassifierTrainer
import logging

import pytorch_lightning as pl
import numpy as np
from torcheeg.datasets.constants import \
    DEAP_CHANNEL_LOCATION_DICT

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = torch.nn.CrossEntropyLoss()
test_accs = []
test_losses = []

os.makedirs('./examples_vanilla_torch/log', exist_ok=True)
logger = logging.getLogger('Training models with vanilla PyTorch')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(
    os.path.join('./examples_vanilla_torch/log', f'{timeticks}.log'))
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def valid(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.to(device)
    model.eval()
    loss, correct = 0, 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            y_ = model(X)
            _,pred=torch.max(y_.data,1)
            loss += loss_fn(y_, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct+=(pred==y).sum().item()
            total += X.shape[0]
    loss /= num_batches
    correct /= total
    logger.info(
        f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n"
    )

    return correct, loss

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    record_step = int(len(dataloader) / 10)
    
    model.to(device)
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        X = batch[0].to(device)
        y = batch[1].to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % record_step == 0:
            loss, current = loss.item(), batch_idx * len(X)
            logger.info(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss

def format_channel_location_dict(channel_list, location_list):
    location_list = np.array(location_list)
    output = {}
    for channel in channel_list:
        if len(np.argwhere(location_list == channel)):
            location = (np.argwhere(location_list == channel)[0]).tolist()
            output[channel] = location
        else:
            output[channel] = None
    return output

DEAP_CHANNEL_LIST_1 = [
    'FP1', 'C3', 'P7', 'O1', 'FP2', 'C4', 'P8', 'O2'
]

DEAP_LOCATION_LIST = [['-', '-', 'FP1', 'FP2', '-', '-'],
                      ['-', 'C3', '-', '-', 'C4', '-'],
                      ['P7', '-', '-', '-', '-', 'P8'],
                      ['-', '-', 'O1', 'O2', '-', '-']]

DEAP_CHANNEL_LOCATION_DICT_8c = format_channel_location_dict(
    DEAP_CHANNEL_LIST_1, DEAP_LOCATION_LIST)

dataset = DEAPDataset(root_path='./data_preprocessed_python/data_preprocessed_python',
                      offline_transform=transforms.Compose([
        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT_8c, apply_to_baseline=True)
    ]),
    online_transform=transforms.Compose(
        [transforms.BaselineRemoval(),
         transforms.ToTensor()]),
    label_transform=transforms.Compose([
        transforms.Select('valence'),
        transforms.Binary(5.0),
    ]),
                      io_path='.torcheeg/datasets_8channels_full_small4_6')

# dataset = DEAPDataset(root_path='./data_preprocessed_python/data_preprocessed_python',
#                       chunk_size=512,
#                       num_baseline=1,
#                       baseline_chunk_size=512,
#                       offline_transform=transforms.Compose([
#                           transforms.PickElectrode(transforms.PickElectrode.to_index_list(
#                           ['FP1', 'C3', 'P7', 'O1', 'FP2', 'C4', 'P8', 'O2'], DEAP_CHANNEL_LIST_1)),
#                           transforms.To2d()
#                       ]),
#                       online_transform=transforms.ToTensor(),
#                       label_transform=transforms.Compose([
#                           transforms.Select('valence'),
#                           transforms.Binary(5.0),
#                       ]),
#                       io_path='.torcheeg/datasets_8channels_full_TSCeption')


k_fold = KFoldGroupbyTrial(split_path='./examples_pipeline/split_8c_tsception',
                           shuffle=True,
                           random_state=42,
                           n_splits=2)

train_dataset, test_dataset = next(k_fold.split(dataset))

# train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42,split_path='./examples_pipeline/split_8c_tt')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = Net(num_classes=2, in_channels=128, grid_size=(4, 6))
optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-4)

epochs = 50
best_val_acc = 0.0
for t in range(epochs):
    train_loss = train(train_loader, model, loss_fn, optimizer)
    val_acc, val_loss = valid(test_loader, model, loss_fn)
    # save the best model based on val_acc
    if val_acc > best_val_acc:
        best_val_acc = val_acc

torch.save(model.state_dict(),
                   f'./examples_vanilla_torch/model.pt')
model.load_state_dict(torch.load(f'./examples_vanilla_torch/model.pt'))
test_acc, test_loss = valid(test_loader, model, loss_fn)

# log the test result
logger.info(
    f"Test Error : \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f}"
)
test_accs.append(test_acc)
test_losses.append(test_loss)
# trainer = ClassifierTrainer(model=model,
#                                 num_classes=2,
#                                 lr=1e-4,
#                                 weight_decay=1e-4,
#                                 accelerator="gpu")

# trainer.fit(train_loader,
#                 test_loader,
#                 max_epochs=5,
#                 default_root_dir=f'./examples_pipeline/model/{0}',
#                 callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
#                 enable_progress_bar=True,
#                 enable_model_summary=True,
#                 limit_val_batches=0.0)
# score = trainer.test(test_loader,
#                          enable_progress_bar=True,
#                          enable_model_summary=True)[0]
# print(f'test accuracy: {score["test_accuracy"]:.4f}')



# for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
#     x,y=next(iter(train_loader))
#     print(x.shape)

    
#     model = Net(num_classes=2, in_channels=128, grid_size=(4, 6))
#     #model = Net2(num_class=2)
#     # model = TSCeption(num_classes=2,
#     #               num_electrodes=8,
#     #               sampling_rate=128,
#     #               num_T=15,
#     #               num_S=15,
#     #               hid_channels=32,
#     #               dropout=0.5)

#     trainer = ClassifierTrainer(model=model,
#                                 num_classes=2,
#                                 lr=1e-4,
#                                 weight_decay=1e-4,
#                                 accelerator="gpu")
#     trainer.fit(train_loader,
#                 val_loader,
#                 max_epochs=20,
#                 default_root_dir=f'./examples_pipeline/model/{i}',
#                 callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
#                 enable_progress_bar=True,
#                 enable_model_summary=True,
#                 limit_val_batches=0.0)
#     score = trainer.test(val_loader,
#                          enable_progress_bar=True,
#                          enable_model_summary=True)[0]
#     print(f'Fold {i} test accuracy: {score["test_accuracy"]:.4f}')