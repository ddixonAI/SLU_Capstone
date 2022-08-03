import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utils.define_unet import build_unet
from utils.define_fct import build_fct
from utils.define_fct import init_weights
from utils.loss import DiceLoss, DiceBCELoss
from utils.prepare_data import seeding, create_dir, epoch_time, DriveDataset, DriveDatasetFCT

def train(model, loader, optimizer, loss_fn, device, model_choice):

    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        if model_choice == 'fct':
            y_pred = model(x)
            loss = loss_fn(y_pred[2], y)
        else:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device, model_choice):

    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            if model_choice == 'fct':
                loss = loss_fn(y_pred[2], y)
            else:
                loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def train_model_unet():
    """ Seeding """
    seeding(31)

    """ Directories """
    create_dir("files")

    """ Load dataset """
    train_x = sorted(glob("C:/Users/Derek/Documents/SLU_Capstone/new_data/train/image/*"))
    train_y = sorted(glob("C:/Users/Derek/Documents/SLU_Capstone/new_data/train/mask/*"))

    valid_x = sorted(glob("C:/Users/Derek/Documents/SLU_Capstone/new_data/test/image/*"))
    valid_y = sorted(glob("C:/Users/Derek/Documents/SLU_Capstone/new_data/test/mask/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 512
    W = 512
    size = (H, W)
    batch_size = 2
    num_epochs = 100
    lr = 1e-4
    checkpoint_path = "files/checkpoint_unet.pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device('cuda')   ## GTX 3090 24GB
    print(f'Device used: {device}')
    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")
    patience = 5
    overfit_counter = 0

    for epoch in range(num_epochs):

        if overfit_counter <= patience:

            start_time = time.time()

            train_loss = train(model, train_loader, optimizer, loss_fn, device, 'unet')
            valid_loss = evaluate(model, valid_loader, loss_fn, device)

            """ Saving the model """
            if valid_loss < best_valid_loss:
                data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
                print(data_str)

                best_valid_loss = valid_loss
                torch.save(model.state_dict(), checkpoint_path)
                overfit_counter = 0
            else:
                overfit_counter += 1
                print(f'No improvement in loss. Patience count now at {overfit_counter}.')

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
            data_str += f'\tTrain Loss: {train_loss:.3f}\n'
            data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
            print(data_str)

        else:
            print(f"Patience of {patience} has been reached on Epoch {epoch}. Ending training.")        
            break    


def train_model_fct():
    """ Seeding """
    seeding(31)

    """ Directories """
    create_dir("files")

    """ Load dataset """
    train_x = sorted(glob("C:/Users/Derek/Documents/SLU_Capstone/new_data/train/image/*"))
    train_y = sorted(glob("C:/Users/Derek/Documents/SLU_Capstone/new_data/train/mask/*"))

    valid_x = sorted(glob("C:/Users/Derek/Documents/SLU_Capstone/new_data/test/image/*"))
    valid_y = sorted(glob("C:/Users/Derek/Documents/SLU_Capstone/new_data/test/mask/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 512
    W = 512
    size = (H, W)
    batch_size = 1
    num_epochs = 200
    lr = 1e-3
    checkpoint_path = "files/checkpoint.pth"

    """ Dataset and loader """
    train_dataset = DriveDatasetFCT(train_x, train_y)
    valid_dataset = DriveDatasetFCT(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    device = torch.device('cuda')   ## GTX 3090 24GB
    print(f'Device used: {device}')
    model = build_fct()
    model = model.to(device)
    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = nn.BCELoss() # not the same loss function from the original implementation

    """ Training the model """
    best_valid_loss = float("inf")
    patience = 5
    overfit_counter = 0

    for epoch in range(num_epochs):

        if overfit_counter <= patience:

            start_time = time.time()

            train_loss = train(model, train_loader, optimizer, loss_fn, device, 'fct')
            valid_loss = evaluate(model, valid_loader, loss_fn, device, 'fct')

            """ Saving the model """
            if valid_loss < best_valid_loss:
                data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
                print(data_str)

                best_valid_loss = valid_loss
                torch.save(model.state_dict(), checkpoint_path)
                overfit_counter = 0
            else:
                overfit_counter += 1
                print(f'No improvement in loss. Patience count now at {overfit_counter}.')

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
            data_str += f'\tTrain Loss: {train_loss:.3f}\n'
            data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
            print(data_str)

        else:
            print(f"Patience of {patience} has been reached on Epoch {epoch}. Ending training.")        
            break    