# coding: utf-8

import random
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch


class ConfigStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")


def main():
    config = dict(
        epochs=600,
        batch_size=2048,
        learning_rate=0.008,
        weight_decay=1e-5,
        dropout=0.05,
        shuffle=True,
        nprocs_filter=True,
        nprocs="4_16_48_64_144_240",
        test_size=0.2,
        split_seed=42,
        random_seed=1234
    )


    MODEL_FILENAME = "SmoothL1Loss_fixed_Adamax_0.2_testSize_new_StandardScaler_2048_batch_0.008_lr_filtered_nprocs_0.05_dropout_pytorch_v1.12.1.tar"
    MODEL_DIR = rf"/home/thes1067/models/blue_waters"
    MODEL_PATH = Path(MODEL_DIR, MODEL_FILENAME)

    DATASET_DIR = r"/home/thes1067/data/blue_waters_dataset"
    DATASET_NAME = "blue_waters_posix_no_outliers_4_16_48_64_144_240_nprocs"
    DATASET_PATH = Path(DATASET_DIR, DATASET_NAME).with_suffix(".csv")

    # ## Load the data
    df_blue_waters_posix = pd.read_csv(DATASET_PATH)


    # ### Separate bandwidth from input features
    bandwidth_df = df_blue_waters_posix.pop('bandwidth')

    # ### Fix seeds for reproducibility
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    torch.manual_seed(config["random_seed"])
    torch.cuda.manual_seed_all(config["random_seed"])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = ConfigStruct(**config)

    # ### Split the data
    X_train, X_test, y_train, y_test = train_test_split(df_blue_waters_posix,
                                                        bandwidth_df,
                                                        test_size=config.test_size,
                                                        random_state=config.split_seed,
                                                        stratify=df_blue_waters_posix["nprocs"] if config.stratified_split else None)

    # ### Scale the input features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    tensor_X_train = torch.Tensor(X_train_scaled).to(device)
    tensor_y_train = torch.Tensor(y_train.values).view(-1, 1).to(
        device)  # Transform to a 2D array to avoid shape mismatch (gives errors)

    training_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    training_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=config.shuffle)

    X_test_scaled = scaler.transform(X_test)

    tensor_X_test = torch.Tensor(X_test_scaled).to(device)
    tensor_y_test = torch.Tensor(y_test.values).view(-1, 1).to(
        device)  # Transform to a 2D array to avoid shape mismatch (gives errors)

    test_dataset = TensorDataset(tensor_X_test, tensor_y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)

    # ### Train a fully-connected Multi-Layered Perceptron

    model = nn.Sequential(
        nn.Linear(97, 2048),
        nn.Dropout(p=config.dropout),
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.Dropout(p=config.dropout),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.Dropout(p=config.dropout),
        nn.ReLU(),
        nn.Linear(128, 1),
    ).to(device)

    # By default Pytorch returns avg loss per minibatch elements. But since the last batch
    # (both in training and test) does not have enough instances, sum all the loss across the batches
    # and then divide it by total number of elements in the the test set.
    loss_fn = nn.SmoothL1Loss(beta=config.smooth_l1_loss_beta, reduction="sum").to(device)

    optimizer = optim.Adamax(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    model_epoch = 0

    # Load previously trained state if available
    if Path(MODEL_PATH).is_file():
        print("Loading pretrained model...")

        checkpoint = torch.load(MODEL_PATH, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        model_epoch = checkpoint['epoch']

        print(f"Current epoch: {model_epoch}")

    model.train()

    def train():
        size = len(training_dataloader)
        for batch, (X, y) in enumerate(training_dataloader):
            y_pred = model(X)
            
            # Divide the summed loss by the number of elements in the current batch to get the average loss
            loss = loss_fn(y, y_pred) / len(X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss = loss.item()
                print(f"loss: {loss:>7f} [{batch:>5d}/{size:>5d}]")

        model.train()

    def test():
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in test_dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item() 

        # Divide the summed test loss by the number of elements in the whole test dataset to get the average loss
        test_loss /= len(test_dataloader.dataset)

        print(f"Avg loss: {test_loss:>8f} \n")

        return test_loss

    for epoch in range(model_epoch, config.epochs):

        print(f"Epoch {epoch + 1}\n-------------------------------")
        train()
        test_loss = test()
        print(test_loss)

        scheduler.step(test_loss)

        model_epoch = epoch

        torch.save({
            'epoch': model_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, MODEL_PATH)

if __name__ == '__main__':
    main()
