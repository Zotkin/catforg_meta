import os
from typing import List, Dict, Union

import torch
import torch.nn as nn
import numpy as np

from meta.backbone import SimCLR
from meta.losses import NT_Xent
from meta.transforms import  transforms_test, transforms_train
from meta.data import DoubleAugmentedCIFAR10, SupervisedImageDataset


NUM_STAGES = 3
CHECKPOINT_DIR = "/home/leet/pytorch_model_checkpoints"
CIFAR10_PATH = "/home/leet/data/cifar10/"
NUM_EPOCHS = 5
BATCH_SIZE = 16
TEMPERATURE = 0.1
LR = 0.1
MOMENTUM = 0.9
USE_DATA_FROM_PREVIOUS_STAGES = False

def get_cifar_10(path: str) -> List[np.ndarray]:
    X_train = np.load(os.path.join(path, "x_train.npy"))
    X_test = np.load(os.path.join(path, "x_test.npy"))
    y_train = np.load(os.path.join(path, "y_train.npy"))
    y_test = np.load(os.path.join(path, "y_test.npy"))
    return [X_train, y_train, X_test, y_test]

def filter_dataset(X_train, y_train, X_test, y_test, proportions, use_data_from_prev_stages):
    if use_data_from_prev_stages:
        train_mask = []
        test_mask = []
        for label, fraction in proportions.items():
            train_idx = list(np.nonzero(y_train == label)[0])
            test_idx = list(np.nonzero(y_test == label)[0])

            num_train_examples_to_retain = int(len(train_idx)*fraction)
            train_mask.append(train_idx[:num_train_examples_to_retain])
            test_mask.append(test_idx)
    else:
        this_stage_classes = filter(lambda x: x[1] == 1, proportions.items())
        this_stage_classes = list(map(lambda x: x[0], this_stage_classes))
        train_mask = np.isin(y_train, this_stage_classes)
        test_mask = np.isin(y_test, this_stage_classes)

    return X_train[train_mask], y_train[train_mask], X_test[test_mask], y_test[test_mask]

def get_this_stage_proportions(stage: int) -> Dict[int, Union[float, int]]:
    d = {
        0: {0:1, 1:1, 2:1},
        1: {0:0.3, 1:0.3, 2:0.3, 3:1, 4:1, 5:1},
        2: {0:0.1, 1:0.1, 2:0.1, 3:0.3, 4:0.3, 5:0.3, 6:1, 7:1, 8:1}
    }
    return d[stage]

if __name__ == "__main__":

    DEVICE = torch.device('cpu')

    model = SimCLR()
    model.to(DEVICE)
    loss_function = NT_Xent(batch_size=BATCH_SIZE, temperature=TEMPERATURE, device=DEVICE)
    X_train, y_train, X_test, y_test = get_cifar_10(CIFAR10_PATH)
    for stage in range(NUM_STAGES):
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
        this_stage_proportions = get_this_stage_proportions(stage)
        X_train_stage, y_train_stage, X_test_stage, y_test_stage = filter_dataset(X_train, y_train, X_test, y_test, this_stage_proportions, USE_DATA_FROM_PREVIOUS_STAGES)

        train_dataset = DoubleAugmentedCIFAR10(X_train, transforms_train)
        val_dataset = DoubleAugmentedCIFAR10(X_test, transforms_train)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=8)

        best_val_loss = float('inf')
        for epoch in range(NUM_EPOCHS):

            running_loss = 0
            num_iter_to_report_progress = max(int(len(train_dataloader)*0.1), 1)
            for i, batch in enumerate(train_dataloader, 0):
                print(f"Iter {i}")
                x1, x2 = batch
                x1.to(DEVICE)
                x2.to(DEVICE)
                _, projection1 = model(x1)
                _, projection2 = model(x2)
                loss = loss_function(projection1, projection2)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % num_iter_to_report_progress-1 == 0:
                    print(f"Epoch {epoch} train_loss {loss}")

            running_loss = 0
            num_batches = 0
            for i, batch in enumerate(val_dataloader, 0):
                with torch.no_grad():
                    x1, x2 = batch
                    x1.to(DEVICE)
                    x2.to(DEVICE)

                    _, projection1 = model(x1)
                    _, projection2 = model(x2)
                    loss = loss_function(projection1, projection2)
                    num_batches += 1
                    running_loss += loss.item()

            print(f"Epoch {epoch}; val_loss: {running_loss/num_batches}")
            if loss/num_batches < best_val_loss:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"stage_{stage}_with_external_data_best.pth")
                print(f"val_loss {loss/num_batches} is better that previous {best_val_loss}. Saving checkpoint to {CHECKPOINT_DIR}")
                best_val_loss = loss
                torch.save(model.state_dict(), checkpoint_path)


