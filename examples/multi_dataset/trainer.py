from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn import MSELoss
from torch.optim import Adam
import logging
import pickle
from datetime import datetime
from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import *
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
from typing import Optional


class ForceMSE(_Loss):
    def __init__(
        self,
        force_kwd: str = FORCE_KEY,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ) -> None:
        super(ForceMSE, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction
        )

        self.force_kwd = force_kwd

    def forward(self, data: AtomicData) -> torch.Tensor:
        if self.force_kwd not in data.out:
            raise RuntimeError(
                f"target property {self.force_kwd} has not been computed in data.out {list(data.out.keys())}"
            )
        if self.force_kwd not in data:
            raise RuntimeError(
                f"target property {self.force_kwd} has no reference in data {list(data.keys())}"
            )

        return F.mse_loss(
            data.out[self.force_kwd],
            data[self.force_kwd],
            reduction=self.reduction,
        )


def simple_train_loop(
    model,
    optimizer,
    loss_func,
    train_loader=None,
    val_loader=None,
    starting_epoch=0,
    num_epochs=30,
    model_save_freq=None,
    model_name="MyModel",
    print_freq=10000,
    save_dir="./",
    device=torch.device("cpu"),
    cummulative_batch=None,
):
    model.to(device)
    if not train_loader:
        raise RuntimeError("Must include train loader")
    if not val_loader:
        raise RuntimeError("Must include test loader")
    if cummulative_batch == None:
        cummulative_batch = train_loader.batch_size
        accumulation_size = 1
    else:
        if cummulative_batch % train_loader.batch_size != 0:
            raise RuntimeError(
                "Cummulative batch size must be evenly divisible by train loader batch size."
            )

        accumulation_size = int(cummulative_batch / train_loader.batch_size)

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Training Started: {}".format(dt_string))

    epochal_train_losses = []
    epochal_test_losses = []

    for epoch in range(0 + starting_epoch, num_epochs + starting_epoch):
        print("Starting epoch {}".format(epoch))
        # Train
        model.train()
        optimizer.zero_grad()
        running_train_loss = 0.00

        num_train_batches = 0
        for i, data in enumerate(train_loader):
            data.to(device)
            data = model(data)
            data.out.update(**data.out[model.name])
            loss = loss_func(data) / accumulation_size
            loss.backward()
            if ((i + 1) % accumulation_size == 0) or (
                (i + 1) == len(train_loader)
            ):
                optimizer.step()
                optimizer.zero_grad()

            loss_numpy = loss.detach().cpu().numpy()
            if i % print_freq == 0:
                print(
                    "Batch = {}, Train loss:".format(i),
                    loss_numpy * accumulation_size,
                )
            running_train_loss += loss_numpy * accumulation_size
            num_train_batches += 1

        train_loss = running_train_loss / num_train_batches
        epochal_train_losses.append(train_loss)
        del data, loss
        # Test
        model.eval()
        optimizer.zero_grad()
        running_test_loss = 0.00

        num_test_batches = 0
        for i, data in enumerate(val_loader):
            data.to(device)
            data = model(data)
            data.out.update(**data.out[model.name])
            loss = loss_func(data)
            loss_numpy = loss.detach().cpu().numpy()
            if i % print_freq == 0:
                print("Batch = {}, Test loss:".format(i), loss_numpy)
            running_test_loss += loss_numpy
            num_test_batches += 1

        test_loss = running_test_loss / num_test_batches
        epochal_test_losses.append(test_loss)
        print(
            "Epoch {}: Train {} \t Test {}".format(
                epoch + starting_epoch, train_loss, test_loss
            )
        )
        logging.info(
            "Epoch {}: Train {} \t Test {}".format(
                epoch + starting_epoch, train_loss, test_loss
            )
        )
        print("Saving epoch {} ...".format(epoch))
        with open(
            save_dir + model_name + "_state_dict_epoch_{}.pkl".format(epoch),
            "wb",
        ) as modelfile:
            pickle.dump(model.to(torch.device("cpu")).state_dict(), modelfile)
        model.to(device)
        np.save(
            save_dir + "{}_epochal_train_losses.npy".format(model_name),
            epochal_train_losses,
        )
        np.save(
            save_dir + "{}_epochal_test_losses.npy".format(model_name),
            epochal_test_losses,
        )

    with open(
        save_dir + model_name + "_state_dict_epoch_{}.pkl".format(epoch), "wb"
    ) as modelfile:
        pickle.dump(model.to(torch.device("cpu")).state_dict(), modelfile)
    np.save(
        save_dir + "{}_epochal_train_losses.npy".format(model_name),
        epochal_train_losses,
    )
    np.save(
        save_dir + "{}_epochal_test_losses.npy".format(model_name),
        epochal_test_losses,
    )

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Training Finished: {}".format(dt_string))


def multi_gpu_train_loop(
    model,
    optimizer,
    loss_func,
    train_loader=None,
    val_loader=None,
    starting_epoch=0,
    num_epochs=30,
    print_freq=10000,
    device=torch.device("cpu"),
    model_name="my_model",
    save_dir="./",
):
    """Simple training loop. Assumes DataParallel object for model
    Parameters
    ----------
    model:
        Model to train
    optimizer:
        torch.optim optimizer
    train_loader:
        Dataloader for the training data
    val_loader:
        Dataloader for the valing data
    num_epochs:
        Number of epochs to train
    print_freq:
        Frequency for printing training output to stdoue
    device:
        torch.device which the model will be moved on and off of
        (between saving model state dicts)
    model_name:
        model name
    save_dir:
        Directory in which results are saved
    """

    model.cuda()
    if not train_loader:
        raise RuntimeError("Must include train loader")
    if not val_loader:
        raise RuntimeError("Must include val loader")

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Training Started: {}".format(dt_string))

    epochal_train_losses = []
    epochal_val_losses = []

    for epoch in range(0 + starting_epoch, num_epochs + starting_epoch):
        print("Starting epoch {}".format(epoch))

        # == Train ==#

        model.train()
        optimizer.zero_grad()
        running_train_loss = 0.00

        num_train_batches = 0
        for i, data in enumerate(train_loader):
            data.cuda()
            data = model(data)
            data.out.update(**data.out[model.module.name])
            loss = loss_func(data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_numpy = loss.detach().cpu().numpy()
            if i % print_freq == 0:
                print("Batch = {}, Train loss:".format(i), loss_numpy)
            running_train_loss += loss_numpy
            num_train_batches += 1

        train_loss = running_train_loss / num_train_batches
        epochal_train_losses.append(train_loss)

        del loss, data
        torch.cuda.empty_cache()

        # == Val ==#

        model.eval()
        optimizer.zero_grad()
        running_val_loss = 0.00

        num_val_batches = 0
        for i, data in enumerate(val_loader):
            data.cuda()
            data = model(data)
            data.out.update(**data.out[model.module.name])
            loss = loss_func(data)
            loss_numpy = loss.detach().cpu().numpy()
            if i % print_freq == 0:
                print("Batch = {}, Test loss:".format(i), loss_numpy)
            running_val_loss += loss_numpy
            num_val_batches += 1

        val_loss = running_val_loss / num_val_batches
        epochal_val_losses.append(val_loss)

        print(
            "Epoch {}: Train {} \t Val {}".format(
                epoch + starting_epoch, train_loss, val_loss
            )
        )
        logging.info(
            "Epoch {}: Train {} \t Val {}".format(
                epoch + starting_epoch, train_loss, val_loss
            )
        )
        print("Saving epoch {} ...".format(epoch))
        with open(
            save_dir
            + model_name
            + "_state_dict_epoch_{}_val_loss_{}.pkl".format(epoch, val_loss),
            "wb",
        ) as modelfile:
            pickle.dump(
                model.module.to(torch.device("cpu")).state_dict(), modelfile
            )
        with open(
            save_dir + model_name + "_optim_dict_epoch_{}.pkl".format(epoch),
            "wb",
        ) as optimfile:
            pickle.dump(optimizer.state_dict(), optimfile)

        model.cuda()

    with open(
        save_dir + model_name + "_state_dict_epoch_final.pkl", "wb"
    ) as modelfile:
        pickle.dump(
            model.module.to(torch.device("cpu")).state_dict(), modelfile
        )
    np.save(
        save_dir + "{}_epochal_train_losses.npy".format(model_name),
        epochal_train_losses,
    )
    np.save(
        save_dir + "{}_epochal_val_losses.npy".format(model_name),
        epochal_val_losses,
    )

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Training Finished: {}".format(dt_string))
