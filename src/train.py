# coding: utf-8
""" training procedure """
from typing import Any, Optional

import numpy as np
import torch  # type: ignore
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score  # type: ignore
from torch import sigmoid

from mlflow_helpers import log_params_from_omegaconf_dict
from pytorchtools import EarlyStopping  # type: ignore


def train_one_epoch(model, device, train_loader, criterion, optimizer) -> float:  # type: ignore
    """Make training for one epoch using all batches

    Parameters
    ----------
    model
    device
    train_loader
    criterion
    optimizer

    Returns
    -------
    float
        loss for this epoch
    """
    running_loss = 0.0
    step = 0
    model.train()
    # for data_idx in train_loader.dataset._indices:
    #     data = train_loader.dataset.get(data_idx)
    for data in train_loader:
        data.to(device)  # Use GPU
        # ####### DEBUG for pytorch Dataset, not for PYG ########
        # real_graph_indices = data.graph_index.detach().to('cpu').numpy() - data.ptr.detach().to('cpu').numpy()[:-1]
        # ######################
        # Reset gradients
        optimizer.zero_grad()
        out = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight,
                    batch=data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y.float().reshape(-1, 1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        running_loss += loss.item()
        step += 1

    return running_loss / step


@torch.no_grad()
def test_one_epoch(model, device, test_loader, criterion, calc_conf_matrix=False) -> tuple[float, ...]:
    model.eval()
    running_loss = 0.0
    step = 0
    batch_auc = []
    batch_accuracy = []
    all_trues = []
    all_preds = []

    for data in test_loader:
        data.to(device)  # Use GPU
        # ####### DEBUG for pytorch Dataset, not for PYG ########
        # real_graph_indices = data.graph_index.detach().to('cpu').numpy() - data.ptr.detach().to('cpu').numpy()[:-1]
        # ######################
        out = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, batch=data.batch)
        loss = criterion(out, data.y.float().reshape(-1, 1))
        running_loss += loss.item()
        step += 1
        pred = sigmoid(out)
        y_true = data.y.detach().to('cpu').numpy()
        y_pred = pred.detach().to('cpu').numpy()
        batch_accuracy.append(accuracy_score(y_true, y_pred.round()))
        batch_auc.append(roc_auc_score(y_true, y_pred))
        if calc_conf_matrix:
            all_trues.extend(y_true)
            all_preds.extend(y_pred)

    if calc_conf_matrix:
        conf_matrix = confusion_matrix(np.asarray(all_trues), np.asarray(all_preds).round())
        return running_loss / step, np.mean(batch_accuracy), np.mean(batch_auc), conf_matrix
    return running_loss / step, np.mean(batch_auc), np.mean(batch_accuracy)


def reset_model(model):
    # Reinitialize layers
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train_valid_model(cfg: DictConfig,
                      model, device,
                      train_loader,
                      valid_loader,
                      criterion,
                      optimizer,
                      scheduler,
                      mlflow_object: Optional[Any] = None) -> float:
    print("Start training ...")
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # Количество эпох для обучения. В случае раннего останова - максимальное количество эпох
    n_epochs = cfg.models.max_epochs
    # инициализируем как None, если ранний останов не используется
    early_stopping = None
    # Если используем ранний останов, то инициализируем объект
    if cfg.models.model.early_stopping:
        early_stopping = EarlyStopping(**cfg.models.stoping)
    if mlflow_object is not None:
        # log param
        log_params_from_omegaconf_dict(cfg)
    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        train_loss = train_one_epoch(model=model, device=device, train_loader=train_loader, criterion=criterion,
                                     optimizer=optimizer)
        # record training loss
        avg_train_losses.append(train_loss)

        ######################
        # validate the model #
        ######################
        valid_loss, valid_acc, valid_auc, valid_conf_matrix = test_one_epoch(model=model,
                                                                             device=device,
                                                                             test_loader=valid_loader,
                                                                             criterion=criterion,
                                                                             calc_conf_matrix=True)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        if mlflow_object is not None:
            mlflow_object.log_metric('train loss', train_loss, step=epoch)
            mlflow_object.log_metric('valid loss', valid_loss, step=epoch)
            mlflow_object.log_metric('valid accuracy', valid_acc, step=epoch)
            mlflow_object.log_metric('valid roc_auc', valid_auc, step=epoch)

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] '
                     f'train_loss: {train_loss:.5f} '
                     f'valid_loss: {valid_loss:.5f} ')

        print(print_msg)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        if cfg.models.model.early_stopping:
            patience_counter = early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                if mlflow_object is not None:
                    mlflow_object.log_param('Learned Epochs', epoch - patience_counter)
                break

        if mlflow_object is not None:
            mlflow_object.log_metric('learning rate', scheduler.get_last_lr()[0], step=epoch)
        scheduler.step()

    # load the last checkpoint with the best model
    if cfg.models.model.early_stopping:
        model.load_state_dict(torch.load(cfg.training.stoping['path']))
    train_loss, train_acc, train_auc, train_conf_matrix = test_one_epoch(model=model,
                                                                             device=device,
                                                                             test_loader=train_loader,
                                                                             criterion=criterion,
                                                                             calc_conf_matrix=True)
    valid_loss, valid_acc, valid_auc, valid_conf_matrix = test_one_epoch(model=model,
                                                                         device=device,
                                                                         test_loader=valid_loader,
                                                                         criterion=criterion,
                                                                         calc_conf_matrix=True)
    print(f'Final TRAIN loss: {train_loss}, accuracy: {train_acc}, AUC ROC: {train_auc}\n'
          f'Final VALID loss: {valid_loss}, accuracy: {valid_acc}, AUC ROC: {valid_auc}\n')

    return train_loss


def test_model(model, device, test_loader, criterion, accuracy, cfg, mlflow_object: Optional[Any] = None):
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(cfg.training.stoping['path']))
    t_loss_test, t_mean_accr_test, conf_matrix = test_one_epoch(model, device, test_loader, criterion,
                                                                accuracy=accuracy, calc_conf_matrix=True, cfg=cfg)
    print(f'Final TEST loss: {t_loss_test},'
          f' accuracy: {t_mean_accr_test}\n'
          f' Conf Matrix: {conf_matrix}')

    # sourcery skip: no-conditionals-in-tests
    if mlflow_object is not None:
        mlflow_object.log_metric('Test loss', t_loss_test)
        mlflow_object.log_metric('Test accuracy', t_mean_accr_test)
        mlflow_object.log_param('Conf Matrix', ', '.join(map(str, conf_matrix.ravel().tolist())))
