"""
Расчет классификации графов с помощью GCN из 3 свёрточных слоев со skip connections
"""

import logging
import random

import hydra

import mlflow
import numpy as np
import torch

from omegaconf import DictConfig
from torch.optim.lr_scheduler import LinearLR
from torch_geometric import set_debug
from torch_geometric.loader.dataloader import DataLoader

from dataset import HCPDataset
from pytorchtools import seed_everything

from models import GCN, SkipGCN
from train import train_valid_model, test_model, reset_model
set_debug(True)
log = logging.getLogger(__name__)


@hydra.main(version_base='1.3', config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Фиксируем seed для воспроизводимости
    seed_everything(cfg.models.random_state)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Для ТЕСТА
    # device = torch.device('cpu')
    # Готовим обучающую и тестовую выборки
    train_dataset = HCPDataset(cfg=cfg).shuffle()
    test_dataset = HCPDataset(cfg=cfg, is_test=True).shuffle()

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.models.model.params['batch_size'],  # len(train_dataset),
                              shuffle=False)
    test_loader = DataLoader(test_dataset,
                              batch_size=len(test_dataset),
                              shuffle=False)
    # Инициализируем модель с параметрами из соответствующего конфигурационного файла
    model = eval(f'{cfg.models.model.name}(model_params={cfg.models.model.params},'
                 f'num_node_features={train_dataset.num_node_features})')
    reset_model(model)  # Реинициализируем модель
    # Загружаем модель на GPU
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.models.model['learning_rate'])
    scheduler = LinearLR(optimizer,
                         total_iters=cfg.models['max_epochs'])
    with mlflow.start_run():  # type: ignore
        train_valid_model(cfg=cfg,
                          model=model,
                          device=device,
                          train_loader=train_loader,
                          valid_loader=test_loader,
                          criterion=criterion,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          mlflow_object=mlflow)
    print('End!')


if __name__ == "__main__":
    main()
