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

# set_debug(True)
log = logging.getLogger(__name__)


@hydra.main(version_base='1.3', config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    # Фиксируем seed для воспроизводимости
    seed_everything(cfg.models.random_state)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ## Для ОТЛАДКИ
    # device = torch.device('cpu')
    # Готовим обучающую, валидационную и тестовую выборки
    train_dataset = HCPDataset(cfg=cfg, kind='train').shuffle()
    valid_dataset = HCPDataset(cfg=cfg, kind='valid').shuffle()
    test_dataset = HCPDataset(cfg=cfg, kind='test').shuffle()

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.models.model.params['batch_size'],  # len(train_dataset),
                              shuffle=False)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=len(valid_dataset),
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
                         start_factor=1,
                         end_factor=0.1,
                         total_iters=cfg.models['max_epochs'])
    mlflow.set_tracking_uri(uri=cfg.mlflow['tracking_uri'])  # type: ignore
    mlflow.set_experiment(experiment_name=cfg.mlflow['experiment_name'])  # type: ignore
    # run_name = f'hp_hidden_channels={cfg.models.model.params.hidden_channels}'
    run_name = 'Best params'
    with mlflow.start_run(run_name=run_name):  # type: ignore
        valid_loss, valid_acc, valid_auc = train_valid_model(cfg=cfg,
                                                             model=model,
                                                             device=device,
                                                             train_loader=train_loader,
                                                             valid_loader=valid_loader,
                                                             criterion=criterion,
                                                             optimizer=optimizer,
                                                             pooling_type=cfg.models.model.params['pooling_type'],
                                                             scheduler=scheduler,
                                                             mlflow_object=mlflow)
        # для инфо
        test_model(model=model,
                   device=device,
                   test_loader=test_loader,
                   criterion=criterion,
                   pooling_type=cfg.models.model.params['pooling_type'],
                   state_path=None,
                   mlflow_object=mlflow)
    print('End!')
    # Возвращаем величину, которую будем оптимизируем при подборе гиперпараметров
    return valid_acc  # Средняя доля верных ответов по валидационному датасету (максимизируем!!)


if __name__ == "__main__":
    main()
