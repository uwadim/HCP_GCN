# Файл для тестирования модели на публичном графовом датасете MUTAG

import logging

import hydra

import mlflow
import torch

from omegaconf import DictConfig
from torch.optim.lr_scheduler import LinearLR
from torch_geometric import set_debug
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.datasets import TUDataset

from dataset import HCPDataset
from pytorchtools import seed_everything

from models import GCN, SkipGCN
from train import train_valid_model, test_model, reset_model
set_debug(True)
log = logging.getLogger(__name__)


@hydra.main(version_base='1.3', config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    # Фиксируем seed для воспроизводимости
    seed_everything(cfg.models.random_state)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Для ТЕСТА
    # device = torch.device('cpu')

    # Загружаем датасет для проверки модели
    dataset_path = './data/test/TUDataset'
    dataset = dataset = TUDataset(root=dataset_path, name='MUTAG', cleaned=True)
    #dataset.download()
    dataset = dataset.shuffle()
    # Готовим обучающую и тестовую выборки
    train_loader = DataLoader(dataset[:0.9], cfg.models.model.params['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset[0.9:], cfg.models.model.params['batch_size'])

    # Инициализируем модель с параметрами из соответствующего конфигурационного файла
    model = eval(f'{cfg.models.model.name}(model_params={cfg.models.model.params},'
                 f'num_node_features={dataset.num_node_features})')
    reset_model(model)  # Реинициализируем модель
    # Загружаем модель на GPU
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.models.model['learning_rate'])
    scheduler = LinearLR(optimizer,
                         total_iters=cfg.models['max_epochs'])
    with mlflow.start_run():  # type: ignore
        valid_loss = train_valid_model(cfg=cfg,
                          model=model,
                          device=device,
                          train_loader=train_loader,
                          valid_loader=test_loader,
                          criterion=criterion,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          mlflow_object=mlflow)
    print('End!')
    return valid_loss


if __name__ == "__main__":
    main()
