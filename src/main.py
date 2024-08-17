"""
Расчет классификации графов с помощью GCN из 3 свёрточных слоев со skip connections
"""

import logging
import random

import hydra

import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric import set_debug
from torch_geometric.loader.dataloader import DataLoader

from dataset import HCPDataset
from pytorchtools import seed_everything

set_debug(True)
log = logging.getLogger(__name__)


@hydra.main(version_base='1.3', config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Фиксируем seed для воспроизводимости
    seed_everything(cfg.model.random_state)
    torch.manual_seed(cfg.model.random_state)
    random.seed(cfg.model.random_state)
    np.random.seed(cfg.model.random_state)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Готовим обучающую и тестовую выборки
    train_dataset = HCPDataset(cfg=cfg).shuffle()
    test_dataset = HCPDataset(cfg=cfg, is_test=True).shuffle()

    print('End!')


if __name__ == "__main__":
    main()
