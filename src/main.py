"""
Расчет классификации графов с помощью GCN из 3 свёрточных слоев со skip connections
"""

import logging
import random
import re
import shutil

import hydra

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy.sparse import csgraph
from sklearn.metrics import roc_auc_score
from torch import sigmoid
from torch_geometric import set_debug
from torch_geometric.loader.dataloader import DataLoader

from dataset import HCPDataset
# from model import GCN, SkipGCN
# from pytorchtools import EarlyStopping
# from samplers import StratifiedSampler, StratifiedKFoldSampler
# import helpers
#
set_debug(True)
log = logging.getLogger(__name__)



@hydra.main(version_base='1.3', config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Готовим обучающую и тестовую выборки
    train_dataset = HCPDataset(cfg=cfg).shuffle()
    test_dataset = HCPDataset(cfg=cfg, is_test=True).shuffle()
    # Фиксируем seed для воспроизводимости
    torch.manual_seed(cfg.model.random_seed)
    random.seed(cfg.model.random_state)
    np.random.seed(cfg.model.random_state)
    print('End!')


if __name__ == "__main__":
    main()
