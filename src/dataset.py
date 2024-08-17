"""
Файл, формирующий датасеты для работы с графовыми нейронными сетями из пакета torch_geometric
"""
import os
import shutil
from pathlib import Path
from typing import Callable, Optional, List

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset


class HCPDataset(Dataset):
    def __init__(self,
                 cfg: DictConfig,
                 rebuild_processed: bool = False,
                 is_test: bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        """
        Инициализирует экземпляр класса HCPDataset.

        Parameters
        ----------
        cfg : DictConfig
            Объект конфигурации.
        rebuild_processed : bool, default=False
            Флаг, указывающий, нужно ли пересобрать обработанные данные.
            По умолчанию False.
        is_test : bool, default=False
            Флаг, указывающий, является ли набор данных тестовым.
            По умолчанию False.
        transform : Optional[Callable], default=None
            Функция преобразования данных.
            По умолчанию None.
        pre_transform : Optional[Callable], default=None
            Функция предварительного преобразования данных.
            По умолчанию None.
        pre_filter : (Optional[Callable], default=None
            Функция фильтрации данных.
            По умолчанию None.

        Returns
        -------
        None
        """
        self.cfg = cfg
        self.is_test = is_test
        # Проверяем, что есть каталоги для распаковки и создаем, если это необходимо
        self.root = self.cfg.data.root_path
        # Удаляем закешированные файлы, если требуется
        # или если тип кодирования в конфиргурационном файле не совпадает с типом кодирования в
        # распакованных файлах
        root_path = Path(self.root)
        proper_files = list(root_path.rglob(f'*{self.cfg.data.coding_type}*'))
        if root_path.exists() and (rebuild_processed or not proper_files):
            shutil.rmtree(root_path)

        super().__init__(self.root, transform, pre_transform, pre_filter)

    @property
    def raw_paths(self) -> List[Path]:
        """
        Возвращает список путей к сырым данным.

        Returns
        -------
        List[Path]
            Список путей к сырым данным, отсортированный для одинакового порядка при разных запусках.
        """
        if Path(self.raw_dir).exists():
            # Сортируем список, чтобы был одинаковый порядок при разных запусках
            return sorted(list(Path(self.raw_dir).glob(f'*{self.cfg.data.coding_type}*')))
        return []

    @property
    def raw_file_names(self):
        try:
            if Path(self.raw_dir).exists():
                return sorted([f.name for f in Path(self.raw_dir).glob(f'*{self.cfg.data.coding_type}*')])
            return []
        except AttributeError:
            return []

    @property
    def processed_file_names(self):
        """ return list of files should be in processed dir, if found - skip processing."""
        try:
            if Path(self.processed_dir).exists():
                if self.is_test:
                    return sorted([f.name for f in Path(self.processed_dir).glob('data_test*.pt')])
                return sorted([f.name for f in Path(self.processed_dir).glob('data_[!test]*.pt')])
            return []
        except AttributeError:
            return []

    def download(self) -> None:
        # Распаковываем архив, как есть d self.raw_dir
        # self.raw_dir создается в родительском классе из self.root
        shutil.unpack_archive(self.cfg.data.dataset.download_url, self.raw_dir)
        # переносим файлы, которые подходят по типу кодирования
        raw_path = Path(self.raw_dir)
        for file in raw_path.rglob(f'*{self.cfg.data.coding_type}*'):
            shutil.move(src=file, dst=raw_path)
        # Ищем и удаляем каталог с ненужными файлами
        for p in raw_path.iterdir():
            if p.is_dir():
                shutil.rmtree(p)

    def process(self):
        for fpath in self.raw_paths:
            # Разбиваем имя файла по символу "_"
            # в предположении, что имя файла должно иметь вид: [id]_[coding_type]_[label]
            # убираем пробелы в имени файла
            try_list = [s.strip() for s in fpath.stem.split('_')]
            label = self.cfg.data.dataset.labels[try_list[2]]
            graph_id = try_list[0]
            data = pd.read_csv(fpath, sep=self.cfg.data.sep)
            edge_indices = torch.tensor(data[self.cfg.data.edges_colnames].to_numpy(), dtype=torch.long)
            edge_weights = torch.tensor(data[self.cfg.data.weights_colname].to_numpy(), dtype=torch.float)
            num_of_nodes = data[self.cfg.data.edges_colnames[0]].nunique()
            # Указываем все единицы в качестве фичей на нодах
            node_features = torch.tensor(np.ones(num_of_nodes).reshape(-1, 1))
            data_to_save = Data(x=node_features,
                                edge_index=edge_indices,
                                # edge_attr=None,
                                edge_weight=edge_weights,
                                y=label)
            if self.is_test:
                if int(graph_id) in self.cfg.data.test_ids:
                    torch.save(data_to_save, os.path.join(self.processed_dir, f'data_test_{graph_id}.pt'))
                else:
                    continue
            else:
                if int(graph_id) in self.cfg.data.train_ids:
                    torch.save(data_to_save, os.path.join(self.processed_dir, f'data_{graph_id}.pt'))
                else:
                    continue

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, idx):
        return torch.load(
            os.path.join(self.processed_dir, self.processed_file_names[idx])
        )