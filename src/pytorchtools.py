import os
import random

import numpy as np
import torch
import torch_geometric


def seed_everything(seed: int) -> None:
    """
    Устанавливает начальное значение генераторов случайных чисел
     для всех используемых библиотек.

    Параметры
    ----------
    seed : int
        Начальное значение для генерации случайных чисел.

    Возвращает
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # При запуске на GPU c CuDNN эти 2 параметра должны быть установлены
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Для hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)