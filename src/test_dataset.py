# File for testing the model on the public graph dataset MUTAG

import logging

import hydra
import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LinearLR
from torch_geometric import set_debug
from torch_geometric.datasets import TUDataset
from torch_geometric.loader.dataloader import DataLoader

import mlflow
from dataset import HCPDataset
from models import GCN, SkipGCN
from pytorchtools import seed_everything
from train import reset_model, test_model, train_valid_model

set_debug(True)
log = logging.getLogger(__name__)


@hydra.main(version_base='1.3', config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Main function to test the model on the MUTAG dataset.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing model and training parameters.

    Returns
    -------
    float
        Validation loss after training.
    """
    # Fix the seed for reproducibility
    seed_everything(cfg.models.random_state)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # For TESTING
    # device = torch.device('cpu')

    # Load the dataset for model validation
    dataset_path = './data/test/TUDataset'
    dataset = TUDataset(root=dataset_path, name='MUTAG', cleaned=True)
    dataset = dataset.shuffle()
    # Prepare the training and test sets
    train_loader = DataLoader(dataset[:0.9], cfg.models.model.params['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset[0.9:], cfg.models.model.params['batch_size'])

    # Initialize the model with parameters from the corresponding configuration file
    model = eval(f'{cfg.models.model.name}(model_params={cfg.models.model.params},'
                 f'num_node_features={dataset.num_node_features})')
    reset_model(model)  # Reinitialize the model
    # Load the model onto the GPU
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
