"""
Main function for training and evaluating a graph classification model using GCN with 3 convolutional layers and skip connections.

This script performs the following steps:
1. Loads the configuration from a YAML file.
2. Sets the random seed for reproducibility.
3. Prepares the training, validation, and test datasets.
4. Initializes the model with parameters from the configuration.
5. Trains the model using the training dataset and validates it using the validation dataset.
6. Evaluates the model on the test dataset.
7. Logs the results using MLflow.
8. Returns the validation accuracy as the metric to optimize during hyperparameter tuning.

Parameters:
    cfg (DictConfig): Configuration object containing all necessary parameters for the experiment.

Returns:
    float: Validation accuracy, which is the metric to optimize during hyperparameter tuning.
"""

import logging
import random

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CyclicLR, LinearLR
from torch_geometric import set_debug
from torch_geometric.loader.dataloader import DataLoader

import mlflow
from dataset import HCPDataset
from models import GCN, SkipGCN
from pytorchtools import seed_everything
from train import reset_model, test_model, train_valid_model

# Enable debug mode in PyTorch Geometric (optional)
# set_debug(True)
log = logging.getLogger(__name__)


@hydra.main(version_base='1.3', config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Main function to execute the training and evaluation pipeline.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing all necessary parameters.

    Returns
    -------
    float:
        Validation accuracy, which is the metric to optimize during hyperparameter tuning.
    """

    # If bootstrap is enabled, update model parameters with the best parameters from a previous experiment
    if 'make_bootstrap' in cfg.keys() and cfg.make_bootstrap:
        config_df = pd.read_csv('configs/dataset_best_params.csv')
        cfg.models.model.params['hidden_channels'] = \
            config_df.query(f'dataset == "{cfg.data.dataset.root}"'
                            f' and dataset_type == "{cfg.data.dataset_type}"')['hidden_channels'].to_list()[0]
        cfg.models.model.params['dropout'] = \
            config_df.query(f'dataset == "{cfg.data.dataset.root}"'
                            f' and dataset_type == "{cfg.data.dataset_type}"')['dropout'].to_list()[0]

    # Fix the seed for reproducibility
    seed_everything(cfg.models.random_state)

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ## For DEBUGGING (use CPU)
    # device = torch.device('cpu')

    # Prepare the training, validation, and test datasets
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

    # Initialize the model with parameters from the configuration file
    model = eval(f'{cfg.models.model.name}(model_params={cfg.models.model.params},'
                 f'num_node_features={train_dataset.num_node_features})')
    reset_model(model)  # Reinitialize the model

    # Move the model to the specified device (GPU or CPU)
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.models.model['learning_rate'])

    # Define the learning rate scheduler
    scheduler = LinearLR(optimizer,
                         start_factor=1,
                         end_factor=0.1,
                         total_iters=cfg.models['max_epochs'])
    # scheduler = CyclicLR(optimizer,
    #                      base_lr=0.0005,
    #                      max_lr=0.05,
    #                      step_size_up=10,
    #                      mode="triangular2")

    # Set up MLflow for experiment tracking
    mlflow.set_tracking_uri(uri=cfg.mlflow['tracking_uri'])  # type: ignore
    mlflow.set_experiment(experiment_name=cfg.mlflow['experiment_name'])  # type: ignore

    # Define the run name for MLflow
    run_name = f'seed={cfg.models.random_state}'

    with mlflow.start_run(run_name=run_name):  # type: ignore
        # Train and validate the model
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

        # Prepare a dictionary for plotting (optional)
        plot_dict = None
        if cfg.mlflow['save_adjacency_matrices']:
            plot_dict = {
                'fname': cfg.mlflow['experiment_name'],
                'processed_dir': f'{cfg.data["root_path"]}/processed',
                'seed': cfg.models.random_state,
                'max_elements': cfg.mlflow['max_elements'],
                'palette_name': cfg.mlflow['palette_name'],
                'dataset_type': cfg.data['dataset_type'],
                'result_path': f'results/{cfg.mlflow["experiment_name"]}.txt',
            }

        # Test the model on the test dataset
        test_model(model=model,
                   device=device,
                   test_loader=test_loader,
                   criterion=criterion,
                   pooling_type=cfg.models.model.params['pooling_type'],
                   state_path=None,
                   mlflow_object=mlflow,
                   plot_dict=plot_dict)

    print('End!')

    # Return the validation accuracy as the metric to optimize during hyperparameter tuning
    return valid_acc  # Mean accuracy on the validation dataset (maximize!!)


if __name__ == "__main__":
    main()