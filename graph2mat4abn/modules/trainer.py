
import time
import torch
import pandas as pd
import numpy as np

from pathlib import Path
from torch_geometric.loader import DataLoader

class Trainer:
    def __init__(self, environment_descriptor, model, config, train_dataset, val_dataset, loss_fn, optimizer, device='cpu', lr_scheduler=None, live_plot=True, live_plot_freq=1, live_plot_matrix = False, live_plot_matrix_freq = 100, history=None, results_dir=None, checkpoint_freq=30, batch_size=1):
        """_summary_

        Args:
            model (_type_): _description_
            config (_type_): _description_
            train_dataset (_type_): _description_
            val_dataset (_type_): _description_
            loss_fn (_type_): _description_
            optimizer (_type_): _description_
            device (str, optional): _description_. Defaults to 'cpu'.
            lr_scheduler (_type_, optional): _description_. Defaults to None.
            live_plot (bool, optional): _description_. Defaults to True.
            live_plot_freq (int, optional): _description_. Defaults to 1.
            live_plot_matrix (bool, optional): _description_. Defaults to False.
            live_plot_matrix_freq (int, optional): _description_. Defaults to 100.
            history (_type_, optional): _description_. Defaults to None.
            results_dir (_type_, optional): _description_. Defaults to None.
            checkpoint_freq (int, optional): _description_. Defaults to 30.
            batch_size (int, optional): _description_. Defaults to 1.
        """

        self.device = device
        self.environment_descriptor = environment_descriptor.to(self.device)
        self.model = model.to(self.device)
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.history = history
        self.results_dir = Path(results_dir)
        self.checkpoint_freq = checkpoint_freq
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')

        # Live plotting setup
        self.live_plot = live_plot
        self.live_plot_freq = live_plot_freq
        self.live_plot_matrix = live_plot_matrix
        self.live_plot_matrix_freq = live_plot_matrix_freq

    def train_epoch(self, dataloader):
        """Run one epoch of training"""
        self.model.train()

        total_loss = 0.0
        total_edge_loss = 0.0
        total_node_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)

            # Get enviroment description
            enviroment_description = self.environment_descriptor(batch)
            enviroment_description["node_feats"] = enviroment_description["node_feats"].detach()

            # Model forward pass
            model_predictions = self.model(data=batch, node_feats=enviroment_description["node_feats"])

            # Compute the loss
            loss, stats = self.loss_fn(
                nodes_pred=model_predictions[0],
                nodes_ref=batch.point_labels,
                edges_pred=model_predictions[1],
                edges_ref=batch.edge_labels,
            )
            total_loss += loss
            total_edge_loss += stats["node_rmse"]**2 # Squared because it returns the root.
            total_node_loss += stats["edge_rmse"]**2

            # Compute gradients
            loss.backward()

            # Update weights
            self.optimizer.step()

            num_batches += 1

        # Average losses
        avg_loss = total_loss / num_batches
        avg_edge_loss = total_edge_loss / num_batches
        avg_node_loss = total_node_loss / num_batches

        self.history["train_loss"].append(avg_loss)
        self.history['train_edge_loss'].append(avg_edge_loss)
        self.history['train_node_loss'].append(avg_node_loss)


    def validate(self, dataloader):
        """Run validation of the model"""
        self.model.eval()

        total_loss = 0.0
        total_edge_loss = 0.0
        total_node_loss = 0.0
        num_batches = 0

        
        for batch in dataloader:
            batch = batch.to(self.device)

            # Get enviroment description
            enviroment_description = self.environment_descriptor(batch)

            with torch.no_grad():
                # Model forward pass
                model_predictions = self.model(data=batch, node_feats=enviroment_description["node_feats"])

                # Compute the loss
                loss, stats = self.loss_fn(
                    nodes_pred=model_predictions[0],
                    nodes_ref=batch.point_labels,
                    edges_pred=model_predictions[1],
                    edges_ref=batch.edge_labels,
                )
                total_loss += loss
                total_edge_loss += stats["node_rmse"]**2
                total_node_loss += stats["edge_rmse"]**2

                num_batches += 1

        # Average losses
        avg_loss = total_loss / num_batches
        avg_edge_loss = total_edge_loss / num_batches
        avg_node_loss = total_node_loss / num_batches

        self.history["val_loss"].append(avg_loss)
        self.history['val_edge_loss'].append(avg_edge_loss)
        self.history['val_node_loss'].append(avg_node_loss)

    def check_for_plateau(self, val_loss, epoch):
        """Check if training has plateaued and adjust learning rate if needed"""
        if epoch > 0 and abs(val_loss - self.val_losses[-1]) < 1e-4:
            self.plateau_counter += 1

            if self.plateau_counter >= self.plateau_patience:
                # Reduce learning rate when plateau is detected
                current_lr = self.optimizer.param_groups[0]['lr']
                if current_lr > self.min_lr:
                    new_lr = current_lr * 0.5
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"Plateau detected! Reducing learning rate from {current_lr} to {new_lr}")
                    self.plateau_counter = 0
                    return True
        else:
            self.plateau_counter = 0
        return False

    def train(self, num_epochs):
        """
        Train the model for specified number of epochs with learning rate scheduling

        Args:
            num_epochs: Number of training epochs
            filename: Path to save the best model (optional)

        Returns:
            model: Trained model
            history (Dict): History of training/validation losses
        """
        # Track the time of training.
        start_time = time.time()

        # Create results directory
        if self.results_dir is not None:
            self.results_dir.mkdir(exist_ok=True, parents=True)

            # Create a separate path for periodic checkpoints
            checkpoint_dir = Path(self.results_dir / "checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)

        # Create history if the model is not pretrained
        if self.history is None:
            self.history = {
                # Total losses
                'train_loss': [],
                'val_loss': [],

                # Component losses
                'train_edge_loss': [],
                'train_node_loss': [],
                'val_edge_loss': [],
                'val_node_loss': [],

                'learning_rate': [],

                "epoch_times": [],
                "elapsed_time": [],
            }


        # === Training loop ===
        # Create dataloaders
        train_dataloader = DataLoader(self.train_dataset, self.batch_size)
        val_dataloader = DataLoader(self.val_dataset, self.batch_size)

        for epoch in range(num_epochs):
            epoch_t0 = time.time()

            # Training phase
            self.train_epoch(train_dataloader)

            # Validation phase
            self.validate(val_dataloader)

            # Store current learning rate
            lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(lr)

            # Update learning rate scheduler if provided
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Compute time
            epoch_time = time.time() - epoch_t0
            elapsed_time = time.time() - start_time
            
            self.history["epoch_times"].append(epoch_time)
            self.history["elapsed_time"].append(elapsed_time)

            # Print progress
            print("="*30, f"Epoch {epoch+1}/{num_epochs}", "="*30)
            print(f"Train stats. \t Total loss: {self.history["train_loss"][-1]:.4f} (edge loss: {self.history['train_edge_loss'][-1]:.4f}, node loss: {self.history['train_node_loss'][-1]:.4f})")
            print(f"Validation stats. \t Total loss: {self.history["val_loss"][-1]:.4f} (edge loss: {self.history['val_edge_loss'][-1]:.4f}, node loss: {self.history['val_node_loss'][-1]:.4f})")
            print(f"Learning rate: {self.history["learning_rate"][-1]:.4f}")
            print(f"Epoch duration: {self.history['epoch_times'][-1]:.2f} s")
            print(f"Total elapsed time: {self.history['elapsed_time'][-1]:.2f} s")

            # Update live plots
            if (self.live_plot and epoch % self.live_plot_freq == 0) or epoch == num_epochs - 1:
                self.update_loss_plots()


            # === Plot hamiltonians while training ===
            # TODO: Implement the matrix plots
            # if (self.live_plot_matrix and epoch % self.live_plot_matrix_freq == 0) or epoch == num_epochs - 1:

            #     dataset_subsets = [train_dataset_subset, validation_dataset_subset]

            #     # Create results directory
            #     results_directory = self.training_info_path + "/" + "hamiltonian_plots_during_training"
            #     create_directory(results_directory)

            #     # === Plot all hamiltonians in the subset ===
            #     for j, dataset in enumerate(dataset_subsets):
            #         dataset_type = ["training", "validation"]
            #         for i, sample in enumerate(dataset):
            #             sample = sample.clone()
            #             sample = sample.to(self.device)
            #             loss, predicted_h, original_h = generate_hamiltonian_prediction(self.model, sample, self.loss_fn)

            #             # === Plot ===
            #             if dataset_type[j] == "training":
            #                 idx_sample = train_subset_indices[i]
            #             elif dataset_type[j] == "validation":
            #                 idx_sample = validation_subset_indices[i]
            #             else:
            #                 raise ValueError("Unknown dataset type")

            #             n_atoms = len(sample["x"])
            #             filepath = f"{results_directory}/{dataset_type[j]}_atoms_{n_atoms}_sample_{idx_sample}_epoch_{epoch}.png"
            #             title = f"Results of sample {idx_sample} from {dataset_type[j]} dataset (seed 4). There are {n_atoms} in the unit cell."
            #             predicted_matrix_text = f"Saved training loss at epoch {epoch}:     {self.history["train_loss"][-1]:.2f} eV²·100\nMSE evaluation:     {loss.item():.2f} eV²·100"
            #             plot_error_matrices(original_h.cpu().numpy(),
            #                                 predicted_h.cpu().numpy() / 100,
            #                                 filepath=filepath,
            #                                 matrix_label="Hamiltonian",
            #                                 figure_title=title,
            #                                 n_atoms=n_atoms,
            #                                 predicted_matrix_text=predicted_matrix_text
            #                                 )
            #             filepath = f"{results_directory}/{dataset_type[j]}_atoms_{n_atoms}_sample_{idx_sample}_epoch_{epoch}.html"
            #             predicted_matrix_text = f"Saved training loss at epoch {epoch}:     {self.history["train_loss"][-1]:.2f} eV²·100<br>MSE evaluation:     {loss.item():.2f} eV²·100"
            #             plot_error_matrices_interactive(original_h.cpu().numpy(),
            #                                 predicted_h.cpu().numpy() / 100,
            #                                 filepath=filepath,
            #                                 matrix_label="Hamiltonian",
            #                                 figure_title=title,
            #                                 n_atoms=n_atoms,
            #                                 predicted_matrix_text=predicted_matrix_text
            #                                 )
            #     print("Hamiltonian plots generated!")

            # Save best model based on training loss
            if self.results_dir is not None and self.history["train_loss"][-1] < self.best_train_loss:
                model_path = Path(self.results_dir / "train_best_model.tar")
                self.save_model(epoch, model_path)
                self.best_train_loss = self.history["train_loss"][-1]
                print(f"New best model saved to {model_path}")

            # Save best model based on validation loss
            if self.results_dir is not None and self.history["val_loss"][-1] < self.best_val_loss:
                model_path = Path(self.results_dir / "val_best_model.tar")
                self.save_model(epoch, model_path)
                self.best_val_loss = self.history["val_loss"][-1]
                print(f"New best model saved to {model_path}")

            # Save periodic checkpoint (every 50 epochs)
            if self.results_dir is not None and epoch % self.checkpoint_freq == 0:
                checkpoint_path = Path(checkpoint_dir / f"model_epoch_{epoch}.pt")
                self.save_model(epoch, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

        # ====== TRAINING LOOP FINISHED ======

        # Force a final model save
        if self.results_dir is not None:
            model_path = Path(self.results_dir / "last_model.tar")
            self.save_model(epoch, model_path)
            print(f"Last model saved to {model_path}")
        
        # Force a final plots update
        self.update_loss_plots()

        return None
    
    def save_model(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'train_loss': self.history["train_loss"][-1],
            'val_loss': self.history["val_loss"][-1],
        }, path)

    def update_loss_plots(self, verbose=False):
        def detach_list(tensor_list):
            return [t.detach().item() if isinstance(t, torch.Tensor) else float(t) for t in tensor_list]
        df = pd.DataFrame(
            np.array([
                detach_list(self.history["train_loss"]),
                detach_list(self.history["train_edge_loss"]),
                detach_list(self.history["train_node_loss"]),
                detach_list(self.history["val_loss"]),
                detach_list(self.history["val_edge_loss"]),
                detach_list(self.history["val_node_loss"]),
            ]).T,
            columns=["Train loss", "Train edge", "Train node", "Val loss", "Val edge", "Val node"],
        )

        fig = df.plot(backend="plotly")
        fig.update_layout(
            yaxis_type="log",
            yaxis_showgrid=True,
            xaxis_showgrid=True,
            yaxis_title="Loss (eV²)",
            xaxis_title="Epoch number",
            title="Loss curves",
        )
        for trace in fig.data:
            if trace["legendgroup"] != "Train loss" and trace["legendgroup"] != "Val loss":
                trace.line.update(dash='dash')
        # Save to HTML
        plot_path = self.results_dir / "plot_loss.html"
        fig.write_html(str(plot_path))
        fig.write_image(str(self.results_dir / "plot_loss.png"))
        if verbose:
            print(f"Loss plot saved to {plot_path}")

