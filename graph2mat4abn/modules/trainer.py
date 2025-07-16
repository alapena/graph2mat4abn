
import time
from graph2mat4abn.tools.import_utils import save_to_yaml
from graph2mat4abn.tools.plot import plot_error_matrices_big, plot_error_matrices_small
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from torch_geometric.loader import DataLoader
from plotly.subplots import make_subplots
from copy import copy

from graph2mat4abn.tools.tools import optimizer_to, read_structures_paths, write_structures_paths
from graph2mat4abn.modules.memory_monitor import MemoryMonitor

class Trainer:
    def __init__(self, model, config, train_dataset, val_dataset, loss_fn, optimizer, device='cpu', lr_scheduler=None, live_plot=True, live_plot_freq=1, live_plot_matrix = False, live_plot_matrix_freq = 100, results_dir=None, checkpoint_freq=30, batch_size=1, processor=None, model_checkpoint=None):
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
        self.model = model.to(self.device)
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.history = model_checkpoint["history"] if model_checkpoint is not None else None
        self.results_dir = Path(results_dir)
        self.checkpoint_freq = checkpoint_freq
        self.model_checkpoint = model_checkpoint

        # Live plotting setup
        self.live_plot = live_plot
        self.live_plot_freq = live_plot_freq
        self.live_plot_matrix = live_plot_matrix
        self.live_plot_matrix_freq = live_plot_matrix_freq
        self.processor = processor

    def train_epoch(self, dataloader):
        """Run one epoch of training"""
        self.model.train()

        total_loss = 0.0
        total_edge_loss = 0.0
        total_node_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # # Get enviroment description
            # print("n_batches: ", len(batch))
            # print("n_atoms: ", len(batch.point_types))
            # enviroment_description = self.environment_descriptor(batch)
            # enviroment_description["node_feats"] = enviroment_description["node_feats"].detach()
            # print("Enviroment description: ", enviroment_description["node_feats"].shape)

            # Model forward pass
            # model_predictions = self.model(data=batch, node_feats=enviroment_description["node_feats"])
            model_predictions = self.model(data=batch)
            # print(model_predictions.keys())

            # Compute the loss
            loss, stats = self.loss_fn(
                nodes_pred=model_predictions["node_labels"],
                nodes_ref=batch.point_labels,
                edges_pred=model_predictions["edge_labels"],
                edges_ref=batch.edge_labels,
            )
            total_loss += loss*10**6
            total_edge_loss += stats["node_rmse"]**2 *10**6 # Squared because it returns the root.
            total_node_loss += stats["edge_rmse"]**2 *10**6

            # Compute gradients
            loss.backward()

            # Update weights
            optimizer_to(self.optimizer, self.device)
            self.optimizer.step()
            
            if self.lr_scheduler is not None and self.config["scheduler"]["type"] == "CyclicLR":
                self.lr_scheduler.step()
            # elif self.config["scheduler"]["type"] == "OneCycleLR" and self.epoch < self.config["scheduler"]["OneCycleLR"].get("epochs_cycle"):
            #     self.lr_scheduler.step()

            num_batches += 1

        # Average losses
        avg_loss = total_loss / num_batches
        avg_edge_loss = total_edge_loss / num_batches
        avg_node_loss = total_node_loss / num_batches

        self.history["train_loss"].append(avg_loss.item())
        self.history['train_edge_loss'].append(avg_edge_loss.item())
        self.history['train_node_loss'].append(avg_node_loss.item())


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
            # enviroment_description = self.environment_descriptor(batch)

            with torch.no_grad():
                # Model forward pass
                model_predictions = self.model(data=batch)

                # Compute the loss
                loss, stats = self.loss_fn(
                    nodes_pred=model_predictions["node_labels"],
                    nodes_ref=batch.point_labels,
                    edges_pred=model_predictions["edge_labels"],
                    edges_ref=batch.edge_labels,
                )
                total_loss += loss*10**6
                total_edge_loss += stats["node_rmse"]**2*10**6
                total_node_loss += stats["edge_rmse"]**2*10**6

                num_batches += 1

        # Average losses
        avg_loss = total_loss / num_batches
        avg_edge_loss = total_edge_loss / num_batches
        avg_node_loss = total_node_loss / num_batches

        self.history["val_loss"].append(avg_loss.item())
        self.history['val_edge_loss'].append(avg_edge_loss.item())
        self.history['val_node_loss'].append(avg_node_loss.item())

    # def check_for_plateau(self, val_loss, epoch):
    #     """Check if training has plateaued and adjust learning rate if needed"""
    #     if epoch > 0 and abs(val_loss - self.val_losses[-1]) < 1e-4:
    #         self.plateau_counter += 1

    #         if self.plateau_counter >= self.plateau_patience:
    #             # Reduce learning rate when plateau is detected
    #             current_lr = self.optimizer.param_groups[0]['lr']
    #             if current_lr > self.min_lr:
    #                 new_lr = current_lr * 0.5
    #                 for param_group in self.optimizer.param_groups:
    #                     param_group['lr'] = new_lr
    #                 print(f"Plateau detected! Reducing learning rate from {current_lr} to {new_lr}")
    #                 self.plateau_counter = 0
    #                 return True
    #     else:
    #         self.plateau_counter = 0
    #     return False

    def train(self, num_epochs):

        # Track the time of training.
        start_time = time.time()

        # Create history if the model is not pretrained
        if self.model_checkpoint is None:
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
            starting_epoch = 0

            self.best_train_loss = float('inf')
            self.best_train_edge_loss = float('inf')
            self.best_train_node_loss = float('inf') 

            self.best_val_loss = float('inf')
            self.best_val_edge_loss = float('inf')
            self.best_val_node_loss = float('inf') 

            self.execution_id = 1

            self.resumed_in_epochs = [-1]
        else:
            # Load from checkpoint
            starting_epoch = self.model_checkpoint["epoch"] + 1
            self.best_train_loss = min(self.model_checkpoint["history"]["train_loss"])
            self.best_val_loss = min(self.model_checkpoint["history"]["val_loss"])

            self.best_train_edge_loss = self.model_checkpoint["history"]["train_edge_loss"][-1]
            self.best_train_node_loss = self.model_checkpoint["history"]["train_node_loss"][-1]

            self.best_val_edge_loss = self.model_checkpoint["history"]["val_edge_loss"][-1]
            self.best_val_node_loss = self.model_checkpoint["history"]["val_node_loss"][-1]

            self.execution_id = self.model_checkpoint["execution_id"] + 1

            self.resumed_in_epochs = self.model_checkpoint["resumed_in_epochs"]
            self.resumed_in_epochs.append(-1)
            


        # Create results directory
        if self.results_dir is not None:
            self.results_dir.mkdir(exist_ok=True, parents=True)

            # Create a separate path for periodic checkpoints
            checkpoint_dir = Path(self.results_dir / "checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)

            # Save config
            save_to_yaml(self.config, self.config["results_dir"] +"/"+ "config.yaml")

            # Save structures paths
            current_dataset_dir = Path(self.config.get("results_dir"))
            current_dataset_dir = current_dataset_dir / "dataset"
            current_dataset_dir.mkdir(exist_ok=True)

            train_structures = []
            val_structures = []
            for data in self.train_dataset:
                train_structures.append(str(data.metadata["path"]))
            for data in self.val_dataset:
                val_structures.append(str(data.metadata["path"]))

            # Check if the training dataset is the same as in the loaded model
            # If first training
            if self.execution_id == 1:
                write_structures_paths(train_structures, (current_dataset_dir / f"train_dataset.txt"))
                write_structures_paths(val_structures, (current_dataset_dir / f"val_dataset.txt"))

            # If we are loading a model
            else:
                previous_dataset_dir = Path(*Path(self.config.get("trained_model_path")).parts[:2]) / "dataset"
                previous_train_structures = read_structures_paths(str(previous_dataset_dir / f"train_dataset.txt"))
                previous_val_structures = read_structures_paths(str(previous_dataset_dir / f"val_dataset.txt"))

                # Check training_dataset
                if set(train_structures) != set(previous_train_structures): # We use set() to compare because order does not matter
                    if not self.config["debug_mode"]:
                        raise ValueError("The training dataset is different from the loaded one!")
                    else:
                        print("Training dataset is not the same but we are in DEBUG MODE so don't worry :).")
                else:
                    print("Training dataset is the same as the loaded one :)")
                    write_structures_paths(train_structures, (current_dataset_dir / f"train_dataset.txt"))

                # Check val_dataset
                if set(val_structures) != set(previous_val_structures): # We use set() to compare because order does not matter
                    if not self.config["debug_mode"]:
                        raise ValueError("The validation dataset is different from the loaded one!")
                    else:
                        print("Validation dataset is not the same but we are in DEBUG MODE so don't worry :).")
                else:
                    print("Validation dataset is the same as the loaded one :)")
                    write_structures_paths(val_structures, (current_dataset_dir / f"val_dataset.txt"))


        # === Training loop ===
        # Create dataloaders
        train_dataloader = DataLoader(self.train_dataset, self.batch_size)
        val_dataloader = DataLoader(self.val_dataset, self.batch_size)

        # Initialize memory monitor
        memory_monitor = MemoryMonitor()

        for epoch in range(starting_epoch, num_epochs):
            epoch_t0 = time.time()
            print("="*30, f"Epoch {epoch+1}/{num_epochs}", "="*30)

            memory_monitor.start_epoch()

            # Training phase
            self.epoch = epoch
            self.train_epoch(train_dataloader)

            # Validation phase
            self.validate(val_dataloader)

            # Store current learning rate
            self.lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(self.lr)
            self.resumed_in_epochs[-1] = epoch

            # Update learning rate scheduler if provided
            if self.lr_scheduler is not None and self.config["scheduler"]["type"] == "CosineAnnealingWarmRestarts":
                self.lr_scheduler.step()
            elif self.lr_scheduler is not None and self.config["scheduler"]["type"] == "ReduceLROnPlateau":
                self.lr_scheduler.step(self.history["train_loss"][-1])
            elif self.lr_scheduler is None:
                # Do sth here manually.
                pass



            # Save best model based on training loss
            if self.results_dir is not None and self.history["train_loss"][-1] < self.best_train_loss:
                model_path = Path(self.results_dir / f"train_best_model.tar")
                self.save_model(epoch, model_path)
                self.best_train_loss = self.history["train_loss"][-1]
                print(f"New best model saved to {model_path}")

            else:

                if self.results_dir is not None and self.history["train_edge_loss"][-1] < self.best_train_edge_loss:
                    model_path = Path(self.results_dir / "train_edge_best_model.tar")
                    self.save_model(epoch, model_path)
                    self.best_train_edge_loss = self.history["train_edge_loss"][-1]
                    print(f"New best model saved to {model_path}")

                if self.results_dir is not None and self.history["train_node_loss"][-1] < self.best_train_node_loss:
                    model_path = Path(self.results_dir / "train_node_best_model.tar")
                    self.save_model(epoch, model_path)
                    self.best_train_node_loss = self.history["train_node_loss"][-1]
                    print(f"New best model saved to {model_path}")



            # Save best model based on validation loss
            if self.results_dir is not None and self.history["val_loss"][-1] < self.best_val_loss:
                model_path = Path(self.results_dir / "val_best_model.tar")
                self.save_model(epoch, model_path)
                self.best_val_loss = self.history["val_loss"][-1]
                print(f"New best model saved to {model_path}")

            else:

                if self.results_dir is not None and self.history["val_edge_loss"][-1] < self.best_val_edge_loss:
                    model_path = Path(self.results_dir / "val_edge_best_model.tar")
                    self.save_model(epoch, model_path)
                    self.best_val_edge_loss = self.history["val_edge_loss"][-1]
                    print(f"New best model saved to {model_path}")

                if self.results_dir is not None and self.history["val_node_loss"][-1] < self.best_val_node_loss:
                    model_path = Path(self.results_dir / "val_node_best_model.tar")
                    self.save_model(epoch, model_path)
                    self.best_val_node_loss = self.history["val_node_loss"][-1]
                    print(f"New best model saved to {model_path}")



            # Save periodic checkpoint
            if self.results_dir is not None and epoch % self.checkpoint_freq == 0:
                checkpoint_path = Path(checkpoint_dir / f"model_epoch_{epoch}.tar")
                self.save_model(epoch, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")



            # Update live plots
            if (self.live_plot and epoch % self.live_plot_freq == 0) or epoch == num_epochs - 1:
                self.update_loss_plots()


            # Plot hamiltonians while training
            if (self.live_plot_matrix and epoch % self.live_plot_matrix_freq == 0 and epoch != 0) or epoch == num_epochs - 1:
                self.plot_hamiltonians(epoch)


            # Compute time
            epoch_time = time.time() - epoch_t0
            elapsed_time = time.time() - start_time
            
            self.history["epoch_times"].append(epoch_time)
            self.history["elapsed_time"].append(elapsed_time)

            # Print progress
            print(f"Train stats. \t Total loss: {self.history["train_loss"][-1]:.4f} (edge loss: {self.history['train_edge_loss'][-1]:.4f}, node loss: {self.history['train_node_loss'][-1]:.4f})")
            print(f"Validation stats. \t Total loss: {self.history["val_loss"][-1]:.4f} (edge loss: {self.history['val_edge_loss'][-1]:.4f}, node loss: {self.history['val_node_loss'][-1]:.4f})")
            print(f"Learning rate: {self.history["learning_rate"][-1]:.1e}")
            print(f"Epoch duration: {self.history['epoch_times'][-1]:.2f} s")
            print(f"Total elapsed time: {self.history['elapsed_time'][-1]:.2f} s")

            memory_monitor.end_epoch()
            memory_monitor.plot_memory_usage(Path(self.results_dir / "memory_usage.png"))

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
            'history': self.history,
            'execution_id': self.execution_id,
            'resumed_in_epochs': self.resumed_in_epochs,
        }, path)

        

    def update_loss_plots(self, verbose=False):
        def detach_list(tensor_list):
            return [t.detach().item() if isinstance(t, torch.Tensor) else float(t) for t in tensor_list]
        
        # Prepare data
        df = pd.DataFrame(
            np.array([
                detach_list(self.history["train_loss"]),
                detach_list(self.history["val_loss"]),
                detach_list(self.history["train_edge_loss"]),
                detach_list(self.history["val_edge_loss"]),
                detach_list(self.history["train_node_loss"]),
                detach_list(self.history["val_node_loss"]),
                detach_list(self.history["learning_rate"]),
            ]).T,
            columns=["Train total", "Val total", "Train edge", "Val edge", "Train node", "Val node", "Learning rate"],
        )

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add loss traces (primary y-axis)
        loss_colors = {
            "Train total": "blue",
            "Val total": "red",
            "Train edge": "blue",
            "Val edge": "red",
            "Train node": "blue",
            "Val node": "red"
        }
        loss_dashes = {
            "Train total": "solid",
            "Val total": "solid",
            "Train edge": "dash",
            "Val edge": "dash",
            "Train node": "dot",
            "Val node": "dot"
        }
        
        for col in df.columns[:-1]:  # All columns except Learning rate
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=col,
                    line=dict(color=loss_colors[col], dash=loss_dashes[col]),
                    legendgroup=col.split()[1] if col.split()[0] in ["Train", "Val"] else col
                ),
                secondary_y=False
            )
        
        # Add learning rate trace (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Learning rate"],
                name="Learning rate",
                line=dict(color="lightgreen", dash="solid"),
                legendgroup="Learning rate"
            ),
            secondary_y=True
        )

        # Add a flag (vertical line) in each epoch where the training was resumed.
        for x_value in self.resumed_in_epochs:
            if x_value == -1:
                continue

            fig.add_vline(
                x=x_value,
                line_dash="solid",
                line_color="lightgray"
            )
        
        # Set axis titles and layout
        ylim_up = int(np.percentile(df.drop(columns=["Learning rate"]).to_numpy().flatten(), 95)) + 10
        fig.update_layout(
            title=f"Loss curves. Training executed {self.execution_id} times.",
            xaxis_title="Epoch number",
            yaxis=dict(
                title="Loss (eV²)",
                showgrid=True,
                # type="log",
                range=[-5, ylim_up]
            ),
            yaxis2=dict(
                title="Learning rate",
                showgrid=False,
                type="log",
                side="right",
                tickformat=".0e", 
                dtick=1,
            ),
            grid=dict(xside="bottom", yside="left"),
            legend=dict(
                x=1.1,  
                xanchor="left",  
                y=1.0,  
                yanchor="top"  
            ),
            margin=dict(r=150)
        )
        
        # Save outputs
        plot_path = self.results_dir / "plot_loss.html"
        fig.write_html(str(plot_path))
        # fig.write_image(str(self.results_dir / "plot_loss.png"))

        del fig
        
        if verbose:
            print(f"Loss plot saved to {plot_path}")

    def plot_hamiltonians(self, epoch):
        results_directory = self.results_dir / "plots_during_training"
        results_directory.mkdir(exist_ok=True)

        # Set the (max) number of each structure type that you want to plot
        n = self.config["trainer"].get("live_plot_matrices_num", 1)
        n_plots_each = {
            2: n,
            3: n,
            8: n,
            32: n,
            64: n,
        }
        n_atoms_list = list(n_plots_each.keys())

        # Dataloaders are needed for some reason; kword "batch" is needed at some point
        train_dataloader = DataLoader(self.train_dataset, 1)
        val_dataloader = DataLoader(self.val_dataset, 1)

        dataloaders = [train_dataloader, val_dataloader]
        for dataloader_id, dataloader in enumerate(dataloaders):
            dataloader_type = "training" if dataloader_id == 0 else "validation"
            print(f"Plotting {dataloader_type} dataset...")

            n_plotted = np.zeros([2, len(n_atoms_list)], dtype=np.int16)
            n_plotted[0] += n_atoms_list
            for j, data in enumerate(dataloader):
                data = data.to(self.device)
                n_atoms = data.num_nodes
                col_idx = np.where(n_plotted[0] == n_atoms)[0]

                # Continue or break if already plotted the required number of plots
                if all(n_plotted[1][i] == n_plots_each[n_plotted[0][i]] for i in range(n_plotted.shape[1])):
                    break
                if n_plotted[1][col_idx] == n_plots_each[n_atoms]:
                    continue 
                
                # Generate prediction
                with torch.no_grad():
                    self.model.eval()
                    model_predictions = self.model(data=data)
                    loss, _ = self.loss_fn(
                        nodes_pred=model_predictions["node_labels"],
                        nodes_ref=data.point_labels,
                        edges_pred=model_predictions["edge_labels"],
                        edges_ref=data.edge_labels,
                    )

                    pred_matrix = self.processor.matrix_from_data(
                        data,
                        predictions={"node_labels": model_predictions["node_labels"], "edge_labels": model_predictions["edge_labels"]},
                    )[0].todense()

                    # Save true matrix
                    true_matrix = self.processor.matrix_from_data(
                        data,
                    )[0].todense()

                # Plot
                title = f"Results of sample {j} of {dataloader_type} dataset (seed {self.config["dataset"]["seed"]}). There are {n_atoms} in the unit cell."
                predicted_matrix_text = f"Saved training loss at epoch {epoch}:     {self.history["train_loss"][-1]:.2f} eV²\nMSE evaluation:     {loss.item():.2f} eV²" if dataloader_type == "training" else f"Saved training loss at epoch {epoch}:     {self.history["val_loss"][-1]:.2f} eV²\nMSE evaluation:     {loss.item():.2f} eV²"
                if self.config["trainer"]["matrix"] == "hamiltonian" or self.config["trainer"]["matrix"] == "overlap":
                    if n_atoms <= 32:
                        plot_error_matrices_big(
                            true_matrix, pred_matrix,
                            matrix_label=self.config["trainer"]["matrix"],
                            figure_title=title,
                            predicted_matrix_text=predicted_matrix_text,
                            filepath = Path(results_directory / f"{dataloader_type}_{n_atoms}atoms_sample{j}_epoch{epoch}.html")
                        )
                    else:
                        plot_error_matrices_big(
                            true_matrix, pred_matrix,
                            matrix_label=self.config["trainer"]["matrix"],
                            figure_title=title,
                            predicted_matrix_text=predicted_matrix_text,
                            filepath = Path(results_directory / f"{dataloader_type}_{n_atoms}atoms_sample{j}_epoch{epoch}.png")
                        )
                elif self.config["trainer"]["matrix"] == "tim":
                    plot_error_matrices_small(
                        true_matrix, pred_matrix,
                        matrix_label="Transfer Integral Matrix",
                        figure_title=title,
                        predicted_matrix_text=predicted_matrix_text,
                        filepath = Path(results_directory / f"{dataloader_type}_{n_atoms}atoms_sample{j}_epoch{epoch}.html")
                    )
                else:
                    raise ValueError(f"Matrix type {self.config['trainer']['matrix']} is not supported for plotting.")

                print(f"Plotted sample {j}")

                
                n_plotted[1][col_idx] += 1

        print("Hamiltonian plots generated!")