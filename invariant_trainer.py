import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import transformers
from transformers.optimization import Adafactor, get_scheduler
from torch.optim import AdamW
from transformers.trainer_callback import TrainerState
from transformers.utils import logging

from tqdm import tqdm

import wandb
import math
import os
import numpy as np
from itertools import cycle
import random
from torch.amp import autocast, GradScaler
from typing import Optional

logger = logging.get_logger(__name__)

def compute_moving_average(values, window_size=10):
        if len(values) < window_size:
            return sum(values) / max(len(values), 1)
        return sum(values[-window_size:]) / window_size

class InvariantTrainer(transformers.Trainer):

    def create_optimizer_and_scheduler(self, model, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        optimizer, lr_scheduler = None, None
        # if self.optimizer is None:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = Adafactor if self.args.adafactor else AdamW
        if self.args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
        optimizer_kwargs["lr"] = self.args.learning_rate
        
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )

        return optimizer, lr_scheduler
    

    def invariant_train(
            self,
            training_set,
            nb_steps: Optional[int] = None,
            nb_steps_heads_saving: Optional[int] = 0,
            resume_from_checkpoint: Optional[str] = None,
            num_train_epochs: Optional[int] = 1,
            nb_steps_model_saving: Optional[int] = 0,
            **kwargs,
    ):

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )

        min_train_set_size = min([len(data["train"]) for _, data in training_set.items()])
        
        # Calcul du nombre d'updates (steps) effectués durant une epoch pour chaque environnement
        steps_per_epoch = math.floor(
            min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size)
        )
        if nb_steps is not None:
            num_train_epochs = max(1, math.floor(nb_steps / steps_per_epoch))
            max_steps = nb_steps
        else:
            max_steps = steps_per_epoch * num_train_epochs

        dataloaders, head_optimizers, head_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(data_features["train"])

            head_optimizers[env_name], head_schedulers[env_name] = self.create_optimizer_and_scheduler(
                self.model.lm_heads[env_name],
                num_training_steps=max_steps
            )

        phi_optimizer, phi_scheduler = self.create_optimizer_and_scheduler(
            self.model.encoder,
            num_training_steps=max_steps
        )

        if self.args.n_gpu > 0:
            self.model.to(self.args.device)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        self.state.global_step = 0
        recent_losses = []

        iter_loaders = {env_name: cycle(dataloaders[env_name]) for env_name in training_set.keys()}

        for epoch in range(int(num_train_epochs)):
            print(f"\n===== ÉPOQUE {epoch + 1}/{num_train_epochs} =====")
            
            # Un round correspond à une itération sur tous les environnements
            for round_idx in tqdm(range(steps_per_epoch)):
                for env_name in random.sample(list(training_set.keys()), k=len(training_set)):
                    if self.state.global_step >= max_steps :
                        break
                    batch = next(iter_loaders[env_name])
                   
                    phi_optimizer.zero_grad()
                    head_optimizers[env_name].zero_grad()
                    self.model.train()

                    batch = {k: v.to(self.args.device) for k, v in batch.items() if torch.is_tensor(v)}
                    outputs = self.model(**batch)
                    loss = outputs.loss

                    loss.backward()

                    # Mise à jour des deux optimizers et mise à jour du scaler
                    optimizer.step()
                    head_optimizers[env_name].step()

                    # Mise à jour des schedulers
                    phi_scheduler.step()
                    head_schedulers[env_name].step()

                    self.state.global_step += 1
                    recent_losses.append(loss.item())
                    moving_avg_loss = compute_moving_average(recent_losses, window_size=20)

                    if self.is_world_process_zero() and self.state.global_step % nb_steps_model_saving == 0:
                        wandb.log({
                            "training/train_loss": loss.item(),
                            "training/train_loss_moving_avg": moving_avg_loss
                        }, step=self.state.global_step
                        )

                    if saving_heads and self.state.global_step % nb_steps_heads_saving == 0:
                        self.save_heads(self.state.global_step)
                    if saving_intermediary_models and self.state.global_step % nb_steps_model_saving == 0:
                        self.save_intermediary_model(self.state.global_step)
                    
                
        print("=== Entraînement du modèle terminé. Nombre total de rounds:", self.state.global_step/len(training_set.keys()))
 

    def invariant_train_games(
        self,
        training_set,
        nb_steps: Optional[int] = None,
        nb_steps_heads_saving: Optional[int] = 0,
        resume_from_checkpoint: Optional[str] = None,
        num_train_epochs: Optional[int] = 1,
        nb_steps_model_saving: Optional[int] = 0,
        update_phi_every_k: Optional[int] = 1,
        **kwargs,
    ):

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )

        # Determine steps per epoch and max steps
        min_train_size = min(len(data["train"]) for _, data in training_set.items())
        steps_per_epoch = math.floor(
            min_train_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size)
        )
        if nb_steps is not None:
            num_train_epochs = max(1, math.floor(nb_steps / steps_per_epoch))
            max_steps = nb_steps
        else:
            max_steps = steps_per_epoch * num_train_epochs

        # Prepare dataloaders, optimizers, schedulers
        dataloaders = {env: self.get_single_train_dataloader(data["train"]) for env, data in training_set.items()}
        head_optimizers = {}
        head_schedulers = {}
        for env in training_set:
            head_optimizers[env], head_schedulers[env] = self.create_optimizer_and_scheduler(
                self.model.lm_heads[env], num_training_steps=max_steps
            )
        phi_optimizer, phi_scheduler = self.create_optimizer_and_scheduler(
            self.model.encoder, num_training_steps=max_steps
        )

        # Move model to device
        if self.args.n_gpu > 0:
            self.model.to(self.args.device)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Create freeze/unfreeze helpers
        def freeze_heads_and_backbone(current):
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            for env, head in self.model.lm_heads.items():
                for p in head.parameters():
                    p.requires_grad = (env == current)
        def unfreeze_all():
            for p in self.model.encoder.parameters():
                p.requires_grad = True
            for head in self.model.lm_heads.values():
                for p in head.parameters():
                    p.requires_grad = True

        self.state.global_step = 0
        phi_counter = 0
        phi_batches = []
        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)

        recent_head_losses = []
        recent_phi_losses = []

        self.scaler = GradScaler()
        iter_loaders = {env: cycle(dl) for env, dl in dataloaders.items()}

        # Training loop
        for epoch in range(int(num_train_epochs)):
            if self.is_world_process_zero():
                print(f"===== EPOCH {epoch+1}/{num_train_epochs} =====")

            for round_idx in tqdm(range(steps_per_epoch)):
                for env in training_set:
                    if self.state.global_step >= max_steps:
                        break

                    # Head update
                    batch = next(iter_loaders[env])

                    freeze_heads_and_backbone(env)
                    head_optimizers[env].zero_grad()
                    self.model.train()

                    batch = {k: v.to(self.args.device) for k, v in batch.items() if torch.is_tensor(v)}
                    with autocast(device_type="cuda"):
                        loss_head = self.model(**batch).loss
                    self.scaler.scale(loss_head).backward()
                    self.scaler.step(head_optimizers[env])
                    self.scaler.update()
                    head_schedulers[env].step()

                    # Track head loss
                    self.state.global_step += 1
                    recent_head_losses.append(loss_head.item())
                    moving_avg_head = compute_moving_average(recent_head_losses, window_size=20)
                    phi_counter += 1
                    phi_batches.append({k: v.detach().cpu() for k, v in batch.items()})

                    # Logging head loss
                    if self.is_world_process_zero() and nb_steps_model_saving > 0 and self.state.global_step % nb_steps_model_saving == 0:
                        wandb.log({
                            "training/head_loss": loss_head.item(),
                            "training/head_loss_moving_avg": moving_avg_head,
                        }, step=self.state.global_step)

                    # Save heads/models
                    if nb_steps_heads_saving and self.state.global_step % nb_steps_heads_saving == 0:
                        self.save_heads(self.state.global_step)
                    if nb_steps_model_saving and self.state.global_step % nb_steps_model_saving == 0:
                        self.save_intermediary_model(self.state.global_step)

                    # Phi update every k head updates
                    if phi_counter >= update_phi_every_k:
                        unfreeze_all()
                        phi_optimizer.zero_grad()
                        total_phi_loss = 0.0
                        
                        for batch_phi in phi_batches:
                            batch_phi = {k: v.to(self.args.device) for k, v in b.items()}
                            with autocast(device_type="cuda"):
                                total_phi_loss += self.model(**batch_phi).loss
                        self.scaler.scale(total_phi_loss).backward()
                        self.scaler.step(phi_optimizer)
                        self.scaler.update()
                        phi_scheduler.step()

                        # Track phi loss
                        recent_phi_losses.append(total_phi_loss.item())
                        moving_avg_phi = compute_moving_average(recent_phi_losses, window_size=20)
                        if self.is_world_process_zero() and nb_steps_model_saving > 0 and self.state.global_step % nb_steps_model_saving == 0:
                            wandb.log({
                                "training/phi_loss": total_phi_loss.item(),
                                "training/phi_loss_moving_avg": moving_avg_phi,
                            }, step=self.state.global_step)

                        # Reset phi buffers
                        phi_counter = 0
                        phi_batches.clear()
        if self.is_world_process_zero():
            print("=== Training complete. Total rounds:", self.state.global_step / len(training_set))


    def save_intermediary_model(self, n_steps):
        fname = os.path.join(self.args.output_dir, f"model-{n_steps}")
        self.save_model(output_dir=fname)

    def save_heads(self, step_count):
        # Ne sauvegarder que si ce processus est le principal
        if not self.is_world_process_zero():
            return
        
        print("saving-heads")
        if not os.path.exists("lm_heads"):
            os.makedirs("lm_heads")

        for env, lm_head in self.model.lm_heads.items():
            filepath = os.path.join("lm_heads", "{}-{}".format(env, step_count))
            
            if hasattr(lm_head, "dense"):
                np.save(filepath, lm_head.dense.weight.data.cpu().numpy())
            elif hasattr(lm_head, "decoder"):
                np.save(filepath, lm_head.decoder.weight.data.cpu().numpy())
            elif hasattr(lm_head, "vocab_projector"):
                np.save(filepath, lm_head.vocab_projector.weight.data.cpu().numpy())
            else:
                print(f"La tête pour l'environnement {env} ne possède pas d'attribut de sauvegarde connu.")


    def get_single_train_dataloader(self, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
        )