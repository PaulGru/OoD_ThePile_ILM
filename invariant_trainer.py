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
import csv
import numpy as np
from itertools import cycle
import random

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

        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(data_features["train"])

            optimizers[env_name], lr_schedulers[env_name] = self.create_optimizer_and_scheduler(
                self.model.lm_heads[env_name],
                num_training_steps=max_steps
            )

        optimizer, lr_scheduler = self.create_optimizer_and_scheduler(
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
                   
                    optimizer.zero_grad()
                    optimizers[env_name].zero_grad()
                    self.model.train()

                    batch = {k: v.to(self.args.device) for k, v in batch.items() if torch.is_tensor(v)}
                    outputs = self.model(**batch)
                    loss = outputs.loss

                    loss.backward()

                    # Mise à jour des deux optimizers et mise à jour du scaler
                    optimizer.step()
                    optimizers[env_name].step()

                    # Mise à jour des schedulers
                    lr_scheduler.step()
                    lr_schedulers[env_name].step()

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