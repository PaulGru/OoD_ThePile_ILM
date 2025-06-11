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

from typing import Optional

logger = logging.get_logger(__name__)


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
    
    def run_evaluation(self):
        """
        Exécute l'évaluation sur self.eval_dataset et renvoie un tuple (eval_loss, perplexity).
        """
        eval_loss = None
        perplexity = None
        if self.eval_dataset is not None:
            eval_metrics = self.evaluate()
            eval_loss = eval_metrics.get("eval_loss")
            if eval_loss is not None:
                perplexity = math.exp(eval_loss)
        return eval_loss, perplexity
    

    def invariant_train(
            self,
            training_set,
            nb_steps: Optional[int] = None,
            nb_steps_heads_saving: Optional[int] = 0,
            num_train_epochs: Optional[int] = 1,
            nb_steps_model_saving: Optional[int] = 0,
            **kwargs,
    ):
        min_train_set_size = min([len(data["train"]) for _, data in training_set.items()])
        
        # Calcul du nombre d'updates (steps) effectués durant une epoch pour chaque environnement
        num_update_steps_per_epoch = math.floor(
            min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size)
        )
        if nb_steps is not None:
            num_train_epochs = max(1, math.floor(nb_steps / num_update_steps_per_epoch))
            max_steps = nb_steps
        else:
            max_steps = num_update_steps_per_epoch * num_train_epochs

        # Préparation des DataLoader, optimizers et lr_schedulers pour chaque environnement
        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(data_features["train"])
            
            optimizer_env, lr_scheduler_env = self.create_optimizer_and_scheduler(
                self.model.lm_heads[env_name],
                num_training_steps=max_steps
            )
            
            optimizers[env_name] = optimizer_env
            lr_schedulers[env_name] = lr_scheduler_env

        # Création de l'optimiseur et du scheduler pour l'ensemble du modèle
        optimizer, lr_scheduler = self.create_optimizer_and_scheduler(self.model.encoder, num_training_steps=max_steps)

        if self.args.n_gpu > 0:
            self.model.to(self.args.device)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Initialisation du scaler pour AMP
        scaler = torch.amp.GradScaler("cuda")

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        self.state.global_step = 0
        cumulative_loss = 0.0
        cumulative_count = 0

        for epoch in range(int(num_train_epochs)):
            print(f"\n===== ÉPOQUE {epoch + 1}/{num_train_epochs} =====")

            iter_loaders = {}
            for env_name in training_set.keys():
                train_loader = dataloaders[env_name]
                iter_loaders[env_name] = iter(train_loader)
            
            # Un round correspond à une itération sur tous les environnements
            for round_idx in tqdm(range(num_update_steps_per_epoch)):
                if self.state.global_step >= max_steps :
                        break

                for env_name in training_set.keys():
                    batch = next(iter_loaders[env_name])
                
                    optimizer.zero_grad()
                    optimizers[env_name].zero_grad()
                    self.model.train()

                   # Forward avec logits moyennés (toutes les têtes) et AMP
                    with torch.amp.autocast("cuda"):
                        batch = {k: v.to(self.args.device) for k, v in batch.items() if torch.is_tensor(v)}
                        outputs = self.model(**batch)
                        loss = outputs.loss

                    # Rétropropagation avec AMP sur encodeur + tête active seulement
                    scaler.scale(loss).backward()

                    # Clipping des gradients si nécessaire
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        scaler.unscale_(optimizers[env_name])
                        
                    # Mise à jour des deux optimizers et mise à jour du scaler
                    scaler.step(optimizer)
                    scaler.step(optimizers[env_name])
                    scaler.update()

                    # Mise à jour des schedulers
                    lr_scheduler.step()
                    lr_schedulers[env_name].step()

                    self.state.global_step += 1
                    cumulative_loss += loss.item()
                    cumulative_count += 1

                    if self.is_world_process_zero() and self.state.global_step % nb_steps_model_saving == 0:
                        wandb.log(
                            {"training/train_loss": loss.item(),},
                        step=self.state.global_step)

                    if saving_heads and self.state.global_step % nb_steps_heads_saving == 0:
                        self.save_heads(self.state.global_step)
                    if saving_intermediary_models and self.state.global_step % nb_steps_model_saving == 0:
                        self.save_intermediary_model(self.state.global_step)
                    
                
        print("=== Entraînement du modèle terminé. Nombre total de rounds:", self.state.global_step/len(training_set.keys()))
        
        average_loss = cumulative_loss / cumulative_count if cumulative_count > 0 else float('inf')
        return {"metrics": {"train_loss": average_loss}}


    def invariant_train_games(
            self,
            training_set,
            nb_steps: Optional[int] = None,
            nb_steps_heads_saving: Optional[int] = 0,
            num_train_epochs: Optional[int] = 1,
            nb_steps_model_saving: Optional[int] = 0,
            update_phi_every_k: Optional[int] = 5,
            **kwargs
    ):
        if nb_steps is None and num_train_epochs is None:
            raise ValueError("Both nb_steps and num_train_epochs can't be None.")
        if len(kwargs) > 0:
            raise TypeError(f"train() received unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        min_train_set_size = min([len(data["train"]) for _, data in training_set.items()])
        
        # Calcul du nombre d'updates (steps) effectués durant une epoch pour chaque environnement
        num_update_steps_per_epoch = math.floor(
            min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size)
        )
        if nb_steps is not None:
            num_train_epochs = max(1, math.ceil(nb_steps / num_update_steps_per_epoch))
            max_steps = nb_steps
        else:
            max_steps = num_update_steps_per_epoch * num_train_epochs

        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(data_features["train"])
            
            if hasattr(self.model, "lm_heads"):
                optimizer_env, lr_scheduler_env = self.create_optimizer_and_scheduler(
                    self.model.lm_heads[env_name],
                    num_training_steps=max_steps
                )
            else:
                optimizer_env, lr_scheduler_env = self.create_optimizer_and_scheduler(
                    self.model, num_training_steps=max_steps
                )
            
            optimizers[env_name] = optimizer_env
            lr_schedulers[env_name] = lr_scheduler_env

        # Optimizer et scheduler pour le modèle partagé (l'encodeur)
        if hasattr(self.model, 'encoder'):
            shared_encoder = self.model.encoder
        elif hasattr(self.model, 'distilbert'):
            shared_encoder = self.model.distilbert
        else:
            raise AttributeError("The model does not have an encoder attribute.")

        optimizer, lr_scheduler = self.create_optimizer_and_scheduler(shared_encoder, num_training_steps=max_steps)

        self.state = TrainerState()
        if self.args.n_gpu > 0:
            self.model.to(self.args.device)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        print("Nombre total d'exemples traités approximativement :", total_train_batch_size * max_steps)
        print(min_train_set_size)

        self.state.global_step = 0
        phi_accum_loss = 0
        phi_update_counter = 0
        phi_batches = []

        best_eval_loss = float('inf')
        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        log_interval = 50

        scaler = torch.amp.GradScaler("cuda")

        csv_file = os.path.join(self.args.output_dir, "training_loss_history.csv")
        if self.is_world_process_zero():
            if os.path.exists(csv_file):
                os.remove(csv_file)
            header = ["Epoch"] + list(training_set.keys())
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

        stop_training = False

        for epoch in range(int(num_train_epochs)):
            print("\n" + "=" * 70)
            print(f"===== DÉBUT DE L'ÉPOQUE {epoch + 1}/{num_train_epochs} =====")
            print("=" * 70 + "\n")

            iter_loaders = {env_name: iter(dataloaders[env_name]) for env_name in training_set.keys()}

            for round_idx in range(num_rounds_per_epoch):
                round_loss_sum = 0.0
                round_loss_count = 0

                print(f"----- Début du round {round_idx + 1}/{num_rounds_per_epoch} : Mise à jour de tous les environnements -----")

                for env_name in training_set.keys():
                    if self.state.global_step >= max_steps:
                        stop_training = True
                        break

                    print(f"[Step {self.state.global_step + 1}] Entraînement sur l'environnement : {env_name}")

                    try:
                        batch = next(iter_loaders[env_name])
                    except StopIteration:
                        iter_loaders[env_name] = iter(dataloaders[env_name])
                        batch = next(iter_loaders[env_name])

                    batch = {k: v.to(self.args.device) for k, v in batch.items()}

                    optimizers[env_name].zero_grad()

                    self.model.train()

                    for name, head in self.model.lm_heads.items():
                        if name != env_name:
                            for param in head.parameters():
                                param.requires_grad = False

                    for param in shared_encoder.parameters():
                        param.requires_grad = False

                    with torch.amp.autocast("cuda"):
                        outputs = self.model(**batch, env_name=env_name)
                        loss = outputs.loss

                    step_loss = loss.item()
                    env_epoch_losses[env_name].append(step_loss)
                    round_loss_sum += step_loss
                    round_loss_count += 1
                    print(f"    Loss pour l'environnement {env_name}: {step_loss:.4f}")

                    scaler.scale(loss).backward(retain_graph=True)

                    for param in shared_encoder.parameters():
                        param.requires_grad = True
                    for name, head in self.model.lm_heads.items():
                        if name != env_name:
                            for param in head.parameters():
                                param.requires_grad = True
                    
                    # Clipping des gradients si nécessaire
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        scaler.unscale_(optimizers[env_name])
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    scaler.step(optimizers[env_name])
                    scaler.update()
                    
                    lr_schedulers[env_name].step()

                    phi_batches.append((batch, env_name))
                    phi_update_counter += 1


                    if phi_update_counter % update_phi_every_k == 0:
                        optimizer.zero_grad()
                        phi_accum_loss = 0

                        for batch_phi, env_phi_name in phi_batches:
                            batch_phi = {k: v.to(self.args.device) for k, v in batch_phi.items()}
                            with torch.amp.autocast("cuda"):
                                outputs_phi = self.model(**batch_phi, env_name=env_phi_name)
                                loss_phi = outputs_phi.loss
                            phi_accum_loss = phi_accum_loss + loss_phi

                        scaler.scale(phi_accum_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        lr_scheduler.step()

                        phi_batches = []
                        phi_update_counter = 0
                        self.state.global_step += 1

                        if self.state.global_step % log_interval == 0:
                            eval_loss, perplexity = self.run_evaluation()
                            if self.is_world_process_zero():
                                log_path = os.path.join(self.args.output_dir, "training_log.csv")
                                if self.state.global_step == log_interval and os.path.exists(log_path):
                                    os.remove(log_path)
                                if self.state.global_step == log_interval and not os.path.exists(log_path):
                                    with open(log_path, "w") as f:
                                        f.write("epoch,global_step,train_loss,val_loss,perplexity\n")
                                with open(log_path, "a") as f:
                                    f.write(f"{epoch + 1},{self.state.global_step},{round_loss_sum / max(1, round_loss_count):.4f},{eval_loss:.4f},{perplexity:.4f}\n")
                                print(f"--> Résumé [Step {self.state.global_step}] : Eval Loss = {eval_loss:.4f}, Perplexity = {perplexity:.4f}")

                            if eval_loss is not None and eval_loss < best_eval_loss:
                                best_eval_loss = eval_loss
                                best_model_path = os.path.join(self.args.output_dir, "best_model")
                                self.model.save_pretrained(best_model_path, safe_serialization=False)
                                print(f"Meilleur modèle sauvegardé à l'étape {self.state.global_step} avec eval_loss = {eval_loss:.4f}")

                        if saving_heads and self.state.global_step % nb_steps_heads_saving == 0:
                            self.save_heads(self.state.global_step)
                        if saving_intermediary_models and self.state.global_step % nb_steps_model_saving == 0:
                            self.save_intermediary_model(self.state.global_step)

                    if stop_training:
                        break

                if round_loss_count > 0:
                    avg_round_loss = round_loss_sum / round_loss_count
                else:
                    avg_round_loss = 0.0

                print(f"----- Fin du round {round_idx + 1}/{num_rounds_per_epoch}: Loss moyenne sur ce round = {avg_round_loss:.4f} -----\n")

                if self.is_world_process_zero():
                    with open(csv_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        row = [epoch + 1]
                        for env_name in training_set.keys():
                            losses = env_epoch_losses[env_name]
                            avg_loss = sum(losses) / len(losses) if losses else ""
                            row.append(avg_loss)
                        writer.writerow(row)

                if stop_training:
                    break

            print(f"===== Fin de l'ÉPOQUE {epoch + 1}/{num_train_epochs} =====")

        print("Entraînement terminé. Nombre total de steps:", self.state.global_step)


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