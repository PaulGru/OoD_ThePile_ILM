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

import math
import os
import numpy as np
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

    def remove_dataparallel_wrapper(self):
        if hasattr(self.model, 'module'):
            self.model = self.model.module
    
    def run_evaluation(self):
        """
        Exécute l'évaluation sur self.eval_dataset et renvoie un tuple (eval_loss, perplexity).
        """
        eval_loss = None
        perplexity = None
        if self.eval_dataset is not None:
            eval_metrics = self.evaluate()  # Utilise la méthode evaluate() héritée (ou définie) dans ta classe Trainer
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
        if nb_steps is None and num_train_epochs is None:
            raise ValueError("Both nb_steps and num_train_epochs can't be None at the same time")

        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        min_train_set_size = min([len(data["train"]) for _, data in training_set.items()])
        
        num_envs = len(training_set)
        # Calcul du nombre d'updates (steps) effectués durant une epoch pour chaque environnement
        num_rounds_per_epoch = math.floor(
            min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size)
        )

        if nb_steps is not None:
            total_steps_per_epoch = num_rounds_per_epoch * num_envs
            num_train_epochs = max(1, math.ceil(nb_steps / total_steps_per_epoch))
            max_steps = nb_steps
        else:
            total_steps_per_epoch = num_rounds_per_epoch * num_envs
            max_steps = total_steps_per_epoch * num_train_epochs

        # Préparation des DataLoader, optimizers et lr_schedulers pour chaque environnement
        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(env_name, data_features["train"])
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

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        total_trained_steps = 0
        log_interval = 5  # Par exemple, log tous les 5 steps

        best_eval_loss = float('inf')
        stop_training = False

        # Initialisation du scaler pour AMP
        scaler = torch.amp.GradScaler("cuda")

        for epoch in range(int(num_train_epochs)):
            print("\n" + "=" * 70)
            print(f"===== DÉBUT DE L'ÉPOQUE {epoch + 1}/{num_train_epochs} =====")
            print("=" * 70 + "\n")

            # Rendre itérables tous les DataLoader par environnement
            iter_loaders = {env_name: iter(dataloaders[env_name]) for env_name in training_set.keys()}

            for round_idx in range(num_rounds_per_epoch):
                round_loss_sum = 0.0
                round_loss_count = 0

                print(f"----- Début du round {round_idx + 1}/{num_rounds_per_epoch} : Mise à jour de tous les environnements -----")
                
                for env_name in training_set.keys():
                    if total_trained_steps >= max_steps:
                        stop_training = True
                        break

                    print(f"[Step {total_trained_steps + 1}] Entraînement sur l'environnement : {env_name}")

                    # Réinitialisation des gradients pour l'encodeur partagé et la tête de l'environnement courant
                    optimizer.zero_grad()
                    optimizers[env_name].zero_grad()

                    try:
                        batch = next(iter_loaders[env_name])
                    except StopIteration:
                        iter_loaders[env_name] = iter(dataloaders[env_name])
                        batch = next(iter_loaders[env_name])

                    # On s'assure que la batch est sur le bon device
                    batch = {k: v.to(self.args.device) for k, v in batch.items()}

                    self.model.train()
                    # Calcul du forward en AMP
                    with torch.amp.autocast("cuda"):               
                        outputs = self.model(**batch, env_name=env_name)
                        loss = outputs.loss

                    # Accumulation de la loss pour le reporting
                    step_loss = loss.item()
                    round_loss_sum += step_loss
                    round_loss_count += 1
                    print(f"    Loss pour l'environnement {env_name}: {step_loss:.4f}")

                    # Rétropropagation avec AMP
                    scaler.scale(loss).backward()

                    # Clipping des gradients si nécessaire
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        scaler.unscale_(optimizers[env_name])
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    # Mise à jour des deux optimizers et mise à jour du scaler
                    scaler.step(optimizer)
                    scaler.step(optimizers[env_name])
                    scaler.update()

                    # Mise à jour des schedulers
                    lr_scheduler.step()
                    lr_schedulers[env_name].step()

                    print(f"    Fin d'update sur l'environnement {env_name}.")

                    total_trained_steps += 1

                    # Sauvegardes éventuelles
                    if saving_heads and total_trained_steps % nb_steps_heads_saving == 0:
                        self.save_heads(total_trained_steps)
                    if saving_intermediary_models and total_trained_steps % nb_steps_model_saving == 0:
                        self.save_intermediary_model(total_trained_steps)
                    
                    if stop_training:
                        break
                
                if round_loss_count > 0:
                    avg_round_loss = round_loss_sum / round_loss_count
                else:
                    avg_round_loss = 0.0

                print(f"----- Fin du round {round_idx + 1}/{num_rounds_per_epoch}: Loss moyenne sur ce round = {avg_round_loss:.4f} -----\n")

                # Logging et évaluation périodique
                if total_trained_steps % log_interval == 0:
                    eval_loss, perplexity = self.run_evaluation()
                    if self.is_world_process_zero():
                        log_path = os.path.join(self.args.output_dir, "training_log.csv")
                        if total_trained_steps == log_interval and os.path.exists(log_path):
                            os.remove(log_path)
                        if total_trained_steps == log_interval and not os.path.exists(log_path):
                            with open(log_path, "w") as f:
                                f.write("epoch,global_step,train_loss,val_loss,perplexity\n")
                        with open(log_path, "a") as f:
                            f.write(f"{epoch + 1},{total_trained_steps},{avg_round_loss:.4f},{eval_loss:.4f},{perplexity:.4f}\n")
                        print(f"--> Résumé [Step {total_trained_steps}] : Loss moyenne = {avg_round_loss:.4f}, Eval Loss = {eval_loss:.4f}, Perplexity = {perplexity:.4f}")
                        

                    if eval_loss is not None and eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_model_path = os.path.join(self.args.output_dir, "best_model")
                        self.model.save_pretrained(best_model_path, safe_serialization=False)
                        print(f"Meilleur modèle sauvegardé à l'étape {total_trained_steps} avec eval_loss = {eval_loss:.4f}")

                
                if stop_training:
                    break

            print(f"===== Fin de l'ÉPOQUE {epoch + 1}/{num_train_epochs} =====\n")
            if stop_training:
                break
        
        print("Entraînement terminé. Nombre total de steps:", total_trained_steps)


    def ensemble_train(
            self,
            training_set,
            nb_steps: Optional[int] = None,
            nb_steps_heads_saving: Optional[int] = 0,
            num_train_epochs: Optional[int] = 1,
            nb_steps_model_saving: Optional[int] = 0,
            **kwargs,
    ):
        if nb_steps is None and num_train_epochs is None:
            raise ValueError("Both nb_steps and num_train_epochs can't be None at the same time")

        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        min_train_set_size = min([len(data["train"]) for _, data in training_set.items()])

        num_envs = len(training_set)
        # le nombre d'updates (steps) effectués durant une epoch.
        num_rounds_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size))

        if nb_steps is not None:
            total_steps_per_epoch = num_rounds_per_epoch * num_envs
            num_train_epochs = max(1, math.ceil(nb_steps / total_steps_per_epoch))
            max_steps = nb_steps
        else:
            total_steps_per_epoch = num_rounds_per_epoch * num_envs
            max_steps = total_steps_per_epoch * num_train_epochs

        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            
            dataloaders[env_name] = self.get_single_train_dataloader(env_name, data_features["train"])
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
        if self.args.n_gpu >  1:
            self.model = torch.nn.DataParallel(self.model)
        
        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        print("Nombre total d'exemples traités approximativement :", total_train_batch_size * max_steps)
        print(min_train_set_size)

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        total_trained_steps = 0
        log_interval = 5

        best_eval_loss = float('inf')
        stop_training = False

        # Initialisation du scaler pour AMP
        scaler = torch.amp.GradScaler("cuda")
        
        for epoch in range(int(num_train_epochs)):
            print("\n" + "=" * 70)
            print(f"===== DÉBUT DE L'ÉPOQUE {epoch + 1}/{num_train_epochs} =====")
            print("=" * 70 + "\n")

            # make all dataloader iterateable
            iter_loaders = {}
            for env_name in training_set.keys():
                train_loader = dataloaders[env_name]
                iter_loaders[env_name] = iter(train_loader)

            for round_idx in range(num_rounds_per_epoch):
                round_loss_sum = 0.0
                round_loss_count = 0

                print(f"----- Début du round {round_idx + 1}/{num_rounds_per_epoch} : Mise à jour de tous les environnements -----")

                for env_name in training_set.keys():
                    if total_trained_steps >= max_steps:
                        stop_training = True
                        break
                    
                    print(f"[Step {total_trained_steps + 1}] Entraînement sur l'environnement : {env_name}")

                    optimizer.zero_grad()
                    for e_n in training_set.keys():
                        optimizers[e_n].zero_grad()

                    try:
                        batch = next(iter_loaders[env_name])
                    except StopIteration:
                        iter_loaders[env_name] = iter(dataloaders[env_name])
                        batch = next(iter_loaders[env_name])

                    # On s'assure que la batch est sur le bon device
                    batch = {k: v.to(self.args.device) for k, v in batch.items()}

                    self.model.train()
                    # Calcul du forward en AMP
                    with torch.amp.autocast("cuda"):               
                        outputs = self.model(**batch)
                        loss = outputs.loss

                    # Accumulation de la loss pour le reporting
                    step_loss = loss.item()
                    round_loss_sum += step_loss
                    round_loss_count += 1
                    print(f"    Loss pour l'environnement {env_name}: {step_loss:.4f}")

                    # Rétropropagation avec AMP
                    scaler.scale(loss).backward()

                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:

                        scaler.unscale_(optimizer)
                        for e_n in training_set.keys():
                            scaler.unscale_(optimizers[e_n])
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    scaler.step(optimizer)
                    for e_n in training_set.keys():
                        scaler.step(optimizers[e_n])
                    scaler.update()

                    lr_scheduler.step()
                    for e_n in training_set.keys():
                        lr_schedulers[e_n].step()

                    print(f"    Fin d'update sur l'environnement {env_name}.")

                    total_trained_steps += 1
                    
                    # Sauvegardes éventuelles
                    if saving_heads and total_trained_steps % nb_steps_heads_saving == 0:
                        self.save_heads(total_trained_steps)
                    if saving_intermediary_models and total_trained_steps % nb_steps_model_saving == 0:
                        self.save_intermediary_model(total_trained_steps)
                    
                    if stop_training:
                        break
                
                if round_loss_count > 0:
                    avg_round_loss = round_loss_sum / round_loss_count
                else:
                    avg_round_loss = 0.0

                print(f"----- Fin du round {round_idx + 1}/{num_rounds_per_epoch}: Loss moyenne sur ce round = {avg_round_loss:.4f} -----\n")

                if total_trained_steps % log_interval == 0:
                    eval_loss, perplexity = self.run_evaluation()
                    if self.is_world_process_zero():
                        log_path = os.path.join(self.args.output_dir, "training_log.csv")
                        if total_trained_steps == log_interval and os.path.exists(log_path):
                            os.remove(log_path)
                        if total_trained_steps == log_interval and not os.path.exists(log_path):
                            with open(log_path, "w") as f:
                                f.write("epoch,global_step,train_loss,val_loss,perplexity\n")
                        with open(log_path, "a") as f:
                            f.write(f"{epoch+1},{total_trained_steps},{avg_round_loss:.4f},{eval_loss:.4f},{perplexity:.4f}\n")

                    # Sauvegarde du meilleur modèle si la loss d'évaluation est meilleure
                    if eval_loss is not None and eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_model_path = os.path.join(self.args.output_dir, "best_model")
                        self.model.save_pretrained(best_model_path, safe_serialization=False)
                        print(f"Meilleur modèle sauvegardé à l'étape {total_trained_steps} avec eval_loss = {eval_loss:.4f}")

                if stop_training:
                    break

            print(f"===== Fin de l'ÉPOQUE {epoch + 1}/{num_train_epochs} =====\n")
            if stop_training:
                break
        print("Entraînement terminé. Nombre total de steps:", total_trained_steps)


    def multitask_train(
        self,
        training_set,
        nb_steps: Optional[int] = None,
        nb_steps_heads_saving: Optional[int] = 0,
        num_train_epochs: Optional[int] = 1,
        nb_steps_model_saving: Optional[int] = 0,
        **kwargs,
    ):
        if nb_steps is None and num_train_epochs is None:
            raise ValueError("Both nb_steps and num_train_epochs can't be None at the same time")
        
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        min_train_set_size = min([len(data["train"]) for _, data in training_set.items()])

        # le nombre d'updates (steps) effectués durant une epoch.
        num_update_steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size))

        if nb_steps is not None:
            max_steps = nb_steps
            num_train_epochs = max(1, math.floor(max_steps / num_update_steps_per_epoch))
        else:
            max_steps = num_update_steps_per_epoch * num_train_epochs


        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(env_name, data_features["train"])
            
            if hasattr(self.model, "lm_heads"):
                optimizer, lr_scheduler = self.create_optimizer_and_scheduler(self.model.lm_heads[env_name], max_steps)
            else:
                optimizer, lr_scheduler = self.create_optimizer_and_scheduler(self.model, num_training_steps=max_steps)

            optimizers[env_name] = optimizer
            lr_schedulers[env_name] = lr_scheduler

        # Optimizer for the shared encoder
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
        num_examples = total_train_batch_size * max_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  num_update_steps_per_epoch = {num_update_steps_per_epoch}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        total_trained_steps = 0
        log_interval = 50  # par exemple, log tous les 100 steps

        print("Num train epoch: ", num_train_epochs)
        print("Batch size: ", total_train_batch_size)
        print("Train data size: ", min_train_set_size)
        print("num_update_steps_per_epoch: ", num_update_steps_per_epoch)
        
        best_eval_loss = float('inf')

        for epoch in range(int(num_train_epochs)):
            print("\n" + "="*50)
            print(f"=== Début de l'Époque {epoch+1} ===")
            print("="*50 + "\n")
        
            # make all dataloader iterateable
            iter_loaders = {}
            for env_name in training_set.keys():
                train_loader = dataloaders[env_name]
                iter_loaders[env_name] = iter(train_loader)

            # [ADDED] Initialize accumulators for the epoch's training loss
            epoch_loss_sum = 0.0
            epoch_loss_count = 0

            for _ in tqdm(range(num_update_steps_per_epoch)):
                if total_trained_steps >= max_steps:
                    break

                for env_name in training_set.keys():
                    logger.info(f" Update on environment {env_name}")
                    # get a batch
                    optimizer.zero_grad()
                    optimizers[env_name].zero_grad()

                    batch = next(iter_loaders[env_name])

                    # Suppose que self.args.device contient le device (ex. "cuda:0" ou "cuda:1")
                    batch = {k: v.to(self.args.device) for k, v in batch.items()}

                    # Pas d'ensemblage ici
                    # Au lieu d'appeler self.training_step() qui ne supporte pas env_name,
                    # on fait un forward explicite en passant l'argument env_name.
                    self.model.train()
                    outputs = self.model(**batch, env_name=env_name)  # Passage explicite de env_name
                    loss = outputs.loss
                    loss.backward()

                    epoch_loss_sum += loss.item()
                    epoch_loss_count += 1

                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        if hasattr(optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            optimizer.clip_grad_norm(self.args.max_grad_norm)
                            optimizers[env_name].clip_grad_norm(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.args.max_grad_norm,
                            )

                    optimizer.step()
                    optimizers[env_name].step()

                    lr_scheduler.step()
                    lr_schedulers[env_name].step()

                    total_trained_steps += 1

                    if saving_heads and total_trained_steps % nb_steps_heads_saving == 0:
                        self.save_heads(total_trained_steps)
                    if saving_intermediary_models and total_trained_steps % nb_steps_model_saving == 0:
                        self.save_intermediary_model(total_trained_steps)


                    # [ADDED] After finishing all update steps of the epoch:
                    if total_trained_steps % log_interval == 0:
                    # Calculate average training loss for the epoch.
                        avg_train_loss = epoch_loss_sum / epoch_loss_count if epoch_loss_count > 0 else 0.0

                        eval_loss, perplexity = self.run_evaluation()

                        val_str = f"{eval_loss:.4f}" if eval_loss is not None else ""
                        ppl_str = f"{perplexity:.4f}" if perplexity is not None else ""

                        if self.is_world_process_zero():
                            log_path = os.path.join(self.args.output_dir, "training_log.csv")
                            
                            if total_trained_steps == log_interval and os.path.exists(log_path):
                                os.remove(log_path)

                            # On first evaluation, write header if file doesn't exist
                            if total_trained_steps == log_interval and not os.path.exists(log_path):
                                with open(log_path, "w") as f:
                                    f.write("epoch,global_step,train_loss,val_loss,perplexity\n")
                            with open(log_path, "a") as f:
                                f.write(f"{epoch+1},{total_trained_steps},{avg_train_loss:.4f},{val_str},{ppl_str}\n")

                            print("-" * 50)
                            print(f"Step {total_trained_steps} (Époque {epoch+1}):")
                            print(f"  Train Loss = {avg_train_loss:.4f}")
                            print(f"  Val Loss   = {val_str if eval_loss is not None else 'N/A'}")
                            print(f"  Perplexity = {ppl_str if perplexity is not None else 'N/A'}")
                            print("-" * 50)

                        # Sauvegarde du meilleur modèle si la loss d'évaluation est meilleure
                        if eval_loss is not None and eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            best_model_path = os.path.join(self.args.output_dir, "best_model")
                            self.model.save_pretrained(best_model_path, safe_serialization=False)
                            print(f"Meilleur modèle sauvegardé à l'étape {total_trained_steps} avec eval_loss = {eval_loss:.4f}")

                        # Réinitialiser les accumulateurs pour le log d'intervalle
                        epoch_loss_sum = 0.0
                        epoch_loss_count = 0

            # Fin d'époque : on peut afficher un résumé si nécessaire
            # (Attention, si l'entraînement s'arrête avant la fin d'une époque, ce résumé risque de couvrir une partie incomplète)
            if epoch_loss_count > 0:
                avg_epoch_loss = epoch_loss_sum / epoch_loss_count
            else:
                avg_epoch_loss = 0.0
            logger.info(f"Fin de l'époque {epoch+1} : Train Loss = {avg_epoch_loss:.4f}")
    

    def save_intermediary_model(self, n_steps):
        fname = os.path.join(self.args.output_dir, f"model-{n_steps}")
        self.save_model(output_dir=fname)

    def save_heads(self, step_count):
        if not hasattr(self.model, "lm_heads"):
            # Si le modèle n'a pas d'attribut lm_heads (mode eLM), on ne sauvegarde rien.
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


    def get_single_train_dataloader(self, env_name, train_dataset):
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
            collate_fn=self.data_collator
        )