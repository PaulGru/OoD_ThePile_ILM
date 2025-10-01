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
from transformers.trainer_utils import TrainOutput

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
    def __init__(self, *args, eirm_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eirm_args = eirm_args

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
            save_milestones: Optional[set] = None,
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

        if nb_steps is not None:
            max_steps = nb_steps
            steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size)
            )
            num_train_epochs = max(1, math.floor(max_steps / steps_per_epoch))
        else:
            steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size)
            )
            max_steps = steps_per_epoch * num_train_epochs

        dataloaders, head_optimizers, head_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(data_features["train"])
            optimizer, head_scheduler = self.create_optimizer_and_scheduler(
                self.model.lm_heads[env_name],
                num_training_steps=max_steps
            )
            head_optimizers[env_name] = optimizer
            head_schedulers[env_name] = head_scheduler

        phi_optimizer, phi_scheduler = self.create_optimizer_and_scheduler(
            self.model.encoder,
            num_training_steps=max_steps
        )

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
        logger.info(f"  steps_per_epoch = {steps_per_epoch}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        saving_heads = bool(nb_steps_heads_saving > 0)
        milestones = set(save_milestones or [])
        saving_intermediary_models = bool(nb_steps_model_saving > 0 or milestones)
        self.state.global_step = 0

        recent_losses = []

        self.scaler = GradScaler()

        self.use_amp = bool(self.args.fp16 and torch.cuda.is_available())
        print(f"use_amp = {self.use_amp}")

        iter_loaders = {env_name: cycle(dataloaders[env_name]) for env_name in training_set.keys()}
        for epoch in range(int(num_train_epochs)):
            logger.info(f" Epoch: {epoch}")

            # iter_loaders = {}
            # for env_name in training_set.keys():
            #     iter_loaders[env_name] = iter(dataloaders[env_name])

            for round_idx in tqdm(range(steps_per_epoch)):
                if self.state.global_step >= max_steps :
                    break

                for env_name in training_set.keys():
                    logger.info(f" Update on environement {env_name}")

                    phi_optimizer.zero_grad()
                    head_optimizers[env_name].zero_grad()

                    batch = next(iter_loaders[env_name])

                    self.model.train()
                    batch = self._prepare_inputs(batch)

                    if self.use_amp:
                        with autocast("cuda", enabled=self.use_amp):
                            loss = self.compute_loss(self.model, batch)
                    else:
                        loss = self.compute_loss(self.model, batch)

                    if self.args.n_gpu > 1:
                        loss = loss.mean()

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    loss = loss.detach()

                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        if self.use_amp:
                            self.scaler.unscale_(phi_optimizer)
                            self.scaler.unscale_(head_optimizers[env_name])

                        if hasattr(phi_optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            phi_optimizer.clip_grad_norm(self.args.max_grad_norm)
                            head_optimizers[env_name].clip_grad_norm(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.args.max_grad_norm,
                            )

                    if self.use_amp:
                        self.scaler.step(phi_optimizer)
                        self.scaler.step(head_optimizers[env_name])
                        self.scaler.update()
                    else:
                        phi_optimizer.step()
                        head_optimizers[env_name].step()

                    # Mise à jour des schedulers
                    phi_scheduler.step()
                    head_schedulers[env_name].step()

                    if (self.state.global_step % max(1, self.args.logging_steps)) == 0:
                        try:
                            lr = phi_scheduler.get_last_lr()[0]
                        except Exception:
                            lr = None
                        if lr is not None:
                            self.log({"training/lr": float(lr)})

                    self.state.global_step += 1

                    recent_losses.append(loss.item())
                    moving_avg_loss = compute_moving_average(recent_losses, window_size=20)
                    if (self.state.global_step % max(1, self.args.logging_steps)) == 0:
                        self.log({
                            "training/train_loss": loss.item(),
                            "training/train_loss_moving_avg": moving_avg_loss
                        })

                    if saving_heads and self.state.global_step % nb_steps_heads_saving == 0:
                        self.save_heads(self.state.global_step)
                    if milestones and (self.state.global_step in milestones):
                        self.save_intermediary_model(self.state.global_step)
                    # Sauvegarde périodique (si demandée)
                    if nb_steps_model_saving and (self.state.global_step % nb_steps_model_saving == 0):
                        self.save_intermediary_model(self.state.global_step)

        print("=== Entraînement du modèle terminé. Nombre total de rounds:", self.state.global_step/len(training_set.keys()))


    # --- EIRM utils -----------------------------------------------------
    def _named_params(self):
        for n, p in self.model.named_parameters():
            yield n, p

    def _phi_parameters(self):
        # ϕ := tous les params sauf les têtes par environnement
        for n, p in self._named_params():
            if not n.startswith("lm_heads."):
                yield p

    def _head_parameters(self, env):
        prefix = f"lm_heads.{env}."
        for n, p in self._named_params():
            if n.startswith(prefix):
                yield p

    def _set_requires_grad(self, module_or_params, flag: bool):
        if hasattr(module_or_params, "parameters"):
            params = module_or_params.parameters()
        else:
            params = module_or_params
        for p in params:
            p.requires_grad = flag

    def _as_train_split(self, ds):
        """Retourne le split d'entraînement si on reçoit un DatasetDict, sinon renvoie ds tel quel."""
        try:
            from datasets import DatasetDict
            if isinstance(ds, DatasetDict):
                if 'train' in ds:
                    return ds['train']
                # fallback: premier split dispo
                first_key = next(iter(ds.keys()))
                return ds[first_key]
        except Exception:
            pass
        return ds


    def _build_env_dataloaders(self):
        # Attendu: train_dataset est un dict {env: Dataset|DatasetDict}
        if isinstance(self.train_dataset, dict):
            envs = list(self.train_dataset.keys())
            dls = {}
            for e in envs:
                ds = self.train_dataset[e]
                ds_train = self._as_train_split(ds)  # <-- assure 'train'
                dls[e] = self.get_single_train_dataloader(ds_train)
            return envs, dls
        raise ValueError("EIRM requiert un train_dataset multi-environnements (dict {env: Dataset}).")

    def _make_optimizer(self, params):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        params_set = set(params)
        grouped = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p in params_set and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p in params_set and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if getattr(self.args, "adafactor", False):
            opt = Adafactor(
                grouped,
                scale_parameter=False,
                relative_step=False,
                weight_decay=self.args.weight_decay,
                lr=self.args.learning_rate,
            )
        else:
            opt = AdamW(
                grouped,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        num_warmup = (
            self.args.get_warmup_steps(self.args.max_steps)
            if hasattr(self.args, "get_warmup_steps")
            else self.args.warmup_steps
        )
        sch = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=opt,
            num_warmup_steps=num_warmup,
            num_training_steps=self.args.max_steps,
        )
        return opt, sch

    def _zero_all_optimizers(self, opts):
        for o in opts:
            o.zero_grad(set_to_none=True)

    # --- Entraînement EIRM ----------------------------------------------
    @torch.no_grad()
    def _log_var_env_loss(self):
        try:
            if not isinstance(self.eval_dataset, dict):
                return None
            losses = []
            for env, ds in self.eval_dataset.items():
                dl = self.get_eval_dataloader(ds)
                tot, n = 0.0, 0
                for batch in dl:
                    loss = self.compute_loss(self.model, batch).detach()
                    tot += loss.item()
                    n += 1
                if n > 0:
                    losses.append(tot / n)
            if losses:
                import statistics
                return statistics.pvariance(losses)
        except Exception:
            return None


    def train_eirm(
        self,
        training_set=None,
        nb_steps=None,
        nb_steps_heads_saving: int = 0,
        nb_steps_model_saving: int = 0,
        save_milestones=None,
        resume_from_checkpoint=None,
    ):
        """
        Entraînement EIRM (IRM-games) avec deux modes :
        - F-IRM : phi gelé, best-responses uniquement sur les têtes
        - V-IRM : best-responses par tête, puis mises à jour périodiques de phi

        Args:
            training_set (dict[str, Dataset] | None): optionnel, permet d'injecter {env: Dataset}
            nb_steps (int|None): si fourni, écrase self.args.max_steps
            nb_steps_heads_saving (int): sauvegarde des têtes toutes les X steps (0 = off)
            nb_steps_model_saving (int): sauvegarde du modèle toutes les X steps (0 = off)
            save_milestones (set[int]|None): steps spécifiques à sauvegarder
            resume_from_checkpoint: ignoré ici (on ne touche pas à ta logique existante)
        """
        if training_set is not None:
            self.train_dataset = training_set

        assert self.eirm_args is not None, "EIRM: arguments manquants (eirm_args)."
        mode = (self.eirm_args.eirm_mode or "none").lower()
        assert mode in {"f-irm", "v-irm"}, f"eirm_mode doit être 'F-IRM' ou 'V-IRM', reçu: {mode}"

        # --- Dataloaders par environnement ---
        envs, dls = self._build_env_dataloaders()

        # --- Optimiseurs par tête + pour phi ---
        head_opts, head_schs = {}, {}
        for e in envs:
            params = list(self._head_parameters(e))
            opt, sch = self._make_optimizer(params)
            head_opts[e], head_schs[e] = opt, sch

        phi_params = list(self._phi_parameters())
        phi_opt, phi_sch = self._make_optimizer(phi_params)

        # --- Phi gelé au départ en F-IRM (et optionnel en V-IRM) ---
        if self.eirm_args.freeze_phi_at_start or mode == "f-irm":
            self._set_requires_grad(phi_params, False)

        # Steps et progbar
        max_steps = int(nb_steps) if nb_steps is not None else (int(self.args.max_steps) if self.args.max_steps else math.inf)
        assert max_steps != math.inf and max_steps > 0, "EIRM: max_steps/nb_steps doit être fixé."

        # État + métriques
        global_step = 0
        round_idx = 0
        sum_loss, n_loss, phi_updates = 0.0, 0, 0

        # Sauvegardes comme dans ta boucle ILM
        milestones = set(save_milestones or [])
        saving_heads = bool(nb_steps_heads_saving and nb_steps_heads_saving > 0)

        # Iterateurs par env
        env_iters = {e: iter(dls[e]) for e in envs}

        progress = tqdm(total=max_steps, disable=not self.is_local_process_zero(), desc="EIRM")
        log_every = max(1, self.args.logging_steps or 1)

        while global_step < max_steps:
            # ----- 1) Best-responses : un *round* = une passe sur chaque env -----
            for env in envs:
                if global_step >= max_steps:
                    break
                try:
                    batch = next(env_iters[env])
                except StopIteration:
                    env_iters[env] = iter(dls[env])
                    batch = next(env_iters[env])

                # a) Phi gelé pendant BR (toujours en F-IRM, et en V-IRM hors phases phi)
                self._set_requires_grad(phi_params, False)

                # b) Seule la tête de l'env actif est entraînable
                for other in envs:
                    self._set_requires_grad(list(self._head_parameters(other)), other == env)

                # Forward/backward sur la tête active
                batch = self._prepare_inputs(batch)
                with self.autocast_smart_context_manager():
                    loss = self.compute_loss(self.model, batch)
                loss.backward()

                head_opts[env].step()
                head_schs[env].step()
                head_opts[env].zero_grad(set_to_none=True)

                # Compteurs + logs
                sum_loss += float(loss.detach().item())
                n_loss += 1
                global_step += 1
                progress.update(1)

                if (global_step % log_every) == 0 and self.is_local_process_zero():
                    logs = {"train/loss": float(loss.detach().item()), "step": int(global_step)}
                    # (Option) variance des losses par env si eval_dataset est un dict
                    var_env = self._log_var_env_loss()
                    if var_env is not None:
                        logs["eval/var_env_loss"] = float(var_env)
                    self.log(logs)

                # >>>>> SAUVEGARDES (réutilise tes fonctions existantes) <<<<<
                if saving_heads and (global_step % nb_steps_heads_saving == 0):
                    self.save_heads(global_step)
                if milestones and (global_step in milestones):
                    self.save_intermediary_model(global_step)
                if nb_steps_model_saving and (global_step % nb_steps_model_saving == 0):
                    self.save_intermediary_model(global_step)
                # <<<<< fin sauvegardes <<<<<

            round_idx += 1

            # ----- 2) Phase d'update de phi (V-IRM uniquement) -----
            if mode == "v-irm" and int(self.eirm_args.phi_update_every or 0) > 0:
                if round_idx % int(self.eirm_args.phi_update_every) == 0 and global_step < max_steps:
                    # Débloquer phi et geler toutes les têtes
                    self._set_requires_grad(phi_params, True)
                    for e in envs:
                        self._set_requires_grad(list(self._head_parameters(e)), False)

                    for _ in range(int(self.eirm_args.phi_update_batches or 1)):
                        if global_step >= max_steps:
                            break
                        e = random.choice(envs)
                        try:
                            batch = next(env_iters[e])
                        except StopIteration:
                            env_iters[e] = iter(dls[e])
                            batch = next(env_iters[e])

                        phi_opt.zero_grad(set_to_none=True)
                        batch = self._prepare_inputs(batch)
                        with self.autocast_smart_context_manager():
                            loss_phi = self.compute_loss(self.model, batch)
                        loss_phi.backward()
                        torch.nn.utils.clip_grad_norm_(list(phi_params), self.args.max_grad_norm or 1.0)
                        phi_opt.step()
                        phi_sch.step()
                        phi_updates += 1

                    # Re-geler phi pour la prochaine phase de BR
                    self._set_requires_grad(phi_params, False)

        progress.close()
        final_train_loss = (sum_loss / max(1, n_loss))
        metrics = {
            "train_loss": float(final_train_loss),
            "rounds": float(round_idx),
            "phi_updates": float(phi_updates),
        }
        return TrainOutput(global_step=int(global_step), training_loss=float(final_train_loss), metrics=metrics)


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

        num_workers = int(getattr(self.args, "dataloader_num_workers", 0) or 0)
        pin_mem = bool(getattr(self.args, "dataloader_pin_memory", False))
        dl_kwargs = dict(
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=num_workers,
            pin_memory=pin_mem,
            persistent_workers=False,
        )
        # prefetch_factor n’est pris en compte que si num_workers > 0
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = 1
        return DataLoader(train_dataset, **dl_kwargs)