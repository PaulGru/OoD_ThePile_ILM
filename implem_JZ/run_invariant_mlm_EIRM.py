#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, either express or implied. See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
import warnings
import logging
import math
import os
import sys
import torch
import wandb
import glob
import re
import time
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Optional
import math as _math

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from invariant_trainer import InvariantTrainer

from invariant_roberta import InvariantRobertaForMaskedLM, InvariantRobertaConfig
from invariant_distilbert import InvariantDistilBertForMaskedLM, InvariantDistilBertConfig

import transformers
from transformers import (
    CONFIG_MAPPING,
    TOKENIZER_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    DistilBertTokenizer,
    DistilBertTokenizerFast,
    RobertaTokenizer,
    RobertaTokenizerFast
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

CONFIG_MAPPING.update({'invariant-distilbert': InvariantDistilBertConfig})
CONFIG_MAPPING.update({'invariant-roberta': InvariantRobertaConfig})

MODEL_FOR_MASKED_LM_MAPPING.update({InvariantDistilBertConfig: InvariantDistilBertForMaskedLM})
MODEL_FOR_MASKED_LM_MAPPING.update({InvariantRobertaConfig: InvariantRobertaForMaskedLM})

TOKENIZER_MAPPING.update({InvariantDistilBertConfig: (DistilBertTokenizer, DistilBertTokenizerFast)})
TOKENIZER_MAPPING.update({InvariantRobertaConfig: (RobertaTokenizer, RobertaTokenizerFast)})

AutoConfig.register("invariant-distilbert", InvariantDistilBertConfig)
AutoConfig.register("invariant-roberta", InvariantRobertaConfig)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    mode: Optional[str] = field(
        default="ilm",
        metadata={
            "help": "Whether to train the heads as an ensemble instead of following the IRM-games dynamics"}
    )
    nb_steps_heads_saving: Optional[int] = field(
        default=0,
        metadata={"help": "Number of training steps between saving the head weights (if 0, the heads are not saved regularly)."},
    )
    nb_steps_model_saving: Optional[int] = field(
        default=0,
        metadata={"help": "Number of training steps between saving the full model (if 0, the heads are not saved regularly)."},
    )
    eval_ckpt_stride: int = field(
        default=1,
        metadata={"help": "Évaluer 1 checkpoint sur N (1 = tous)."}
    )
    eval_ckpt_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Limiter au K derniers checkpoints (None = pas de limite)."}
    )
    save_milestones: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated steps where full model checkpoints must be saved."},
    )
    evaluate_checkpoints_after_train: bool = field(
        default=False,
        metadata={"help": "If True, evaluate checkpoints at the end of training (otherwise use external script)."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file or a directory)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input validation data file or directory (a text file or directory)."},
    )
    ood_validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Fichier de validation Out-Of-Distribution (.txt) jamais vu à l'entraînement"}
    )
    tokenized_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Répertoire racine des datasets pré-tokenisés (datasets.save_to_disk)."}
    )
    eval_fraction: float = field(default=1.0, metadata={"help": "Fraction de l'IND/OOD à utiliser en évaluation (0<frac<=1)."})
    eval_seed: Optional[int] = field(default=None, metadata={"help": "Seed pour le sous-échantillonnage d'évaluation (défaut = training_args.seed)."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether to treat each line of the input files as a separate document."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    nb_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of training steps."}
    )
    use_erm: bool = field(
        default=False,
        metadata={"help": "Si True, utilise tokenized_dir/erm comme unique environnement d'entraînement (ERM). Sinon, tous les envs (ILM)."}
    )

    def __post_init__(self):
    # Priorité: si on fournit un dossier tokenisé, on l'utilise
        if self.tokenized_dir and str(self.tokenized_dir).strip():
            td = os.path.expandvars(os.path.expanduser(self.tokenized_dir))
            if not os.path.isdir(td):
                raise ValueError(f"--tokenized_dir '{td}' introuvable (pas un dossier).")
            # OK: on a une source valide -> on sort sans lever d'erreur
            self.tokenized_dir = td
            return
        # Sinon, si un train_file est donné, on laisse faire le flux texte brut
        if self.train_file and str(self.train_file).strip():
            return
        # Sinon, message explicite
        raise ValueError(
            "Aucun fichier d'entraînement ni dataset n'a été spécifié. "
            "Fournis --tokenized_dir=/chemin/vers/pile_tok_ilm (recommandé) "
            "ou --train_file=... (texte brut)."
        )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    On surcharge la classe par défaut pour permettre la boucle d'entraînement IRM Games.
    """
    head_updates_per_encoder_update : Optional[int] = field(
        default=1,
        metadata={"help": "Number of head updates per encoder update (IRM Games)"}
    )


@dataclass
class EIRMArguments:
    """Arguments spécifiques à l'entraînement EIRM (IRM-games)."""
    eirm_mode: str = field(
        default="none", metadata={"help": "Mode EIRM: one of ['none', 'F-IRM', 'V-IRM']"}
    )
    phi_update_every: int = field(
        default=0,
        metadata={"help": "En V-IRM: nombre de *rounds* (passes sur tous les envs) entre deux updates de ϕ. 0 => jamais."}
    )
    phi_update_batches: int = field(
        default=1,
        metadata={"help": "En V-IRM: nb de mini-batches (mix envs) pour chaque update de ϕ."}
    )
    freeze_phi_at_start: bool = field(
        default=False,
        metadata={"help": "Geler ϕ dès le départ (utile pour F-IRM ou pour warmup)."}
    )
    eirm_log_var_env_loss: bool = field(
        default=True,
        metadata={"help": "Logger la variance de la loss par environnement sur la val (pour early-stopping style EIRM)."}
    )


def main():
    if int(os.environ.get("RANK", "0")) == 0:
        print("[ARGS]", " ".join(sys.argv))
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,EIRMArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, eirm_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, eirm_args = parser.parse_args_into_dataclasses()

    if training_args.local_rank != -1:
        torch.cuda.set_device(training_args.local_rank)

    if "wandb" in (training_args.report_to or []):
        if is_main_process(training_args.local_rank):
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "Comparaison"),
                name=training_args.run_name or f"run-{time.strftime('%Y%m%d-%H%M%S')}",
                dir=training_args.output_dir,
                config=training_args.to_dict(),
                reinit=True,
                )

    nb_steps = data_args.nb_steps

    milestones = None
    if model_args.save_milestones:
        milestones = sorted({int(x) for x in model_args.save_milestones.split(",") if x.strip()})
        # Si nb_steps n'est pas donné, on s'arrête au max des milestones
        if nb_steps is None and len(milestones) > 0:
            nb_steps = max(milestones)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_warning()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)


    train_folder = data_args.train_file
    raw_datasets = {}

    # --- Chargement datasets (pré-tokenisés ou bruts) ---
    if data_args.tokenized_dir is not None:
        tok_root = data_args.tokenized_dir
        if training_args.do_train:
            if data_args.use_erm:
                # --- ERM: un seul environnement 'erm'
                erm_p = os.path.join(tok_root, "erm")
                if not os.path.isdir(erm_p):
                    raise ValueError(f"'erm' introuvable dans {tok_root} (crée-le avec le script de concat).")
                loaded = load_from_disk(erm_p)
                # loaded peut être un Dataset (probable) ou un DatasetDict
                if hasattr(loaded, "keys"):
                    split = "train" if "train" in loaded else list(loaded.keys())[0]
                    raw_datasets["erm"] = loaded[split]
                else:
                    raw_datasets["erm"] = loaded


            else:
                for name in sorted(os.listdir(tok_root)):
                    d = os.path.join(tok_root, name)
                    if not os.path.isdir(d):
                        continue
                    if name in ("ind-validation", "ood-validation", "erm"):
                        continue
                    raw_datasets[name] = load_from_disk(d)  # DatasetDict avec split 'train'

        if training_args.do_eval:
            ind_p = os.path.join(tok_root, "ind-validation")
            ood_p = os.path.join(tok_root, "ood-validation")
            if os.path.isdir(ind_p):
                raw_datasets["ind-validation"] = load_from_disk(ind_p)
            else:
                raise ValueError("ind-validation introuvable dans --tokenized_dir")
            if os.path.isdir(ood_p):
                raw_datasets["ood-validation"] = load_from_disk(ood_p)
            else:
                raise ValueError("ood-validation introuvable dans --tokenized_dir")
    else:
        # Chargement des datasets d'entraînement depuis des .txt bruts
        if training_args.do_train:
            if train_folder is not None:
                for file in os.listdir(train_folder):
                    if file.endswith('.txt'):
                        env_name = file.split(".")[0]
                        data_files = {"train": os.path.join(train_folder, file)}
                        raw_datasets[env_name] = load_dataset("text", data_files=data_files)
            else:
                raise ValueError("Aucun fichier d'entraînement ni dataset n'a été spécifié.")

        # Chargement de la validation (fichiers .txt)
        if training_args.do_eval:
            if data_args.validation_file is not None:
                data_files = {"validation": data_args.validation_file}
                raw_datasets["ind-validation"] = load_dataset("text", data_files=data_files)
            else:
                raise ValueError("Aucun fichier de validation n'est spécifié pour l'évaluation.")

            if data_args.ood_validation_file is not None:
                data_files = {"validation": data_args.ood_validation_file}
                raw_datasets["ood-validation"] = load_dataset("text", data_files=data_files)
            else:
                raise ValueError("Aucun fichier de validation hors distribution n'est spécifié pour l'évaluation.")


    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        if not model_args.model_type:
            raise ValueError("Si --model_name_or_path n'est pas fourni, spécifie --model_type.")
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir, # répertoire où HuggingFace stock les fichiers téléchargés (tokenizer, vocab, etc.).
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if training_args.do_train:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        envs = [k for k in raw_datasets.keys() if 'validation' not in k]

        if 'envs' not in config.to_dict():
            if 'distil' in model_args.model_name_or_path:
                inv_config = InvariantDistilBertConfig(envs=envs, **config.to_dict())
                irm_model = InvariantDistilBertForMaskedLM(inv_config, model)
            else:
                inv_config = InvariantRobertaConfig(envs=envs, **config.to_dict())
                irm_model = InvariantRobertaForMaskedLM(inv_config, model)
        else:
            irm_model = model

    else:
        model_type = getattr(config, "model_type", None)
        if model_type == "invariant-distilbert":
            irm_model = InvariantDistilBertForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

        elif model_type == "invariant-roberta":
            irm_model = InvariantRobertaForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.model_name_or_path,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

        else:
            # fallback si jamais on évalue un checkpoint HF standard
            irm_model = AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    irm_model.resize_token_embeddings(len(tokenizer))

    # Pré-traitement des datasets d'entraînement : tokenisation et regroupement.
    if data_args.tokenized_dir is not None:
        irm_tokenized_datasets = raw_datasets
    else:
        irm_tokenized_datasets = {}
        for env_name, ds in raw_datasets.items():
            if training_args.do_train and 'validation' not in env_name:
                column_names = ds["train"].column_names
            elif training_args.do_eval and 'validation' in env_name:
                column_names = ds["validation"].column_names
            text_column_name = "content" if "content" in column_names else column_names[0]

            # Calcul de max_seq_length pour l'entraînement
            if data_args.max_seq_length is None:
                max_seq_length = tokenizer.model_max_length
                if max_seq_length > 1024:
                    logger.warning(f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). Picking 1024 instead.")
                    max_seq_length = 1024
            else:
                max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

            if data_args.line_by_line:
                # When using line_by_line, we just tokenize each nonempty line.
                padding = "max_length" if data_args.pad_to_max_length else False

                def tokenize_function(examples):
                    # Remove empty lines
                    examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
                    return tokenizer(
                        examples["text"],
                        padding=padding,
                        truncation=True,
                        max_length=max_seq_length,
                        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                        # receives the `special_tokens_mask`.
                        return_special_tokens_mask=True,
                    )

                tokenized_datasets = ds.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=[text_column_name],
                    load_from_cache_file=not data_args.overwrite_cache,
                )
                irm_tokenized_datasets[env_name] = tokenized_datasets

            else:
                def tokenize_function(examples):
                    return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

                tokenized_datasets = ds.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )

                def group_texts(examples):
                    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
                    total_length = len(concatenated_examples[list(examples.keys())[0]])
                    total_length = (total_length // max_seq_length) * max_seq_length
                    result = {
                        k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                        for k, t in concatenated_examples.items()
                    }
                    return result

                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
                irm_tokenized_datasets[env_name] = tokenized_datasets

    # Data collator pour MLM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)

    # --- Normalise tous les environnements pour avoir un split "train" ---
    _normalized = {}
    for env_name, ds in irm_tokenized_datasets.items():
        if isinstance(ds, DatasetDict):
            _normalized[env_name] = ds
        elif isinstance(ds, Dataset):
            _normalized[env_name] = DatasetDict({"train": ds})
        else:
            raise TypeError(f"{env_name}: type de dataset non supporté ({type(ds)}).")
    irm_tokenized_datasets = _normalized

    train_tokenized_datasets = {k: v for k, v in irm_tokenized_datasets.items() if 'ind-validation' not in k and 'ood-validation' not in k}
    eval_ind_tokenized_datasets = None
    eval_ood_tokenized_datasets = None

    if training_args.do_eval:
        def _get_validation_split(ds_or_dict):
            if hasattr(ds_or_dict, "keys"):
                keys = list(ds_or_dict.keys())
                split = "validation" if "validation" in keys else keys[0]
                return ds_or_dict[split]
            return ds_or_dict

        if "ind-validation" not in raw_datasets:
            raise ValueError("`--do_eval` est actif mais 'ind-validation' est absent de --tokenized_dir.")

        eval_ind_tokenized_datasets = _get_validation_split(raw_datasets["ind-validation"])
        if "ood-validation" in raw_datasets:
            eval_ood_tokenized_datasets = _get_validation_split(raw_datasets["ood-validation"])

        def _subsample(ds, frac, seed):
            if ds is None or not (0.0 < frac < 1.0):
                return ds
            n = max(1, int(frac * len(ds)))
            return ds.shuffle(seed=seed).select(range(n))

        seed_eval = data_args.eval_seed if data_args.eval_seed is not None else training_args.seed
        eval_ind_tokenized_datasets = _subsample(eval_ind_tokenized_datasets, data_args.eval_fraction, seed_eval)
        eval_ood_tokenized_datasets = _subsample(eval_ood_tokenized_datasets, data_args.eval_fraction, seed_eval)

    _backend = getattr(training_args, "half_precision_backend",
               getattr(training_args, "fp16_backend", None))
    print("fp16 demandé :", training_args.fp16, "backend :", _backend)


    # Initialisation du Trainer
    trainer = InvariantTrainer(
        model=irm_model,
        args=training_args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=eval_ind_tokenized_datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eirm_args=eirm_args,
    )


    if training_args.do_train:
        # --- checkpoints (comme avant) ---
        if last_checkpoint is not None:
            check_point = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            check_point = model_args.model_name_or_path
        else:
            check_point = None
            warnings.warn("No checkpoint found. Training from scratch.")

        # Si tu veux que EIRM respecte nb_steps, on le pousse dans HF TrainingArguments
        if 'nb_steps' in globals() and nb_steps is not None:
            training_args.max_steps = nb_steps

        # --- dispatch EIRM vs modes existants ---
        if eirm_args.eirm_mode.lower() in {"f-irm", "v-irm"}:
            if nb_steps is not None:
                training_args.max_steps = nb_steps
            # EIRM prend la priorité si activé
            train_result = trainer.train_eirm(
            training_set=train_tokenized_datasets,
            nb_steps=nb_steps,
            nb_steps_heads_saving=model_args.nb_steps_heads_saving,
            nb_steps_model_saving=model_args.nb_steps_model_saving,
            resume_from_checkpoint=check_point,
            save_milestones=set(milestones) if milestones else None,
        )
            # (Option) si tu veux gérer un resume pour EIRM, on pourra étendre train_eirm(resume_from_checkpoint=check_point)
        else:
            # Ton comportement historique
            if model_args.mode == "ilm":
                train_result = trainer.invariant_train(
                    training_set=train_tokenized_datasets,
                    nb_steps=nb_steps,
                    nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                    nb_steps_model_saving=model_args.nb_steps_model_saving,
                    resume_from_checkpoint=check_point,
                    save_milestones=set(milestones) if milestones else None,
                )
            else:
                # fallback si jamais un autre mode apparait
                train_result = trainer.train(resume_from_checkpoint=check_point)

        # --- sauvegardes (inchangées) ---
        output_dir = training_args.output_dir
        trainer.model.save_pretrained(output_dir, safe_serialization=False)  # modèle
        tokenizer.save_pretrained(output_dir)  # tokenizer
        trainer.save_state()

        if trainer.is_world_process_zero() and wandb.run:
            wandb.finish()

    # --- Évaluation du modèle courant (point "final") ---
    if training_args.do_eval and not training_args.do_train:
        def _eval_and_log(tag, dataset):
            if dataset is None:
                return
            metrics = trainer.evaluate(eval_dataset=dataset)
            loss = float(metrics["eval_loss"])
            ppl = math.exp(loss)
            if trainer.is_world_process_zero():
                print(f"[EVAL] {tag}: loss={loss:.4f}  ppl={ppl:.2f}")
            if wandb.run:
                wandb.log({f"evaluation/{tag}_loss": loss, f"evaluation/{tag}_perplexity": ppl})

        _eval_and_log("ind", eval_ind_tokenized_datasets)
        _eval_and_log("ood", eval_ood_tokenized_datasets)


    if model_args.evaluate_checkpoints_after_train:
        ckpts = sorted(
            glob.glob(os.path.join(training_args.output_dir, "model-*")),
            key=lambda p: int(re.search(r"model-(\d+)$", p).group(1))
        )
        iterator = tqdm(ckpts, desc="Évaluation des checkpoints") if trainer.is_world_process_zero() else ckpts
        for checkpoint_path in iterator:
            step = int(os.path.basename(checkpoint_path).split("-")[-1])
            model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)
            model.to(training_args.device); model.eval()
            trainer.model = model
            ind_output = trainer.evaluate(eval_dataset=eval_ind_tokenized_datasets)
            ind_loss = float(ind_output["eval_loss"]); ind_ppl = math.exp(ind_loss)
            ood_ppl = None
            if eval_ood_tokenized_datasets is not None:
                ood_output = trainer.evaluate(eval_dataset=eval_ood_tokenized_datasets)
                ood_loss = float(ood_output["eval_loss"]); ood_ppl = math.exp(ood_loss)
            if trainer.is_world_process_zero():
                wandb.log({"evaluation/ind_perplexity": ind_ppl,
                            **({"evaluation/ood_perplexity": ood_ppl} if ood_ppl is not None else {})},
                            step=step)

        # GC pour éviter la montée mémoire entre checkpoints
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if wandb.run:
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()