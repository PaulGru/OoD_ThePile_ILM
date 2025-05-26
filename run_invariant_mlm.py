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
import logging
import math
import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm 

from invariant_trainer import InvariantTrainer

from invariant_roberta import InvariantRobertaForMaskedLM, InvariantRobertaConfig
from invariant_distilbert import InvariantDistilBertForMaskedLM, InvariantDistilBertConfig
from invariant_xlmroberta import InvariantXLMRobertaForMaskedLM, InvariantXLMRobertaConfig

from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer

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
CONFIG_MAPPING.update({'invariant-xlm-roberta': InvariantXLMRobertaConfig})

MODEL_FOR_MASKED_LM_MAPPING.update({InvariantDistilBertConfig: InvariantDistilBertForMaskedLM})
MODEL_FOR_MASKED_LM_MAPPING.update({InvariantRobertaConfig: InvariantRobertaForMaskedLM})
MODEL_FOR_MASKED_LM_MAPPING.update({InvariantXLMRobertaConfig: InvariantXLMRobertaForMaskedLM})

TOKENIZER_MAPPING.update({InvariantDistilBertConfig: (DistilBertTokenizer, DistilBertTokenizerFast)})
TOKENIZER_MAPPING.update({InvariantRobertaConfig: (RobertaTokenizer, RobertaTokenizerFast)})
TOKENIZER_MAPPING.update({InvariantXLMRobertaConfig: (XLMRobertaTokenizer, XLMRobertaTokenizerFast)})

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
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
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
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script with private models)."
        },
    )
    init_head: Optional[bool] = field(
        default=False,
        metadata={"help": "Re-initialize the language modeling heads to random weights before training"}
    )
    init_base: Optional[bool] = field(
        default=False,
        metadata={"help": "Re-initialize the base language model (and thus the language modeling heads) before training"}
    )
    mode: Optional[str] = field(
        default="iLM",
        metadata={"help": "Whether to train the heads as an ensemble instead of following the IRM-games dynamics"}
    )
    nb_steps_heads_saving: Optional[int] = field(
        default=0,
        metadata={"help": "Number of training steps between saving the head weights (if 0, the heads are not saved regularly)."},
    )
    nb_steps_model_saving: Optional[int] = field(
        default=0,
        metadata={"help": "Number of training steps between saving the full model (if 0, the heads are not saved regularly)."},
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Taux de dropout pour le modèle."}
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "Taux de dropout pour l'attention."}
    )
    irm_games_mode: Optional[str] = field(
        default="simplified",
        metadata={
            "help": "Choix entre 'simplified' pour update phi à chaque batch (ILM) ou 'full' pour IRM-Games complet."
        }
    )
    update_phi_every_k: Optional[int] = field(
        default=5,  # Valeur par défaut raisonnable
        metadata={
            "help": "Nombre d'updates des têtes w^e avant une mise à jour du backbone phi dans l'entraînement IRM-Games."
        }
    )
    do_bias_eval_on_wikitext: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, run bias evaluation using Wikitext-2 val_ood after training."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file or a directory)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input validation data file or directory (a text file or directory)."},
    )
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
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "Whether to pad all samples to `max_seq_length`."}
    )
    nb_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of training steps."}
    )
    eval_type: Optional[str] = field(
        default="ind",
        metadata={"help": "Type d'évaluation à utiliser: 'ind' pour InD ou 'ood' pour OoD"}
    )

    def __post_init__(self):
        if self.train_file is None and self.dataset_name is None:
            if self.validation_file is None:
                raise ValueError("Aucun fichier d'entraînement ni dataset n'a été spécifié.")

# Fonction d'ordre supérieur pour créer la fonction de regroupement des textes
def create_group_texts(max_seq_length):
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result
    return group_texts

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    nb_steps = data_args.nb_steps
    #training_args.local_rank = -1

    # Force local_rank à -1 si non défini (on n'utilise pas le training distribué)
    #if training_args.local_rank is None:
    #    training_args.local_rank = -1

    # Détection du dernier checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.WARNING
    )
    logger.setLevel(logging.ERROR)

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_warning()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    set_seed(training_args.seed)

    # Chargement des datasets d'entraînement
    if training_args.do_train:
        if data_args.train_file is not None:
            # data_args.train_file doit pointer vers le dossier "train_env"
            if os.path.isdir(data_args.train_file):
                train_folder = data_args.train_file
                print("Contenu de train_env :", os.listdir(train_folder))
                train_datasets = {}
                
                if model_args.mode == "eLM":
                    # eLM doit avoir exactement un environnement global
                    all_train_path = os.path.join(train_folder, "all_train.txt")
                    if os.path.exists(all_train_path):
                        data_files = {"train": all_train_path}
                        dataset = load_dataset("text", data_files=data_files)
                        train_datasets = {"all_train": dataset}
                    else:
                        # Fusion explicite de tous les fichiers d'environnements
                        merged_file_path = os.path.join(train_folder, "merged_all_train.txt")
                        with open(merged_file_path, 'w', encoding='utf-8') as outfile:
                            for fname in os.listdir(train_folder):
                                if fname.endswith('.txt'):
                                    with open(os.path.join(train_folder, fname), encoding='utf-8') as infile:
                                        for line in infile:
                                            outfile.write(line)
                        dataset = load_dataset("text", data_files={"train": merged_file_path})
                        train_datasets = {"all_train": dataset}
                else:
                    # iLM/ensLM : chargement normal par environnement
                    for file in os.listdir(train_folder):
                        if file.endswith('.txt'):
                            env_name = file.split(".")[0]
                            if env_name == "all_train":
                                continue
                            data_files = {"train": os.path.join(train_folder, file)}
                            train_datasets[env_name] = load_dataset("text", data_files=data_files)
                        
            else:
                data_files = {"train": data_args.train_file}
                dataset = load_dataset("text", data_files=data_files)
                train_datasets = {"all_train": dataset}
        else:
            raise ValueError("Aucun fichier d'entraînement ni dataset n'a été spécifié.")


        # Définition de la variable envs à partir des clés de train_datasets.
        if model_args.mode == "eLM":
            envs = ["all_train"]  # Toujours un unique environnement explicite en eLM
        else:
            envs = list(train_datasets.keys())

    else:
        train_datasets = {}
        # Mode évaluation uniquement ; définir envs à une valeur par défaut.
        envs = []  # ou éventuellement envs = ['default'] si votre modèle en a besoin

    # Chargement de la validation depuis le dossier "val_env"
    if training_args.do_eval:
        if data_args.validation_file is not None:
            if os.path.isdir(data_args.validation_file):
                val_folder = data_args.validation_file
                eval_datasets = {}
                for file in os.listdir(val_folder):
                    if file.endswith('.txt'):
                        env_name = file.split(".")[0]  # attend "val_ind" ou "val_ood"
                        data_files = {"validation": os.path.join(val_folder, file)}
                        eval_datasets[env_name] = load_dataset("text", data_files=data_files)
            else:
                data_files = {"validation": data_args.validation_file}
                eval_datasets = {"validation-file": load_dataset("text", data_files=data_files)}
        else:
            raise ValueError("Aucun fichier de validation n'est spécifié pour l'évaluation.")

        # Sélection du dataset de validation en fonction du paramètre eval_type
        eval_key = f"val_{data_args.eval_type}"
        if eval_key in eval_datasets:
            selected_eval_dataset = eval_datasets[eval_key]["validation"]
        else:
            raise ValueError(f"Aucun dataset de validation correspondant à {eval_key} n'a été trouvé dans {data_args.validation_file}")
    else:
        selected_eval_dataset = None

    # Configuration du modèle et du tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError("You are instantiating a new tokenizer from scratch. This is not supported by this script.")

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)
    
    model.config.dropout = 0.25
    model.config.attention_dropout = 0.25

    if len(envs) > 1:
        if 'envs' not in config.to_dict():

            if model_args.model_type == "invariant-distilbert":
                config_dict = config.to_dict()
                config_dict.pop("envs", None)
                inv_config = InvariantDistilBertConfig(envs=envs, **config_dict)
                irm_model = InvariantDistilBertForMaskedLM(inv_config, model)

            elif model_args.model_type == "invariant-xlm-roberta":
                config_dict = config.to_dict()
                config_dict.pop("envs", None)
                inv_config = InvariantXLMRobertaConfig(envs=envs, **config_dict)
                irm_model = InvariantXLMRobertaForMaskedLM(inv_config, model)

            elif model_args.model_type == "invariant-roberta":
                config_dict = config.to_dict()
                config_dict.pop("envs", None)
                inv_config = InvariantRobertaConfig(envs=envs, **config_dict)
                irm_model = InvariantRobertaForMaskedLM(inv_config, model)

            else:
                raise ValueError(f"Unknown model_type: {model_args.model_type}")

        else:
            irm_model = model
    else:
        irm_model = model

    irm_model.resize_token_embeddings(len(tokenizer))

    # Vérification du nombre de têtes du modèle (important pour le mode eLM)
    num_heads = len(irm_model.lm_heads) if hasattr(irm_model, 'lm_heads') else "Standard (une tête)"
    print(f"[DEBUG] Nombre de têtes du modèle en mode {model_args.mode} : {num_heads}")

    # AJOUT : enregistrement des hooks de débogage pour identifier les NaNs
    #from hooks import register_nan_forward_hooks, register_nan_backward_hooks
    #register_nan_forward_hooks(irm_model)
    #register_nan_backward_hooks(irm_model)

    # Pré-traitement des datasets d'entraînement : tokenisation et regroupement.
    irm_tokenized_train = {}
    for env_name, datasets in train_datasets.items():
        if isinstance(datasets, dict):
            column_names = datasets["train"].column_names
        else:
            column_names = datasets.column_names
        text_column_name = "content" if "content" in column_names else column_names[0]
        
        # Calcul de max_seq_length pour l'entraînement
        if data_args.max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
            if max_seq_length > 1024:
                logger.warn(f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). Picking 1024 instead.")
                max_seq_length = 1024
        else:
            max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=False)
        
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        # Création de la fonction group_texts avec max_seq_length capturé
        group_texts_fn = create_group_texts(max_seq_length)
        tokenized_datasets = tokenized_datasets.map(
            group_texts_fn,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        irm_tokenized_train[env_name] = tokenized_datasets

    print("Clés d'irm_tokenized_train :", list(irm_tokenized_train.keys()))

    # Pré-traitement du dataset d'évaluation
    if training_args.do_eval and selected_eval_dataset is not None:
        column_names = selected_eval_dataset.column_names
        text_column_name = "content" if "content" in column_names else column_names[0]

        def tokenize_function_eval(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=False)
        
        tokenized_eval_dataset = selected_eval_dataset.map(
            tokenize_function_eval,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        if data_args.max_seq_length is None:
            eval_max_seq_length = tokenizer.model_max_length
            if eval_max_seq_length > 1024:
                logger.warn(f"eval_max_seq_length élevé ({tokenizer.model_max_length}), on limite à 1024.")
                eval_max_seq_length = 1024
        else:
            eval_max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        
        group_texts_eval_fn = create_group_texts(eval_max_seq_length)
        tokenized_eval_dataset = tokenized_eval_dataset.map(
            group_texts_eval_fn,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        tokenized_eval_dataset = None

    # Data collator pour MLM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)
    
    # Initialisation du Trainer
    trainer = InvariantTrainer(
        model=irm_model,
        args=training_args,
        eval_dataset=tokenized_eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
    )

    # Entraînement
    if training_args.do_train:
        if model_args.mode == "eLM":
            print("Entraînement eLM (fine tuning classique)")
            train_result = trainer.empirical_train(
                training_set=irm_tokenized_train,  # dans ce cas, training_set devrait contenir une unique clé "all_train"
                nb_steps=data_args.nb_steps,
                num_train_epochs=training_args.num_train_epochs,
            )
        elif model_args.mode == "mtLM":
            print("TRAINING WITH DIVERSITY -- NOT FOLLOWING IRM-GAMES DYNAMIC")
            train_result = trainer.multitask_train(
                training_set=irm_tokenized_train,
                nb_steps=nb_steps,
                nb_steps_model_saving=model_args.nb_steps_model_saving,
                num_train_epochs=training_args.num_train_epochs,
            )
        elif model_args.mode == "ensLM":
            print("TRAINING WITH ENSEMBLE -- NOT FOLLOWING IRM-GAMES DYNAMIC")
            train_result = trainer.ensemble_train(
                training_set=irm_tokenized_train,
                nb_steps=nb_steps,
                nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                nb_steps_model_saving=model_args.nb_steps_model_saving,
                num_train_epochs=training_args.num_train_epochs,
            )
        elif model_args.mode == "iLM":
            print("TRAINING WITH INVARIANCE -- FOLLOWING IRM-GAMES DYNAMIC")
            if model_args.irm_games_mode == "full":
                train_result = trainer.invariant_train_games(
                    training_set=irm_tokenized_train,
                    nb_steps=nb_steps,
                    nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                    nb_steps_model_saving=model_args.nb_steps_model_saving,
                    num_train_epochs=training_args.num_train_epochs,
                    update_phi_every_k=model_args.update_phi_every_k  # si tu ajoutes ce paramètre
                )
            else:  # mode simplifié
                train_result = trainer.invariant_train(
                    training_set=irm_tokenized_train,
                    nb_steps=nb_steps,
                    nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                    nb_steps_model_saving=model_args.nb_steps_model_saving,
                    num_train_epochs=training_args.num_train_epochs,
                )

        output_dir = training_args.output_dir
        trainer.model.save_pretrained(output_dir, safe_serialization=False)
        tokenizer.save_pretrained(output_dir)
        
        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Évaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Charger le meilleur modèle sauvegardé (celui avec la meilleure eval loss sur InD)
        best_model_path = os.path.join(training_args.output_dir, "best_model")
        if os.path.isdir(best_model_path):
            print("Rechargement du meilleur modèle sauvegardé pour l'évaluation.")

            if model_args.mode == "eLM":
                # Si c'est eLM (standard fine-tuning), on recharge normalement
                best_model = AutoModelForMaskedLM.from_pretrained(best_model_path)
            else:
                # Sinon on recharge un modèle invariant spécifique
                if model_args.model_type == "invariant-distilbert":
                    best_model = InvariantDistilBertForMaskedLM.from_pretrained(best_model_path)
                elif model_args.model_type == "invariant-roberta":
                    best_model = InvariantRobertaForMaskedLM.from_pretrained(best_model_path)
                elif model_args.model_type == "invariant-xlm-roberta":
                    best_model = InvariantXLMRobertaForMaskedLM.from_pretrained(best_model_path)
                else:
                    raise ValueError(f"Unknown invariant model_type: {model_args.model_type}")

            best_model.to(training_args.device)
            trainer.model = best_model  # Remplacer le modèle courant par le meilleur modèle
        else:
            print("Aucun modèle 'best_model' trouvé, on utilise le modèle final.")
            best_model = trainer.model
            best_model.to(training_args.device)
        

        # Création du DataLoader pour l'évaluation. On utilise ici la taille de batch définie pour l'évaluation.
        eval_dataloader = DataLoader(
            tokenized_eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=data_collator
        )

        # Passage du modèle en mode evaluation
        trainer.model.eval()
        total_eval_loss = 0.0
        nb_eval_steps = 0

        # Boucle d'évaluation avec torch.no_grad() et AMP
        with torch.no_grad():
            pbar = tqdm(eval_dataloader, desc="Évaluation", unit="batch")
            for batch in pbar:
                # Déplacement des tensors sur le device
                batch = {k: v.to(training_args.device) for k, v in batch.items()}
                # Calcul en mode AMP
                with torch.amp.autocast("cuda"):
                    outputs = best_model(**batch)
                    loss = outputs.loss
                total_eval_loss += loss.item()
                nb_eval_steps += 1
                # Mise à jour de la barre de progression avec la loss du batch courant
                pbar.set_postfix(loss=f"{loss.item():.4f}")


        # Calcul de la loss moyenne et de la perplexité
        avg_eval_loss = total_eval_loss / nb_eval_steps if nb_eval_steps > 0 else 0.0
        # Pour éviter des valeurs aberrantes, on teste si la loss moyenne est raisonnable pour calculer l'exponentielle
        perplexity = math.exp(avg_eval_loss) if avg_eval_loss < 100 else float('inf')
        results = {"eval_loss": avg_eval_loss, "perplexity": perplexity}


        print("***** Eval results *****")
        for key, value in sorted(results.items()):
            print(f"{key} = {value:.4f}")

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_mlm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
    return results

if __name__ == "__main__":
    main()
