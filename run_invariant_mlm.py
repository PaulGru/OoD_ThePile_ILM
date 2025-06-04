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
import wandb
import glob
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
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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
    nb_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of training steps."}
    )
    ood_validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Fichier de validation Out-Of-Distribution (.txt) jamais vu à l'entraînement"}
    )

    def __post_init__(self):
        if self.train_file is None:
            if self.validation_file is None:
                raise ValueError("Aucun fichier d'entraînement ni dataset n'a été spécifié.")

# Fonction de regroupement des textes
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

    wandb.init(
        project="invariant-language-modeling",
        name=training_args.run_name,
        config={
            "learning_rate": training_args.learning_rate,
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "nb_steps": data_args.nb_steps
        }
    )
           
    nb_steps = data_args.nb_steps
    #training_args.local_rank = -1

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

    train_folder = data_args.train_file
    datasets = {}
    # Chargement des datasets d'entraînement
    if training_args.do_train:
        if train_folder is not None:
            if os.path.isdir(train_folder): # on vérifie si le chemin mène à un répertoire
                # iLM prend plusieurs environnements, on charge les fichiers d'entraînement par environnement
                print("Contenu de train_env :", os.listdir(train_folder)) # liste les éléments du dossier
                
                for file in os.listdir(train_folder):
                    if file.endswith('.txt'):
                        env_name = file.split(".")[0]
                        data_files = {"train": os.path.join(train_folder, file)}
                        datasets[env_name] = load_dataset("text", data_files=data_files)
                     
            else: # si c'est un fichier unique, c'est pour eLM
                data_files = {"train": data_args.train_file}
                dataset = load_dataset("text", data_files=data_files)
                datasets = {"all_train": dataset}
        else:
            raise ValueError("Aucun fichier d'entraînement ni dataset n'a été spécifié.")

    # Chargement de la validation depuis le dossier "val_env"
    if training_args.do_eval:
        if data_args.validation_file is not None:
            data_files = {"validation": data_args.validation_file}
            datasets["validation-file"] = load_dataset("text", data_files=data_files)
        else:
            raise ValueError("Aucun fichier de validation n'est spécifié pour l'évaluation.")
    
        # Chargement de la validation Out-of-Distribution
        if data_args.ood_validation_file is not None:
            data_files = {"validation": data_args.ood_validation_file}
            datasets["ood-validation"] = load_dataset("text", data_files=data_files)
   
    # Configuration du modèle et du tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir, # répertoire où HuggingFace stock les fichiers téléchargés (tokenizer, vocab, etc.).
        "use_fast": model_args.use_fast_tokenizer,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    
    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
 
    envs = [k for k in datasets.keys() if 'validation' not in k]
    
    if 'envs' not in config.to_dict():
        if model_args.model_name_or_path:
            inv_config = InvariantDistilBertConfig(envs=envs, **config.to_dict())
            irm_model = InvariantDistilBertForMaskedLM(inv_config, model)
        else:
            raise ValueError("Modèle inconnu")
    else:
        irm_model = model
    
    irm_model.resize_token_embeddings(len(tokenizer))

    # Pré-traitement des datasets d'entraînement : tokenisation et regroupement.
    irm_tokenized_datasets = {}
    for env_name, datasets in datasets.items():
        if training_args.do_train and 'validation' not in env_name:
            column_names = datasets["train"].column_names
        elif training_args.do_eval and 'validation' in env_name:
            column_names = datasets["validation"].column_names
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
        irm_tokenized_datasets[env_name] = tokenized_datasets

    print("Clés d'irm_tokenized_datasets :", list(irm_tokenized_datasets.keys()))

    # Data collator pour MLM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)
    
    train_tokenized_datasets = {k: v for k, v in irm_tokenized_datasets.items() if 'validation-file' not in k and 'ood-validation' not in k}
    eval_tokenized_datasets = irm_tokenized_datasets['validation-file']['validation']
    eval_ood_tokenized_datasets = irm_tokenized_datasets["ood-validation"]["validation"]

    # Initialisation du Trainer
    trainer = InvariantTrainer(
        model=irm_model,
        args=training_args,
        eval_dataset=eval_tokenized_datasets if training_args.do_eval else None,
        data_collator=data_collator,
    )

    if training_args.do_train:
        if model_args.irm_games_mode == "full":
            train_result = trainer.invariant_train_games(
                training_set=train_tokenized_datasets,
                nb_steps=nb_steps,
                nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                nb_steps_model_saving=model_args.nb_steps_model_saving,
                num_train_epochs=training_args.num_train_epochs,
                update_phi_every_k=model_args.update_phi_every_k
            )
        else:  # mode simplifié
            train_result = trainer.invariant_train(
                training_set=train_tokenized_datasets,
                nb_steps=nb_steps,
                nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                nb_steps_model_saving=model_args.nb_steps_model_saving,
                num_train_epochs=training_args.num_train_epochs,
            )

        output_dir = training_args.output_dir
        trainer.model.save_pretrained(output_dir, safe_serialization=False) # sauvegarde le modèle
        tokenizer.save_pretrained(output_dir) # sauvegarde le tokenizer

        if trainer.is_world_process_zero():
            logger.info("***** Train results *****")
            for key, value in sorted(train_result["metrics"].items()):
                logger.info(f"  {key} = {value}")

            wandb.log({f"final/{key}": value for key, value in train_result["metrics"].items()})
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
    

    # Préparer l'évaluation OOD (à faire AVANT cette boucle)
    ood_eval_dataset = eval_ood_tokenized_datasets  # <- assure-toi que ce dataset a été préparé !

    if trainer.is_world_process_zero():
        checkpoints = sorted(glob.glob(os.path.join(training_args.output_dir, "model-*")))

        for checkpoint_path in checkpoints:
            step = int(checkpoint_path.split("-")[-1])

            # Recharger le modèle depuis le checkpoint
            model = InvariantDistilBertForMaskedLM.from_pretrained(checkpoint_path)
            trainer.model = model  # Mise à jour du modèle

            # Évaluation In-Distribution
            eval_output = trainer.evaluate(eval_dataset=eval_tokenized_datasets)
            eval_loss = eval_output["eval_loss"]
            perplexity = math.exp(eval_loss)

            wandb.log({
                "eval_in/loss": eval_loss,
                "eval_in/perplexity": perplexity,
                "eval_in/step": step
            })

            # Évaluation OOD
            if ood_eval_dataset is not None:
                ood_output = trainer.evaluate(eval_dataset=ood_eval_dataset)
                ood_loss = ood_output["eval_loss"]
                ood_perplexity = math.exp(ood_loss)

                wandb.log({
                    "eval_ood/loss": ood_loss,
                    "eval_ood/perplexity": ood_perplexity,
                    "eval_ood/step": step
                })

        
    if trainer.is_world_process_zero():
        if wandb.run:
            wandb.finish()

    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
