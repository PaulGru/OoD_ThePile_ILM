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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset

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
    # Trainer,
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

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
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
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
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
        metadata={
            "help": "Whether to train the heads as an ensemble instead of following the IRM-games dynamics"}
    )
    nb_steps_heads_saving: Optional[int] = field(
        default=0,
        metadata={"help": "Number of training steps between saving the head weights (if 0, the heads are not saved regularly)."},
    )
    nb_steps_model_saving: Optional[int] = field(
        default=0,
        metadata={
            "help": "Number of training steps between saving the full model (if 0, the heads are not saved regularly)."},
    )
    do_lower_case: Optional[bool] = field(
        default=True,
        metadata={"help": "Lower-case during tokenization."},
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Taux de dropout pour le modèle."}
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "Taux de dropout pour l'attention."}
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
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    nb_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of training steps."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        # else:
        #     continue

            # if self.train_file is not None:
            #     extension = self.train_file.split(".")[-1]
            #     assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            # if self.validation_file is not None:
            #     extension = self.validation_file.split(".")[-1]
            #     assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    nb_steps = data_args.nb_steps
    #training_args.local_rank = -1  # Force explicitement le local_rank à -1 (pas de distributed)

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
        level=logging.WARNING
    )
    logger.setLevel(logging.ERROR)

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_warning()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    #logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    

    if data_args.train_file is not None:
        
        if os.path.isfile(data_args.train_file):
            # eLM : 1 seul fichier (all_train.txt)
            data_files = {"train": data_args.train_file}
            dataset = load_dataset("text", data_files=data_files)
            irm_datasets = {"all": dataset}
            #envs_list = ["all"]
        
        else:
            # Cas multi-environnements : plusieurs fichiers dans un dossier
            irm_folder = data_args.train_file
            irm_datasets = {}
            for file in os.listdir(irm_folder):
                if file.endswith('.txt'):
                    env_name = file.split(".")[0]
                    if env_name.startswith("val_"):
                        continue
                    data_files = {"train": os.path.join(irm_folder, file)}
                    #data_files = {}
                    #data_files["train"] = os.path.join(irm_folder, file)
                    datasets = load_dataset("text", data_files=data_files)
                    irm_datasets[env_name] = datasets
                #envs_list = list(irm_datasets.keys())


    elif data_args.dataset_name is not None:
            
        # Charger le dataset directement depuis Hugging Face
        train_dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split="train",
            # streaming=True  # Optionnel : utile si le dataset est très volumineux
        )

        irm_datasets = {"train": train_dataset}
        #envs_list = ["train"]

    else:
        raise ValueError("Aucun fichier d'entraînement ni dataset n'a été spécifié.")

    if data_args.validation_file is not None:
        data_files = {}
        data_files["validation"] = data_args.validation_file
        eval_datasets = load_dataset("text", data_files=data_files)
        irm_datasets['validation-file'] = eval_datasets

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
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
    

    #config.envs = envs_list

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
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


    envs = [k for k in irm_datasets.keys() if 'validation' not in k]

    if len(envs) > 1:
        # Plusieurs environnements : on souhaite utiliser iLM, mtLM ou ensLM avec multi-têtes.
        if 'envs' not in config.to_dict():
            if 'distil' in model_args.model_name_or_path:
                config_dict = config.to_dict()
                config_dict.pop("envs", None)
                inv_config = InvariantDistilBertConfig(envs=envs, **config_dict)
                irm_model = InvariantDistilBertForMaskedLM(inv_config, model)
            else:
                config_dict = config.to_dict()
                config_dict.pop("envs", None)
                inv_config = InvariantRobertaConfig(envs=envs, **config_dict)
                irm_model = InvariantRobertaForMaskedLM(inv_config, model)
        else:
            irm_model = model
    else:
        # Cas eLM : un seul environnement, donc on n'enveloppe pas le modèle.
        irm_model = model

    irm_model.resize_token_embeddings(len(tokenizer))


    # Freeze la couche d'embedding et les 4 premières couches de l'encodeur DistilBert
    if hasattr(irm_model.encoder, "embeddings"):
        for param in irm_model.encoder.embeddings.parameters():
            param.requires_grad = False
        print("Le freeze a été appliqué à la couche d'embedding")
    else:
        print("La couche d'embedding n'a pas été gelée.")

    if hasattr(irm_model.encoder, "transformer") and hasattr(irm_model.encoder.transformer, "layer"):
        for layer in irm_model.encoder.transformer.layer[:4]:
            for param in layer.parameters():
                param.requires_grad = False
        
        print("Le freeze a été appliqué aux 4 premières couches du modèle")
    
    else:
        print("Les couches n'ont pas été gelées.")

    
    if model_args.init_head:
        irm_model.init_head()
    if model_args.init_base:
        irm_model.init_base()

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    irm_tokenized_datasets = {}


    for env_name, datasets in irm_datasets.items():
        # En mode évaluation seule, ne traiter que l'environnement de validation
        if not training_args.do_train and "validation" not in env_name:
            continue

        if training_args.do_train and 'validation' not in env_name:
            if isinstance(datasets, dict):
                column_names = datasets["train"].column_names
            else:
                # Si la clé "train" n'existe pas, on suppose que l'objet datasets est déjà un Dataset.
                column_names = datasets.column_names

        elif training_args.do_eval and 'validation' in env_name:
            column_names = datasets["validation"].column_names
        text_column_name = "content" if "content" in column_names else column_names[0]


        if data_args.max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
            if max_seq_length > 1024:
                logger.warn(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
                )
                max_seq_length = 1024
        else:
            if data_args.max_seq_length > tokenizer.model_max_length:
                logger.warn(
                    f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
            
        
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=False)

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        irm_tokenized_datasets[env_name] = tokenized_datasets

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)

    train_tokenized_datasets = {k: v for k, v in irm_tokenized_datasets.items() if not('validation-file' in k)}
    eval_tokenized_datasets = irm_tokenized_datasets['validation-file']['validation']

    # Initialize our Trainer
    trainer = InvariantTrainer(
        model=irm_model,
        args=training_args,
        # train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=eval_tokenized_datasets if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:

        if model_args.mode == "mtLM":
            logger.info("TRAINING WITH DIVERSITY -- NOT FOLLOWING IRM-GAMES DYNAMIC")
            train_result = trainer.multitask_train(training_set=train_tokenized_datasets,
                                                   nb_steps=nb_steps,
                                                   nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                                                   nb_steps_model_saving=model_args.nb_steps_model_saving,
                                                   num_train_epochs=training_args.num_train_epochs,
                                                   )

        if model_args.mode == "ensLM":
            logger.info("TRAINING WITH ENSEMBLE -- NOT FOLLOWING IRM-GAMES DYNAMIC")
            train_result = trainer.ensemble_train(training_set=train_tokenized_datasets,
                                                   nb_steps=nb_steps,
                                                   nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                                                   nb_steps_model_saving=model_args.nb_steps_model_saving,
                                                   num_train_epochs=training_args.num_train_epochs,
                                                   )
        if model_args.mode == "iLM":
            train_result = trainer.invariant_train(training_set=train_tokenized_datasets,
                                                    nb_steps=nb_steps,
                                                    nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                                                    nb_steps_model_saving=model_args.nb_steps_model_saving,
                                                    num_train_epochs=training_args.num_train_epochs,
                                                    )
        
        output_dir = training_args.output_dir  # ou votre répertoire de sortie
        trainer.model.save_pretrained(output_dir, safe_serialization=False)
        trainer.tokenizer.save_pretrained(output_dir)


        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")

            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

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