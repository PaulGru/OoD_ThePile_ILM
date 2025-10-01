#!/usr/bin/env python
import os, re, glob, math, argparse, json
import torch
import wandb

# importe tes classes custom pour qu'AutoModel sache les (de)serializer
from invariant_roberta import InvariantRobertaForMaskedLM, InvariantRobertaConfig  # noqa: F401
from invariant_distilbert import InvariantDistilBertForMaskedLM, InvariantDistilBertConfig  # noqa: F401

from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling, TrainingArguments)
from invariant_trainer import InvariantTrainer

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tokenized_dir", required=True)
    ap.add_argument("--steps", required=True, help="Ex: 2500,5000,25000,50000")
    ap.add_argument("--per_device_eval_batch_size", type=int, default=128)
    ap.add_argument("--fp16_full_eval", action="store_true")
    ap.add_argument("--max_eval_samples", type=int, default=None,
                    help="Évaluer sur les N premiers exemples (accélère fortement).")
    ap.add_argument("--eval_fraction", type=float, default=None)
    ap.add_argument("--eval_seed", type=int, default=42)
    ap.add_argument("--run_name", default="eval_milestones")
    return ap.parse_args()

def random_subset(ds, max_samples=None, fraction=None, seed=42):
    """Retourne un sous-ensemble aléatoire et reproductible."""
    if fraction is not None:
        n = max(1, int(len(ds) * float(fraction)))
    elif max_samples is not None:
        n = min(int(max_samples), len(ds))
    else:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def main():
    args = parse_args()
    out = args.output_dir
    tok_root = args.tokenized_dir

    os.environ.setdefault("WANDB_MODE", "offline")
    wandb_dir = os.path.join(out, "wandb_eval")
    os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(project=os.environ.get("WANDB_PROJECT","Comparaison"),
               name=args.run_name, dir=wandb_dir, reinit=True)

    # Datasets
    ind = load_from_disk(os.path.join(tok_root, "ind-validation"))["validation"]
    ood = None
    ood_path = os.path.join(tok_root, "ood-validation")
    if os.path.isdir(ood_path):
        ood = load_from_disk(ood_path)["validation"]

    ind = random_subset(ind, max_samples=args.max_eval_samples, fraction=args.eval_fraction, seed=args.eval_seed)
    if ood is not None:
        ood = random_subset(ood, max_samples=args.max_eval_samples, fraction=args.eval_fraction, seed=args.eval_seed)

    # Tokenizer (depuis OUT_DIR où tu as sauvegardé le tokenizer)
    tok = AutoTokenizer.from_pretrained(out)
    collator = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)

    # Args d'éval
    targs = TrainingArguments(
        output_dir=out,
        do_train=False, do_eval=True,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16_full_eval=args.fp16_full_eval,
        dataloader_num_workers=16,
        report_to=["wandb"]
    )

    # Dummy model (sera remplacé à chaque ckpt)
    model = AutoModelForMaskedLM.from_pretrained(out)
    trainer = InvariantTrainer(model=model, args=targs, eval_dataset=ind,
                               tokenizer=tok, data_collator=collator)

    device = targs.device
    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    for step in steps:
        ck = os.path.join(out, f"model-{step}")
        if not os.path.isdir(ck):
            print(f"[WARN] checkpoint absent: {ck}")
            continue
        m = AutoModelForMaskedLM.from_pretrained(ck)
        m.to(device); m.eval()
        trainer.model = m

        ind_out = trainer.evaluate(eval_dataset=ind)
        ind_loss = float(ind_out["eval_loss"]); ind_ppl = math.exp(ind_loss)
        log = {"evaluation/ind_perplexity": ind_ppl, "training/global_step": step}

        if ood is not None:
            ood_out = trainer.evaluate(eval_dataset=ood)
            ood_loss = float(ood_out["eval_loss"]); ood_ppl = math.exp(ood_loss)
            log["evaluation/ood_perplexity"] = ood_ppl

        wandb.log(log, step=step)
        # Free VRAM entre ckpts
        del m
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    wandb.finish()

if __name__ == "__main__":
    main()