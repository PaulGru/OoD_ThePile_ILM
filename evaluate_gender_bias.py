import torch
from transformers import AutoTokenizer
from torch.nn.functional import softmax
from tqdm import tqdm

def compute_binary_entropy(p):
    return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))

def compute_bias_score(logits, tokenizer, male_token, female_token):
    # Obtenir les indices vocab pour les tokens genrés
    male_id = tokenizer.convert_tokens_to_ids(male_token)
    female_id = tokenizer.convert_tokens_to_ids(female_token)
    probs = softmax(logits, dim=-1)
    p_f = probs[female_id]
    p_m = probs[male_id]
    p = p_f / (p_f + p_m)
    return 1 - compute_binary_entropy(p) / compute_binary_entropy(torch.tensor(0.5))

def evaluate_gender_bias(model, tokenizer, sentences, male_token="he", female_token="she", env_name=None):
    model.eval()
    device = next(model.parameters()).device
    scores = []
    for sent in tqdm(sentences):
        if "[MASK]" not in sent:
            continue
        encoded = tokenizer(sent, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded, env_name=env_name)
            logits = outputs.logits
        mask_idx = (encoded["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        mask_logits = logits[0, mask_idx[0], :]
        score = compute_bias_score(mask_logits, tokenizer, male_token, female_token)
        scores.append(score.item())
    return scores
