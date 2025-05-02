import copy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaPreTrainedModel, XLMRobertaModel, XLMRobertaLMHead
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig


class InvariantXLMRobertaConfig(XLMRobertaConfig):
    model_type = "invariant-xlm-roberta"

    def __init__(self, envs=1, **kwargs):
        """Constructs InvariantXLMRobertaConfig."""
        super().__init__(**kwargs)
        self.envs = envs


class InvariantXLMRobertaForMaskedLM(XLMRobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids", r"predictions.decoder.bias"]
    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config, model=None):
        super().__init__(config)
        self.config = config

        if not hasattr(config, "envs") or len(config.envs) == 0:
            self.envs = ["erm"]
        else:
            self.envs = config.envs

        self.encoder = XLMRobertaModel(config, add_pooling_layer=False)
        
        self.lm_heads = nn.ModuleDict({
            env_name: XLMRobertaLMHead(config)
            for env_name in self.envs
        })


        if model is not None:
            self.encoder = copy.deepcopy(model.roberta)
            
            self.lm_heads = nn.ModuleDict({
                env_name: XLMRobertaLMHead(config)
                for env_name in self.envs
            })

        for env_name, lm_head in self.lm_heads.items():
            self.__setattr__(env_name + '_head', lm_head)

        self.n_environments = len(self.lm_heads)


    def print_lm_w(self):
        for env, lm_h in self.lm_heads.items():
            print(lm_h.dense.weight)

    def init_head(self):
        for env_name in self.envs:
            self.lm_heads[env_name] = XLMRobertaLMHead(self.config)
            self.lm_heads[env_name].to('cuda')

    def init_base(self):
        self.encoder.init_weights()
        self.init_head()

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)

    def get_output_embeddings(self):
        for env, lm_head in self.lm_heads.items():
            return lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        for env, lm_head in self.lm_heads.items():
            lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        env_name=None,
        **kwargs
    ):
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")

        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        if self.n_environments == 1:
            prediction_scores = list(self.lm_heads.values())[0](sequence_output)

        elif env_name is not None:
            lm_head = self.lm_heads[env_name]
            prediction_scores = lm_head(sequence_output)

            # Vérification anti-NaN
            if torch.isnan(prediction_scores).any() or torch.isinf(prediction_scores).any():
                print(f"[NaN DETECTED] Logits contain NaN or Inf in env {env_name}")
                raise ValueError("NaN detected in logits! Abort forward pass.")

        else:
            prediction_scores = 0.
            for lm_head in self.lm_heads.values():
                prediction_scores += lm_head(sequence_output) / self.n_environments

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
