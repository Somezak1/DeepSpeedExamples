# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from dschat.utils.model.reward_model import RewardModel
from dschat.utils.utils import load_state_dict_into_model, print_rank_0


def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


def causal_lm_model_to_fp32_loss(model):
    """ Convert CausalLM model to calculate loss in fp32 """

    def causal_lm_forward(
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **deprecated_arguments,
    ):
        kwargs = dict() if model.config.model_type == "llama" else dict(
            head_mask=head_mask)
        output = model.__original_forward__(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs)

        return_dict = isinstance(output, dict)
        lm_logits = output.logits if return_dict else output[0]
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].float().contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length))

        if not return_dict:
            # re-pack output with fp32 loss
            return ((loss, ) + output) if loss is not None else output

        output.loss = loss
        return output

    model.__original_forward__ = model.forward
    model.forward = causal_lm_forward


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    dropout=None):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    configure_dropout(model_config, dropout)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model


def create_critic_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        dropout=None,
                        zero_stage=0,
                        compute_fp32_loss=False):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule

    # model_name_or_path: '/data0/csw/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/output_step2_llama_7b_epoch1_lr9.65e-6'
    # tokenizer: LlamaTokenizer(name_or_path='/data1/csw_model_weights/Llama-2-7b-chat-hf', vocab_size=32000, ...)
    # num_padding_at_beginning: 1
    # rlhf_training: True
    # dropout: None
    # zero_stage: 3
    # compute_fp32_loss: False

    import time

    start = time.time()
    critic_model = create_hf_model(AutoModel, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training, dropout)
    # critic_model: LlamaModel(
    #   (embed_tokens): Embedding(32008, 4096, padding_idx=2)
    #   (layers): ModuleList(
    #     (0-31): 32 x LlamaDecoderLayer(
    #       (self_attn): LlamaAttention(
    #         (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
    #         (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
    #         (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
    #         (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
    #         (rotary_emb): LlamaRotaryEmbedding()
    #       )
    #       (mlp): LlamaMLP(
    #         (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
    #         (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
    #         (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
    #         (act_fn): SiLU()
    #       )
    #       (input_layernorm): LlamaRMSNorm((0,), eps=1e-06)
    #       (post_attention_layernorm): LlamaRMSNorm((0,), eps=1e-06)
    #     )
    #   )
    #   (norm): LlamaRMSNorm((0,), eps=1e-06)
    #   (rotary_emb): LlamaRotaryEmbedding()
    # )
    end = time.time()
    print_rank_0(f">Creating model from_config took {end - start} seconds",
                 None)

    critic_model = RewardModel(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning,
        compute_fp32_loss=compute_fp32_loss)
    # RewardModel(
    #   (v_head): Linear(in_features=4096, out_features=1, bias=False)
    #   (rwtransformer): LlamaModel(
    #     (embed_tokens): Embedding(32008, 4096, padding_idx=2)
    #     (layers): ModuleList(
    #       (0-31): 32 x LlamaDecoderLayer(
    #         (self_attn): LlamaAttention(
    #           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
    #           (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
    #           (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
    #           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
    #           (rotary_emb): LlamaRotaryEmbedding()
    #         )
    #         (mlp): LlamaMLP(
    #           (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
    #           (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
    #           (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
    #           (act_fn): SiLU()
    #         )
    #         (input_layernorm): LlamaRMSNorm((0,), eps=1e-06)
    #         (post_attention_layernorm): LlamaRMSNorm((0,), eps=1e-06)
    #       )
    #     )
    #     (norm): LlamaRMSNorm((0,), eps=1e-06)
    #     (rotary_emb): LlamaRotaryEmbedding()
    #   )
    # )

    # rlhf_training: True
    if rlhf_training:
        # load critic model from checkpoint

        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        start = time.time()
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location='cpu')
        end = time.time()
        print_rank_0(f">Creating model from_config took {end - start} seconds",
                     None)

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        start = time.time()
        load_state_dict_into_model(critic_model,
                                   model_ckpt_state_dict,
                                   "",
                                   zero_stage=zero_stage)
        end = time.time()

        print_rank_0(f">Creating model from_config took {end - start} seconds",
                     None)

    return critic_model
