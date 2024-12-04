# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import time
import torch
import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import AutoModelForCausalLM, get_scheduler

from dschat.utils.ds_utils import get_train_ds_config, get_eval_ds_config
from dschat.utils.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from dschat.utils.model.model_utils import create_hf_model, create_critic_model
from dschat.utils.utils import get_optimizer_grouped_parameters
"""
TODOs:
  * support HF models for critic (for debugging), must be a previously saved ckpt from step-2
  * determine ds_config/zero_stage based on model size, gpu style, world size, etc
    - get model size by creating simple meta model
    - 1.3b: zero-2 for actor/ref models, zero-0 for others
    - 13b+: zero-3 for all models
"""


def log_init(model_name, stime=None):
    if torch.distributed.get_rank() == 0:
        tag = "start" if stime is None else "end"
        suffix = "ing" if stime is None else "ed"
        duration = ""
        if stime is not None:
            duration = "(duration: {:.2f}s)".format(time.time() - stime)
        msg = f"[{tag}] Initializ{suffix} {model_name} Model [{tag}] {duration}"
        stars = (90 - len(msg)) // 2
        extra_star = "*" if (90 - len(msg)) % 2 == 1 else ""
        print("*" * stars + msg + "*" * stars + extra_star)
        return time.time()


class DeepSpeedRLHFEngine():

    def __init__(self, actor_model_name_or_path, critic_model_name_or_path,
                 tokenizer, args, num_total_iters):
        # actor_model_name_or_path: '/data1/csw_model_weights/Llama-2-7b-chat-hf'
        # critic_model_name_or_path: '/data0/csw/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/output_step2_llama_7b_epoch1_lr9.65e-6'
        # tokenizer: LlamaTokenizer(name_or_path='/data1/csw_model_weights/Llama-2-7b-chat-hf', vocab_size=32000, ...)
        # num_total_iters: 23650

        self.args = args
        self.num_total_iters = num_total_iters
        self.tokenizer = tokenizer

        self.actor = self._init_actor(
            actor_model_name_or_path=actor_model_name_or_path)
        self.ref = self._init_ref(
            actor_model_name_or_path=actor_model_name_or_path)
        self.actor_ema = None
        # self.args.enable_ema: False
        if self.args.enable_ema:
            self.actor_ema = self._init_ema(
                actor_model_name_or_path=actor_model_name_or_path)
        self.critic = self._init_critic(
            critic_model_name_or_path=critic_model_name_or_path)
        self.reward = self._init_reward(
            critic_model_name_or_path=critic_model_name_or_path)

        # self.args.critic_gradient_checkpointing: True
        if self.args.critic_gradient_checkpointing:
            self.critic.gradient_checkpointing_enable()

    def _init_actor(self, actor_model_name_or_path):
        stime = log_init("Actor")

        # DS Config
        ds_config = get_train_ds_config(
            offload=self.args.offload,
            dtype=self.args.dtype,
            stage=self.args.actor_zero_stage,
            enable_hybrid_engine=self.args.enable_hybrid_engine,
            inference_tp_size=self.args.inference_tp_size,
            release_inference_cache=self.args.release_inference_cache,
            pin_parameters=(not self.args.unpin_actor_parameters),
            tp_gather_partition_size=self.args.tp_gather_partition_size,
            max_out_tokens=self.args.max_prompt_seq_len +
            self.args.max_answer_seq_len,
            enable_tensorboard=self.args.enable_tensorboard,
            enable_mixed_precision_lora=self.args.enable_mixed_precision_lora,
            tb_path=self.args.tensorboard_path,
            tb_name="step3_actor")
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor

        # Model
        actor_model = create_hf_model(
            model_class=AutoModelForCausalLM,
            model_name_or_path=actor_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_config,
            dropout=self.args.actor_dropout)
        # actor_model: LlamaForCausalLM(
        #   (model): LlamaModel(
        #     (embed_tokens): Embedding(32008, 4096)
        #     (layers): ModuleList(
        #       (0-31): 32 x LlamaDecoderLayer(
        #         (self_attn): LlamaSdpaAttention(
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
        #   (lm_head): Linear(in_features=4096, out_features=32008, bias=False)
        # )

        # LoRA
        # self.args.actor_lora_dim: 64
        if self.args.actor_lora_dim > 0:
            actor_model = convert_linear_layer_to_lora(
                actor_model, self.args.actor_lora_module_name,
                self.args.actor_lora_dim)
            # self.args.only_optimize_lora: False
            if self.args.only_optimize_lora:
                actor_model = only_optimize_lora_parameters(actor_model)
                # 只优化 Q/K/V/O/Up/Gate/Down 这些层的 lora 参数
                # 其余包括 input_layernorm/post_layernorm/norm/lm_head/embed_tokens 这些层中的参数都不优化
                actor_model = make_model_gradient_checkpointing_compatible(
                    actor_model)

        # Optimizer
        # self.args.offload: False
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(
            actor_model, self.args.actor_weight_decay,
            self.args.actor_lora_learning_rate)
        # 正常情况下会优化 Q/K/V/O/Up/Gate/Down 这些层的 lora 参数 和 input_layernorm/post_layernorm/norm/lm_head/embed_tokens 这些层中的参数
        optim = AdamOptimizer(optim_params,
                              lr=self.args.actor_learning_rate,
                              betas=(0.9, 0.95))
        # self.args.actor_learning_rate: 9.65e-06

        # LR Scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        #TODO: move enable_hybrid_engine and pin_parameters to ds_config
        actor_engine, *_ = deepspeed.initialize(model=actor_model,
                                                optimizer=optim,
                                                lr_scheduler=lr_scheduler,
                                                config=ds_config)
        # actor_engine: DeepSpeedEngine(
        #   (module): LlamaForCausalLM(
        #     (model): LlamaModel(
        #       (embed_tokens): Embedding(32008, 4096)
        #       (layers): ModuleList(
        #         (0-31): 32 x LlamaDecoderLayer(
        #           (self_attn): LlamaSdpaAttention(
        #             (q_proj): LinearLayer_LoRA(
        #               (lora_dropout): Identity()
        #             )
        #             (k_proj): LinearLayer_LoRA(
        #               (lora_dropout): Identity()
        #             )
        #             (v_proj): LinearLayer_LoRA(
        #               (lora_dropout): Identity()
        #             )
        #             (o_proj): LinearLayer_LoRA(
        #               (lora_dropout): Identity()
        #             )
        #             (rotary_emb): LlamaRotaryEmbedding()
        #           )
        #           (mlp): LlamaMLP(
        #             (gate_proj): LinearLayer_LoRA(
        #               (lora_dropout): Identity()
        #             )
        #             (up_proj): LinearLayer_LoRA(
        #               (lora_dropout): Identity()
        #             )
        #             (down_proj): LinearLayer_LoRA(
        #               (lora_dropout): Identity()
        #             )
        #             (act_fn): SiLU()
        #           )
        #           (input_layernorm): LlamaRMSNorm((0,), eps=1e-06)
        #           (post_attention_layernorm): LlamaRMSNorm((0,), eps=1e-06)
        #         )
        #       )
        #       (norm): LlamaRMSNorm((0,), eps=1e-06)
        #       (rotary_emb): LlamaRotaryEmbedding()
        #     )
        #     (lm_head): Linear(in_features=4096, out_features=32008, bias=False)
        #   )
        # )

        log_init("Actor", stime=stime)

        return actor_engine

    def _init_ref(self, actor_model_name_or_path):
        stime = log_init("Ref")
        # DS Config
        zero_stage = self.args.actor_zero_stage
        # zero_stage: 3
        if zero_stage != 3:
            # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory for ref model
            zero_stage = 0
        ds_config = get_eval_ds_config(self.args.offload_reference_model,
                                       self.args.dtype, zero_stage)
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor

        ref_model = create_hf_model(AutoModelForCausalLM,
                                    actor_model_name_or_path, self.tokenizer,
                                    ds_config)
        # ref_model: LlamaForCausalLM(
        #   (model): LlamaModel(
        #     (embed_tokens): Embedding(32008, 4096)
        #     (layers): ModuleList(
        #       (0-31): 32 x LlamaDecoderLayer(
        #         (self_attn): LlamaSdpaAttention(
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
        #   (lm_head): Linear(in_features=4096, out_features=32008, bias=False)
        # )

        ref_engine, *_ = deepspeed.initialize(model=ref_model,
                                              config=ds_config)
        # ref_engine: DeepSpeedEngine(
        #   (module): LlamaForCausalLM(
        #     (model): LlamaModel(
        #       (embed_tokens): Embedding(32008, 4096)
        #       (layers): ModuleList(
        #         (0-31): 32 x LlamaDecoderLayer(
        #           (self_attn): LlamaSdpaAttention(
        #             (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
        #             (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
        #             (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
        #             (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        #             (rotary_emb): LlamaRotaryEmbedding()
        #           )
        #           (mlp): LlamaMLP(
        #             (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
        #             (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
        #             (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
        #             (act_fn): SiLU()
        #           )
        #           (input_layernorm): LlamaRMSNorm((0,), eps=1e-06)
        #           (post_attention_layernorm): LlamaRMSNorm((0,), eps=1e-06)
        #         )
        #       )
        #       (norm): LlamaRMSNorm((0,), eps=1e-06)
        #       (rotary_emb): LlamaRotaryEmbedding()
        #     )
        #     (lm_head): Linear(in_features=4096, out_features=32008, bias=False)
        #   )
        # )

        log_init("Ref", stime=stime)
        return ref_engine

    def _init_ema(self, actor_model_name_or_path):
        stime = log_init("EMA")
        # DS Config
        zero_stage = self.args.actor_zero_stage
        if zero_stage != 3:
            # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            zero_stage = 0
        ds_config = get_eval_ds_config(self.args.offload_reference_model,
                                       self.args.dtype, zero_stage)
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor

        actor_model_ema = create_hf_model(AutoModelForCausalLM,
                                          actor_model_name_or_path,
                                          self.tokenizer, ds_config)
        if self.args.actor_lora_dim > 0:
            actor_model_ema = convert_linear_layer_to_lora(
                actor_model_ema, self.args.actor_lora_module_name,
                self.args.actor_lora_dim)

        ema_engine, *_ = deepspeed.initialize(model=actor_model_ema,
                                              config=ds_config)

        log_init("EMA", stime=stime)
        return ema_engine

    def _init_critic(self, critic_model_name_or_path):
        # critic_model_name_or_path: '/data0/csw/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/output_step2_llama_7b_epoch1_lr9.65e-6'
        stime = log_init("Critic")
        ds_config = get_train_ds_config(
            offload=self.args.offload,
            dtype=self.args.dtype,
            stage=self.args.critic_zero_stage,
            enable_tensorboard=self.args.enable_tensorboard,
            tb_path=self.args.tensorboard_path,
            tb_name="step3_critic")
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps

        ds_eval_config = get_eval_ds_config(offload=False,
                                            dtype=self.args.dtype,
                                            stage=self.args.critic_zero_stage)
        # We need to set train batch size and micro batch size here to pass the sanity check of DeepSpeed engine.
        ds_eval_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        ds_eval_config[
            'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps

        # Model
        critic_model = create_critic_model(
            model_name_or_path=critic_model_name_or_path,
            # model_name_or_path: '/data0/csw/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/output_step2_llama_7b_epoch1_lr9.65e-6'
            tokenizer=self.tokenizer,
            # tokenizer: LlamaTokenizer(name_or_path='/data1/csw_model_weights/Llama-2-7b-chat-hf', vocab_size=32000, ...)
            ds_config=ds_eval_config,
            num_padding_at_beginning=self.args.num_padding_at_beginning,
            # num_padding_at_beginning: 1
            rlhf_training=True,
            dropout=self.args.critic_dropout,
            # dropout: None
            zero_stage=self.args.critic_zero_stage)
            # zero_stage: 3

        # critic_model: RewardModel(
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

        # LoRA
        if self.args.critic_lora_dim > 0:
            critic_model = convert_linear_layer_to_lora(
                critic_model, self.args.critic_lora_module_name,
                self.args.critic_lora_dim)
            if self.args.only_optimize_lora:
                critic_model = only_optimize_lora_parameters(critic_model)
                critic_model = make_model_gradient_checkpointing_compatible(
                    critic_model)

        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(
            critic_model, self.args.critic_weight_decay,
            self.args.critic_lora_learning_rate)
        optim = AdamOptimizer(optim_params,
                              lr=self.args.critic_learning_rate,
                              betas=(0.9, 0.95))

        # LR Scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        critic_engine, *_ = deepspeed.initialize(model=critic_model,
                                                 optimizer=optim,
                                                 lr_scheduler=lr_scheduler,
                                                 config=ds_config)
        # critic_engine: DeepSpeedEngine(
        #   (module): RewardModel(
        #     (v_head): Linear(in_features=4096, out_features=1, bias=False)
        #     (rwtransformer): LlamaModel(
        #       (embed_tokens): Embedding(32008, 4096, padding_idx=2)
        #       (layers): ModuleList(
        #         (0-31): 32 x LlamaDecoderLayer(
        #           (self_attn): LlamaAttention(
        #             (q_proj): LinearLayer_LoRA(
        #               (lora_dropout): Identity()
        #             )
        #             (k_proj): LinearLayer_LoRA(
        #               (lora_dropout): Identity()
        #             )
        #             (v_proj): LinearLayer_LoRA(
        #               (lora_dropout): Identity()
        #             )
        #             (o_proj): LinearLayer_LoRA(
        #               (lora_dropout): Identity()
        #             )
        #             (rotary_emb): LlamaRotaryEmbedding()
        #           )
        #           (mlp): LlamaMLP(
        #             (gate_proj): LinearLayer_LoRA(
        #               (lora_dropout): Identity()
        #             )
        #             (up_proj): LinearLayer_LoRA(
        #               (lora_dropout): Identity()
        #             )
        #             (down_proj): LinearLayer_LoRA(
        #               (lora_dropout): Identity()
        #             )
        #             (act_fn): SiLU()
        #           )
        #           (input_layernorm): LlamaRMSNorm((0,), eps=1e-06)
        #           (post_attention_layernorm): LlamaRMSNorm((0,), eps=1e-06)
        #         )
        #       )
        #       (norm): LlamaRMSNorm((0,), eps=1e-06)
        #       (rotary_emb): LlamaRotaryEmbedding()
        #     )
        #   )
        # )

        log_init("Critic", stime=stime)
        return critic_engine

    def _init_reward(self, critic_model_name_or_path):
        stime = log_init("Reward")
        # DS Config
        zero_stage = self.args.critic_zero_stage
        if zero_stage != 3:
            # If critic is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            zero_stage = 0

        ds_config = get_eval_ds_config(offload=self.args.offload_reward_model,
                                       dtype=self.args.dtype,
                                       stage=zero_stage)

        # We need to set train batch size and micro batch size here to pass the sanity check of DeepSpeed engine.
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        ds_config[
            'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps

        # Model
        reward_model = create_critic_model(
            model_name_or_path=critic_model_name_or_path,
            # model_name_or_path: '/data0/csw/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/output_step2_llama_7b_epoch1_lr9.65e-6'
            tokenizer=self.tokenizer,
            # tokenizer: LlamaTokenizer(name_or_path='/data1/csw_model_weights/Llama-2-7b-chat-hf', vocab_size=32000, ...)
            ds_config=ds_config,
            num_padding_at_beginning=self.args.num_padding_at_beginning,
            # num_padding_at_beginning: 1
            rlhf_training=True,
            dropout=self.args.critic_dropout,
            # dropout: None
            zero_stage=zero_stage)
            # zero_stage: 3

        # reward_model: RewardModel(
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

        reward_engine, *_ = deepspeed.initialize(model=reward_model,
                                                 config=ds_config)
        # reward_engine: DeepSpeedEngine(
        #   (module): RewardModel(
        #     (v_head): Linear(in_features=4096, out_features=1, bias=False)
        #     (rwtransformer): LlamaModel(
        #       (embed_tokens): Embedding(32008, 4096, padding_idx=2)
        #       (layers): ModuleList(
        #         (0-31): 32 x LlamaDecoderLayer(
        #           (self_attn): LlamaAttention(
        #             (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
        #             (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
        #             (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
        #             (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        #             (rotary_emb): LlamaRotaryEmbedding()
        #           )
        #           (mlp): LlamaMLP(
        #             (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
        #             (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
        #             (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
        #             (act_fn): SiLU()
        #           )
        #           (input_layernorm): LlamaRMSNorm((0,), eps=1e-06)
        #           (post_attention_layernorm): LlamaRMSNorm((0,), eps=1e-06)
        #         )
        #       )
        #       (norm): LlamaRMSNorm((0,), eps=1e-06)
        #       (rotary_emb): LlamaRotaryEmbedding()
        #     )
        #   )
        # )

        log_init("Reward", stime=stime)
        return reward_engine
