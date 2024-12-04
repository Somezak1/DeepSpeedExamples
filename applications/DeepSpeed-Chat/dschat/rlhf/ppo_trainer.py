# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import time
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator

from dschat.utils.utils import print_rank_0


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).to(
        get_accelerator().current_device_name())
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    # logits: [1, 511, 32008]
    # labels: [1, 511]
    log_probs = F.log_softmax(logits, dim=-1)
    # log_probs: [1, 511, 32008]
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    # log_probs_labels: [1, 511, 1]
    return log_probs_labels.squeeze(-1)


class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        # self.tokenizer: LlamaTokenizer(name_or_path='/data1/csw_model_weights/Llama-2-7b-chat-hf', vocab_size=32000, ...)
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        # self.max_answer_seq_len: 256
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        # self.end_of_conversation_token_id: 29958
        self.z3_enabled = args.actor_zero_stage == 3
        # self.z3_enabled: True
        self.compute_fp32_loss = self.args.compute_fp32_loss
        # self.compute_fp32_loss: False

        # In case the generated experience is not valid (too short), we use the last valid
        # generated experience. Alternatively, we can skip the step (on all workers).
        # For now, use the last valid experience which is a simpler solution
        self.last_generated_experience = None

        # Those value can be changed
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.generate_time = 0.0

    def _generate_sequence(self, prompts, mask, step):
        # 使用 actor model 回答 prompts 中的 args.per_device_generation_batch_size 个 question, 并将回复长度大于 1 的返回
        # 直接使用预训练模型作为 actor model 可能会出现回复长度过短的现象
        # 数据并行时各个 GPU 实际处理的样本数量可能会因为过滤机制而不同

        # prompts.shape: [1, 256]
        # mask.shape: [1, 256]

        max_min_length = self.max_answer_seq_len + prompts.shape[1]
        # 512 = 256 + 256

        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        if self.actor_model.module.config.model_type == "llama":
            kwargs = dict(do_sample=False)
        else:
            kwargs = dict()

        with torch.no_grad():
            seq = self.actor_model.module.generate(
                prompts,
                attention_mask=mask,
                max_length=max_min_length,
                pad_token_id=self.tokenizer.pad_token_id,
                synced_gpus=self.z3_enabled,
                **kwargs)
            # seq.shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len]
            # 由于 prompts 里有多个样本, 模型对每个样本回复的长度不一致, 假设这些样本中最长的回复长度为 batch_ans_max_len, batch_ans_max_len 不大于 max_answer_seq_len
            # 那么回复长度不及 batch_ans_max_len 的样本, 其 answer 的内容中会有 padding

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        # prompt_length: 256
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        # ans.shape: [args.per_device_generation_batch_size, batch_ans_max_len]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        # valid_ans_len: [args.per_device_generation_batch_size, ]

        if self.args.print_answers and (step % self.args.print_answers_interval
                                        == 0):
            # self.args.print_answers: False
            # self.args.print_answers_interval: 1
            print(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[
                    i] <= 1:  # if the answer is shorter than 1 token, drop it
                print(
                    f'Dropping too short generated answer: {step=}: \n'
                    f'prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                    f'answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
                )
                continue
            else:
                out_seq.append(seq[i:i + 1])

        if not out_seq:
            print(
                f'All generated results are too short for rank={self.args.local_rank} step={step}\n'
                f'-> prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                f'-> answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
            )
            return None

        out_seq = torch.cat(out_seq, dim=0)  # concat output in the batch dim
        # out_seq.shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len]

        return out_seq

    def generate_experience(self, prompts, mask, step):
        # args.per_device_generation_batch_size: 1
        # prompts.shape: [args.per_device_generation_batch_size, 256], prompts 其实由 padding + question 组成, 如果 question 够长那就没有 padding
        # mask.shape: [args.per_device_generation_batch_size, 256]
        self.eval()
        generate_start = time.time()

        # 使用 actor model 回答 prompts 中的 args.per_device_generation_batch_size 个 question, 并将回复长度大于 1 的返回
        # 直接使用预训练模型作为 actor model 可能会出现回复长度过短的现象
        # 数据并行时各个 GPU 实际处理的样本数量可能会因为过滤机制而不同
        seq = self._generate_sequence(prompts, mask, step)
        # seq.shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len]
        # 由于 prompts 里有多个样本, 模型对每个样本回复的长度不一致, 假设这些样本中最长的回复长度为 batch_ans_max_len
        # 那么回复长度不及 batch_ans_max_len 的样本, 其 answer 的内容中会有 padding
        generate_end = time.time()
        if seq is None:
            # 如果这个 batch 中没有样本满足要求, 就拿上次符合要求的数据作为 经验样本
            assert self.last_generated_experience is not None, f'Invalid generated experience at {step=}'
            prompts = self.last_generated_experience['prompts']
            seq = self.last_generated_experience['seq']
        else:
            # 记录本次满足要求的样本, 以备不时之需
            self.last_generated_experience = {'prompts': prompts, 'seq': seq}
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        # pad_token_id: 32000
        attention_mask = seq.not_equal(pad_token_id).long()
        # attention_mask.shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len], pad_token_id 为 0, 其余 token 为 1
        with torch.no_grad():
            # seq 左侧是 padding(也可能因为 question 够长所以没有 padding) 中间是 question 右边是 answer + padding
            output = self.actor_model(seq, attention_mask=attention_mask)
            # output.logits.shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len, 32008]
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            # output_ref.logits.shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len, 32008]
            reward_score = self.reward_model.forward_value(
                seq, attention_mask,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach(
                )
            # reward_score.shape: [args.per_device_generation_batch_size, ], 表示 reward model 对每个样本 answer 最后一个 token 的打分
            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]
            # values.shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len - 1], 表示 critic model 对每个样本除最后一个 token 以外其他所有 token 的价值估计

        logits = output.logits
        # logits.shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len, 32008], actor model 对于每个 token 预测的概率分布
        logits_ref = output_ref.logits
        # logits_ref.shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len, 32008], ref model 对于每个 token 预测的概率分布

        # self.compute_fp32_loss: False
        if self.compute_fp32_loss:
            logits = logits.to(torch.float)
            logits_ref = logits_ref.to(torch.float)

        self.generate_time = generate_end - generate_start

        return {
            'prompts': prompts,
            # prompts: [args.per_device_generation_batch_size, 256], 原始 batch 样本, 每个样本由 padding + question 组成
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            # logits[:, :-1, :].shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len - 1, 32008]
            # seq[:, 1:].shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len - 1]
            # logprobs.shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len - 1], 表示 actor model 对于每个标签 token 的对数预测概率
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:]),
            # logits_ref[:, :-1, :].shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len - 1, 32008]
            # seq[:, 1:].shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len - 1]
            # ref_logprobs.shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len - 1], 表示 ref model 对于每个标签 token 的对数预测概率
            'value': values,
            # values.shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len - 1], 表示 critic model 对每个样本除最后一个 token 以外其他所有 token 的价值估计
            'rewards': reward_score,
            # reward_score.shape: [args.per_device_generation_batch_size, ], 表示 reward model 对每个样本 answer 最后一个 token 的打分
            'input_ids': seq,
            # seq.shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len], seq 包括 padding + question + answer + padding
            "attention_mask": attention_mask
            # attention_mask.shape: [args.per_device_generation_batch_size, 256 + batch_ans_max_len], pad_token_id 为 0, 其余 token 为 1
        }

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):
        # args.per_device_training_batch_size: 1
        # prompts.shape:       [args.per_device_training_batch_size, 256], 原始 batch 样本, 每个样本由 padding + question 组成
        # log_probs.shape:     [args.per_device_training_batch_size, 256 + batch_ans_max_len - 1], 表示 actor model 对于每个标签 token 的对数预测概率
        # ref_log_probs.shape: [args.per_device_training_batch_size, 256 + batch_ans_max_len - 1], 表示 ref model 对于每个标签 token 的对数预测概率
        # reward_score.shape:  [args.per_device_training_batch_size, ], 表示 reward model 对每个样本 answer 最后一个 token 的打分
        # action_mask.shape:   [args.per_device_training_batch_size, 256 + batch_ans_max_len - 1]

        # self.kl_ctl: 0.1
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        # kl_divergence_estimate: [args.per_device_training_batch_size, 256 + ans_len - 1]
        rewards = kl_divergence_estimate
        # rewards: [args.per_device_training_batch_size, 256 + ans_len - 1]
        start = prompts.shape[1] - 1
        # start: 255
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        # self.clip_reward_value: 5
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        # attention_mask  i        token        probs          ref_probs        reward_score        rewards
        # 0               0        [P]          P_{0}^{old}    P_{0}^{'}        -                   -self.kl_ctl * log( P_{0}^{old} / P_{0}^{'} )
        # 0               1        [P]          P_{1}^{old}    P_{1}^{'}        -                   -self.kl_ctl * log( P_{1}^{old} / P_{1}^{'} )
        # 1               2        你           P_{2}^{old}    P_{2}^{'}        -                   -self.kl_ctl * log( P_{2}^{old} / P_{2}^{'} )
        # 1               3        是           P_{3}^{old}    P_{3}^{'}        -                   -self.kl_ctl * log( P_{3}^{old} / P_{3}^{'} )
        # 1               4        谁           P_{4}^{old}    P_{4}^{'}        -                   -self.kl_ctl * log( P_{4}^{old} / P_{4}^{'} )
        # 1               5        ？           P_{5}^{old}    P_{5}^{'}        -                   -self.kl_ctl * log( P_{5}^{old} / P_{5}^{'} )
        # 1               6        我           P_{6}^{old}    P_{6}^{'}        -                   -self.kl_ctl * log( P_{6}^{old} / P_{6}^{'} )
        # 1               7        是           P_{7}^{old}    P_{7}^{'}        -                   -self.kl_ctl * log( P_{7}^{old} / P_{7}^{'} )
        # 1               8        机           P_{8}^{old}    P_{8}^{'}        -                   -self.kl_ctl * log( P_{8}^{old} / P_{8}^{'} )
        # 1               9        器           P_{9}^{old}    P_{9}^{'}        -                   -self.kl_ctl * log( P_{9}^{old} / P_{9}^{'} )
        # 1               10       人           P_{10}^{old}   P_{10}^{'}       -                   -self.kl_ctl * log( P_{10}^{old} / P_{10}^{'} )
        # 1               11       。           P_{11}^{old}   P_{11}^{'}       reward_score        -self.kl_ctl * log( P_{11}^{old} / P_{11}^{'} ) + clip(reward_score, -5, 5)
        # 0               12       [P]          P_{12}^{old}   P_{12}^{'}       -                   -self.kl_ctl * log( P_{12}^{old} / P_{12}^{'} )
        # 0               13       [P]          P_{13}^{old}   P_{13}^{'}       -                   -self.kl_ctl * log( P_{13}^{old} / P_{13}^{'} )
        # 0               14       [P]          -              -                -                   -

        return rewards

    def train_rlhf(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        # args.per_device_training_batch_size: 1

        prompts = inputs['prompts']
        # [args.per_device_training_batch_size, 256], 原始 batch 样本, 每个样本由 padding + question 组成

        log_probs = inputs['logprobs']
        # [args.per_device_training_batch_size, 256 + batch_ans_max_len - 1], 表示 actor model 对于每个标签 token 的对数预测概率

        ref_log_probs = inputs['ref_logprobs']
        # [args.per_device_training_batch_size, 256 + batch_ans_max_len - 1], 表示 ref model 对于每个标签 token 的对数预测概率

        reward_score = inputs['rewards']
        # [args.per_device_training_batch_size, ], 表示 reward model 对每个样本 answer 最后一个 token 的打分

        values = inputs['value']
        # [args.per_device_training_batch_size, 256 + batch_ans_max_len - 1], 表示 critic model 对每个样本除最后一个 token 以外其他所有 token 的价值估计

        attention_mask = inputs['attention_mask']
        # [args.per_device_training_batch_size, 256 + batch_ans_max_len], pad_token_id 为 0, 其余 token 为 1

        seq = inputs['input_ids']
        # [args.per_device_training_batch_size, 256 + batch_ans_max_len], seq 包括 padding + question + answer + padding

        start = prompts.size()[-1] - 1
        # start: 255

        action_mask = attention_mask[:, 1:]
        # action_mask.shape: [args.per_device_training_batch_size, 256 + batch_ans_max_len - 1]

        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)
            # old_rewards: [args.per_device_training_batch_size, 256 + batch_ans_max_len - 1]
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong

            # attention_mask  i        token        old_values        old_rewards
            # 0               0        [P]          V_{0}^{old}       -self.kl_ctl * log( P_{0}^{old} / P_{0}^{'} )
            # 0               1        [P]          V_{1}^{old}       -self.kl_ctl * log( P_{1}^{old} / P_{1}^{'} )
            # 1               2        你            V_{2}^{old}       -self.kl_ctl * log( P_{2}^{old} / P_{2}^{'} )
            # 1               3        是            V_{3}^{old}       -self.kl_ctl * log( P_{3}^{old} / P_{3}^{'} )
            # 1               4        谁            V_{4}^{old}       -self.kl_ctl * log( P_{4}^{old} / P_{4}^{'} )
            # 1               5        ？            V_{5}^{old}       -self.kl_ctl * log( P_{5}^{old} / P_{5}^{'} )
            # 1               6        我            V_{6}^{old}       -self.kl_ctl * log( P_{6}^{old} / P_{6}^{'} )
            # 1               7        是            V_{7}^{old}       -self.kl_ctl * log( P_{7}^{old} / P_{7}^{'} )
            # 1               8        机            V_{8}^{old}       -self.kl_ctl * log( P_{8}^{old} / P_{8}^{'} )
            # 1               9        器            V_{9}^{old}       -self.kl_ctl * log( P_{9}^{old} / P_{9}^{'} )
            # 1               10       人            V_{10}^{old}      -self.kl_ctl * log( P_{10}^{old} / P_{10}^{'} )
            # 1               11       。            V_{11}^{old}      -self.kl_ctl * log( P_{11}^{old} / P_{11}^{'} ) + clip(reward_score, -5, 5)
            # 0               12       [P]           V_{12}^{old}      -self.kl_ctl * log( P_{12}^{old} / P_{12}^{'} )
            # 0               13       [P]           V_{13}^{old}      -self.kl_ctl * log( P_{13}^{old} / P_{13}^{'} )
            # 0               14       [P]           -                 -

            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0

            # attention_mask  i        token        old_values        old_rewards
            # 0               0        [P]          V_{0}^{old}       -self.kl_ctl * log( P_{0}^{old} / P_{0}^{'} )
            # 0               1        [P]          V_{1}^{old}       -self.kl_ctl * log( P_{1}^{old} / P_{1}^{'} )
            # 1               2        你            V_{2}^{old}       -self.kl_ctl * log( P_{2}^{old} / P_{2}^{'} )
            # 1               3        是            V_{3}^{old}       -self.kl_ctl * log( P_{3}^{old} / P_{3}^{'} )
            # 1               4        谁            V_{4}^{old}       -self.kl_ctl * log( P_{4}^{old} / P_{4}^{'} )
            # 1               5        ？            V_{5}^{old}       -self.kl_ctl * log( P_{5}^{old} / P_{5}^{'} )
            # 1               6        我            V_{6}^{old}       -self.kl_ctl * log( P_{6}^{old} / P_{6}^{'} )
            # 1               7        是            V_{7}^{old}       -self.kl_ctl * log( P_{7}^{old} / P_{7}^{'} )
            # 1               8        机            V_{8}^{old}       -self.kl_ctl * log( P_{8}^{old} / P_{8}^{'} )
            # 1               9        器            V_{9}^{old}       -self.kl_ctl * log( P_{9}^{old} / P_{9}^{'} )
            # 1               10       人            V_{10}^{old}      -self.kl_ctl * log( P_{10}^{old} / P_{10}^{'} )
            # 1               11       。            V_{11}^{old}      -self.kl_ctl * log( P_{11}^{old} / P_{11}^{'} ) + clip(reward_score, -5, 5)
            # 0               12       [P]           0                 0
            # 0               13       [P]           0                 0
            # 0               14       [P]           -                 -

            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)
            # attention_mask  i        token         old_values        old_rewards                                                                  delta                                               advantages                       returns
            # 0               0        [P]           V_{0}^{old}       -self.kl_ctl * log( P_{0}^{old} / P_{0}^{'} )                                -                                                   -                                -
            # 0               1        [P]           V_{1}^{old}       -self.kl_ctl * log( P_{1}^{old} / P_{1}^{'} )                                -                                                   -                                -
            # 1               2        你            V_{2}^{old}       -self.kl_ctl * log( P_{2}^{old} / P_{2}^{'} )                                -                                                   -                                -
            # 1               3        是            V_{3}^{old}       -self.kl_ctl * log( P_{3}^{old} / P_{3}^{'} )                                -                                                   -                                -
            # 1               4        谁            V_{4}^{old}       -self.kl_ctl * log( P_{4}^{old} / P_{4}^{'} )                                -                                                   -                                -
            # 1               5        ？            V_{5}^{old}       -self.kl_ctl * log( P_{5}^{old} / P_{5}^{'} )                                old_rewards[5]+γ*old_values[6]-old_values[5]        delta[5]+γ*λ*advantanges[6]      advantages[5]+old_values[5]
            # 1               6        我            V_{6}^{old}       -self.kl_ctl * log( P_{6}^{old} / P_{6}^{'} )                                old_rewards[6]+γ*old_values[7]-old_values[6]        delta[6]+γ*λ*advantanges[7]      advantages[6]+old_values[6]
            # 1               7        是            V_{7}^{old}       -self.kl_ctl * log( P_{7}^{old} / P_{7}^{'} )                                old_rewards[7]+γ*old_values[8]-old_values[7]        delta[7]+γ*λ*advantanges[8]      advantages[7]+old_values[7]
            # 1               8        机            V_{8}^{old}       -self.kl_ctl * log( P_{8}^{old} / P_{8}^{'} )                                old_rewards[8]+γ*old_values[9]-old_values[8]        delta[8]+γ*λ*advantanges[9]      advantages[8]+old_values[8]
            # 1               9        器            V_{9}^{old}       -self.kl_ctl * log( P_{9}^{old} / P_{9}^{'} )                                old_rewards[9]+γ*old_values[10]-old_values[9]       delta[9]+γ*λ*advantanges[10]     advantages[9]+old_values[9]
            # 1               10       人            V_{10}^{old}      -self.kl_ctl * log( P_{10}^{old} / P_{10}^{'} )                              old_rewards[10]+γ*old_values[11]-old_values[10]     delta[10]+γ*λ*advantanges[11]    advantages[10]+old_values[10]
            # 1               11       。            V_{11}^{old}      -self.kl_ctl * log( P_{11}^{old} / P_{11}^{'} ) + clip(reward_score, -5, 5)  old_rewards[11]+γ*0-old_values[11]                  delta[11]+γ*λ*0                  advantages[11]+old_values[11]
            # 0               12       [P]           0                 0                                                                            old_rewards[12]+γ*0-0=0                             delta[12]+γ*λ*0=0                advantages[12]+old_values[12]=0
            # 0               13       [P]           0                 0                                                                            old_rewards[13]+γ*0-0=0                             delta[13]+γ*λ*0=0                advantages[13]+old_values[13]=0
            # 0               14       [P]           -                 -                                                                            -                                                   -                                -


        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])
        self.actor_model.backward(actor_loss)

        if not self.args.align_overflow:
            self.actor_model.step()

        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                       start:],
                                          returns, action_mask[:, start:])
        self.critic_model.backward(critic_loss)

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(
                external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()

        self.critic_model.step()

        return actor_loss, critic_loss

    def get_overflow(self):
        # Overflow is not expected when using bf16
        # Therefore, DeepSpeed's BF16_Optimizer does not maintain an overflow indication
        if self.args.dtype == "bf16":
            return False, False

        # bf16 的表示范围远大于 fp16, 但是精度不如 fp16
        # 因此使用 fp16 时需要验证, 优化器在更新时是否发生了数值溢出
        actor_overflow = self.actor_model.optimizer.overflow
        critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow, critic_overflow

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss

        # mask          attention_mask  i        token        new_prob        old_prob         advantages                       returns
        # 1             1               5        ？            P_{5}^{new}     P_{5}^{old}      delta[5]+γ*λ*advantanges[6]      advantages[5]+old_values[5]
        # 1             1               6        我            P_{6}^{new}     P_{6}^{old}      delta[6]+γ*λ*advantanges[7]      advantages[6]+old_values[6]
        # 1             1               7        是            P_{7}^{new}     P_{7}^{old}      delta[7]+γ*λ*advantanges[8]      advantages[7]+old_values[7]
        # 1             1               8        机            P_{8}^{new}     P_{8}^{old}      delta[8]+γ*λ*advantanges[9]      advantages[8]+old_values[8]
        # 1             1               9        器            P_{9}^{new}     P_{9}^{old}      delta[9]+γ*λ*advantanges[10]     advantages[9]+old_values[9]
        # 1             1               10       人            P_{10}^{new}    P_{10}^{old}     delta[10]+γ*λ*advantanges[11]    advantages[10]+old_values[10]
        # 0             1               11       。            P_{11}^{new}    P_{11}^{old}     delta[11]+γ*λ*0                  advantages[11]+old_values[11]
        # 0             0               12       [P]           P_{12}^{new}    P_{12}^{old}     delta[12]+γ*λ*0=0                advantages[12]+old_values[12]=0
        # 0             0               13       [P]           P_{13}^{new}    P_{13}^{old}     delta[13]+γ*λ*0=0                advantages[13]+old_values[13]=0
        # -             0               14       [P]           -               -                -                                -                                                                            -                                                   -                                -

        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        # self.cliprange: 0.2
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss

        # mask          attention_mask  i        token        new_value        old_value        advantages                       returns
        # 1             1               5        ？            V_{5}^{new}     V_{5}^{old}      delta[5]+γ*λ*advantanges[6]      advantages[5]+old_values[5]
        # 1             1               6        我            V_{6}^{new}     V_{6}^{old}      delta[6]+γ*λ*advantanges[7]      advantages[6]+old_values[6]
        # 1             1               7        是            V_{7}^{new}     V_{7}^{old}      delta[7]+γ*λ*advantanges[8]      advantages[7]+old_values[7]
        # 1             1               8        机            V_{8}^{new}     V_{8}^{old}      delta[8]+γ*λ*advantanges[9]      advantages[8]+old_values[8]
        # 1             1               9        器            V_{9}^{new}     V_{9}^{old}      delta[9]+γ*λ*advantanges[10]     advantages[9]+old_values[9]
        # 1             1               10       人            V_{10}^{new}    V_{10}^{old}     delta[10]+γ*λ*advantanges[11]    advantages[10]+old_values[10]
        # 0             1               11       。            V_{11}^{new}    V_{11}^{old}     delta[11]+γ*λ*0                  advantages[11]+old_values[11]
        # 0             0               12       [P]           V_{12}^{new}    V_{12}^{old}     delta[12]+γ*λ*0=0                advantages[12]+old_values[12]=0
        # 0             0               13       [P]           V_{13}^{new}    V_{13}^{old}     delta[13]+γ*λ*0=0                advantages[13]+old_values[13]=0
        # -             0               14       [P]           -               -                -                                -                                                                            -                                                   -                                -

        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        if self.compute_fp32_loss:
            values = values.float()
            values_clipped = values_clipped.float()
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        # values: [args.per_device_training_batch_size, 256 + batch_ans_max_len - 1]
        # rewards: [args.per_device_training_batch_size, 256 + batch_ans_max_len - 1]

        # attention_mask  i        token        values            rewards
        # 0               0        [P]           V_{0}^{old}       -self.kl_ctl * log( P_{0}^{old} / P_{0}^{'} )
        # 0               1        [P]           V_{1}^{old}       -self.kl_ctl * log( P_{1}^{old} / P_{1}^{'} )
        # 1               2        你            V_{2}^{old}       -self.kl_ctl * log( P_{2}^{old} / P_{2}^{'} )
        # 1               3        是            V_{3}^{old}       -self.kl_ctl * log( P_{3}^{old} / P_{3}^{'} )
        # 1               4        谁            V_{4}^{old}       -self.kl_ctl * log( P_{4}^{old} / P_{4}^{'} )
        # 1               5        ？            V_{5}^{old}       -self.kl_ctl * log( P_{5}^{old} / P_{5}^{'} )
        # 1               6        我            V_{6}^{old}       -self.kl_ctl * log( P_{6}^{old} / P_{6}^{'} )
        # 1               7        是            V_{7}^{old}       -self.kl_ctl * log( P_{7}^{old} / P_{7}^{'} )
        # 1               8        机            V_{8}^{old}       -self.kl_ctl * log( P_{8}^{old} / P_{8}^{'} )
        # 1               9        器            V_{9}^{old}       -self.kl_ctl * log( P_{9}^{old} / P_{9}^{'} )
        # 1               10       人            V_{10}^{old}      -self.kl_ctl * log( P_{10}^{old} / P_{10}^{'} )
        # 1               11       。            V_{11}^{old}      -self.kl_ctl * log( P_{11}^{old} / P_{11}^{'} ) + clip(reward_score, -5, 5)
        # 0               12       [P]           0                 0
        # 0               13       [P]           0                 0
        # 0               14       [P]           -                 -

        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        # self.gamma: 1
        # self.lam: 0.95
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]

        # attention_mask  i        token         values            rewards                                                                      delta                                   advantages                      returns
        # 0               0        [P]           V_{0}^{old}       -self.kl_ctl * log( P_{0}^{old} / P_{0}^{'} )                                -                                       -                                -
        # 0               1        [P]           V_{1}^{old}       -self.kl_ctl * log( P_{1}^{old} / P_{1}^{'} )                                -                                       -                                -
        # 1               2        你            V_{2}^{old}       -self.kl_ctl * log( P_{2}^{old} / P_{2}^{'} )                                -                                       -                                -
        # 1               3        是            V_{3}^{old}       -self.kl_ctl * log( P_{3}^{old} / P_{3}^{'} )                                -                                       -                                -
        # 1               4        谁            V_{4}^{old}       -self.kl_ctl * log( P_{4}^{old} / P_{4}^{'} )                                -                                       -                                -
        # 1               5        ？            V_{5}^{old}       -self.kl_ctl * log( P_{5}^{old} / P_{5}^{'} )                                rewards[5]+γ*values[6]-values[5]        delta[5]+γ*λ*advantanges[6]      advantages[5]+values[5]
        # 1               6        我            V_{6}^{old}       -self.kl_ctl * log( P_{6}^{old} / P_{6}^{'} )                                rewards[6]+γ*values[7]-values[6]        delta[6]+γ*λ*advantanges[7]      advantages[6]+values[6]
        # 1               7        是            V_{7}^{old}       -self.kl_ctl * log( P_{7}^{old} / P_{7}^{'} )                                rewards[7]+γ*values[8]-values[7]        delta[7]+γ*λ*advantanges[8]      advantages[7]+values[7]
        # 1               8        机            V_{8}^{old}       -self.kl_ctl * log( P_{8}^{old} / P_{8}^{'} )                                rewards[8]+γ*values[9]-values[8]        delta[8]+γ*λ*advantanges[9]      advantages[8]+values[8]
        # 1               9        器            V_{9}^{old}       -self.kl_ctl * log( P_{9}^{old} / P_{9}^{'} )                                rewards[9]+γ*values[10]-values[9]       delta[9]+γ*λ*advantanges[10]     advantages[9]+values[9]
        # 1               10       人            V_{10}^{old}      -self.kl_ctl * log( P_{10}^{old} / P_{10}^{'} )                              rewards[10]+γ*values[11]-values[10]     delta[10]+γ*λ*advantanges[11]    advantages[10]+values[10]
        # 1               11       。            V_{11}^{old}      -self.kl_ctl * log( P_{11}^{old} / P_{11}^{'} ) + clip(reward_score, -5, 5)  rewards[11]+γ*0-values[11]              delta[11]+γ*λ*0                  advantages[11]+values[11]
        # 0               12       [P]           0                 0                                                                            rewards[12]+γ*0-0=0                     delta[12]+γ*λ*0=0                advantages[12]+values[12]=0
        # 0               13       [P]           0                 0                                                                            rewards[13]+γ*0-0=0                     delta[13]+γ*λ*0=0                advantages[13]+values[13]=0
        # 0               14       [P]           -                 -                                                                            -                                       -                                -

        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
