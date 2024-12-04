# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn


## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):

    def __init__(self,
                 base_model,
                 tokenizer,
                 num_padding_at_beginning=0,
                 compute_fp32_loss=False):
        # num_padding_at_beginning: 0
        # compute_fp32_loss: False
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        # hasattr(self.config, "word_embed_proj_dim"): False
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            # self.config.n_embd: 4096
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.rwtransformer = base_model
        # self.rwtransformer: LlamaModel(
        #   (embed_tokens): Embedding(32008, 4096)
        #   (layers): ModuleList(
        #     (0-31): 32 x LlamaDecoderLayer(
        #       (self_attn): LlamaSdpaAttention(
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
        self.PAD_ID = tokenizer.pad_token_id
        # self.PAD_ID: 32000
        self.compute_fp32_loss = compute_fp32_loss

    def gradient_checkpointing_enable(self):
        self.rwtransformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtransformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                tokenizer=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False):
        # input_ids.size():      [16, 512]
        # attention_mask.size(): [16, 512]
        # 其他参数都是默认值

        loss = None

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        hidden_states = transformer_outputs[0]
        # hidden_states.size(): [16, 512, 4096]
        rewards = self.v_head(hidden_states).squeeze(-1)
        # rewards.size(): [16, 512]
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        # bs: 8
        seq_len = input_ids.shape[1]
        # seq_len: 512

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        # [8, 512]
        rejected_ids = input_ids[bs:]
        # [8, 512]
        chosen_rewards = rewards[:bs]
        # [8, 512]
        rejected_rewards = rewards[bs:]
        # [8, 512]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0.
        for i in range(bs):
            chosen_id = chosen_ids[i]
            # [512,]
            rejected_id = rejected_ids[i]
            # [512,]
            chosen_reward = chosen_rewards[i]
            # [512,]
            rejected_reward = rejected_rewards[i]
            # [512,]

            c_inds = (chosen_id == self.PAD_ID).nonzero()
            # self.PAD_ID: 32000
            # nonzero() 函数用于返回非零元素的索引, 此处是返回 input ids 中值为 PAD_ID 的索引
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()
            # 此处是返回样本对中 input ids 不同的索引

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            if self.compute_fp32_loss:
                c_truncated_reward = c_truncated_reward.float()
                r_truncated_reward = r_truncated_reward.float()
            loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
                                                    r_truncated_reward).mean()

            # tokenizer.decode(chosen_id):   '<s> \n\nHuman: How big should a baby play yard be?\n\nAssistant: Well, it depends on how big the child is.  If the baby is the size of an average 6 month old infant, then they’ll probably need a play yard that’s about 2 feet by 3 feet by 2 feet.  But if the baby is much bigger, then we’ll need a larger play yard.\n\nHuman: That seems kind of small since they get bigger so fast.\n\nAssistant: Yes, and it can be confusing to have a play yard that’s too small or too big, because the baby won’t fit well and may also be able to climb out of it.  If the baby does climb out of the play yard, we’ll need a larger one.  But a really large play yard can cause problems with falling, so it’s also important to find a balance between too large and too small.<|endoftext|>[PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD]'
            # tokenizer.decode(rejected_id): '<s> \n\nHuman: How big should a baby play yard be?\n\nAssistant: Well, it depends on how big the child is.  If the baby is the size of an average 6 month old infant, then they’ll probably need a play yard that’s about 2 feet by 3 feet by 2 feet.  But if the baby is much bigger, then we’ll need a larger play yard.\n\nHuman: That seems kind of small since they get bigger so fast.\n\nAssistant: Actually I’m afraid it is pretty small for a baby that big.  The average 6 month old is only 3 feet tall.  If the baby is much bigger, then we’ll need to double those dimensions.<|endoftext|>[PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD]'

            # for i, (c, r, cm, rm) in enumerate(zip(
            #     tokenizer.convert_ids_to_tokens(chosen_id),
            #     tokenizer.convert_ids_to_tokens(rejected_id),
            # 	  attention_mask[0],
            # 	  attention_mask[8],
            # )):
            #     print(f"id: {str(i).rjust(3)}  chosen_token: {c.rjust(12)}  rejected_token: {r.rjust(12)}    chosen_mask: {cm}    rejected_mask: {rm}")
            # id:   0  chosen_token:          <s>  rejected_token:          <s>    chosen_mask: 1    rejected_mask: 1
            # id:   1  chosen_token:            ▁  rejected_token:            ▁    chosen_mask: 1    rejected_mask: 1
            # id:   2  chosen_token:       <0x0A>  rejected_token:       <0x0A>    chosen_mask: 1    rejected_mask: 1
            # id:   3  chosen_token:       <0x0A>  rejected_token:       <0x0A>    chosen_mask: 1    rejected_mask: 1
            # id:   4  chosen_token:            H  rejected_token:            H    chosen_mask: 1    rejected_mask: 1
            # id:   5  chosen_token:         uman  rejected_token:         uman    chosen_mask: 1    rejected_mask: 1
            # id:   6  chosen_token:            :  rejected_token:            :    chosen_mask: 1    rejected_mask: 1
            # id:   7  chosen_token:         ▁How  rejected_token:         ▁How    chosen_mask: 1    rejected_mask: 1
            # id:   8  chosen_token:         ▁big  rejected_token:         ▁big    chosen_mask: 1    rejected_mask: 1
            # id:   9  chosen_token:      ▁should  rejected_token:      ▁should    chosen_mask: 1    rejected_mask: 1
            # id:  10  chosen_token:           ▁a  rejected_token:           ▁a    chosen_mask: 1    rejected_mask: 1
            # id:  11  chosen_token:        ▁baby  rejected_token:        ▁baby    chosen_mask: 1    rejected_mask: 1
            # id:  12  chosen_token:        ▁play  rejected_token:        ▁play    chosen_mask: 1    rejected_mask: 1
            # id:  13  chosen_token:        ▁yard  rejected_token:        ▁yard    chosen_mask: 1    rejected_mask: 1
            # id:  14  chosen_token:          ▁be  rejected_token:          ▁be    chosen_mask: 1    rejected_mask: 1
            # id:  15  chosen_token:            ?  rejected_token:            ?    chosen_mask: 1    rejected_mask: 1
            # id:  16  chosen_token:       <0x0A>  rejected_token:       <0x0A>    chosen_mask: 1    rejected_mask: 1
            # id:  17  chosen_token:       <0x0A>  rejected_token:       <0x0A>    chosen_mask: 1    rejected_mask: 1
            # id:  18  chosen_token:          Ass  rejected_token:          Ass    chosen_mask: 1    rejected_mask: 1
            # id:  19  chosen_token:       istant  rejected_token:       istant    chosen_mask: 1    rejected_mask: 1
            # id:  20  chosen_token:            :  rejected_token:            :    chosen_mask: 1    rejected_mask: 1
            # id:  21  chosen_token:        ▁Well  rejected_token:        ▁Well    chosen_mask: 1    rejected_mask: 1
            # id:  22  chosen_token:            ,  rejected_token:            ,    chosen_mask: 1    rejected_mask: 1
            # id:  23  chosen_token:          ▁it  rejected_token:          ▁it    chosen_mask: 1    rejected_mask: 1
            # id:  24  chosen_token:     ▁depends  rejected_token:     ▁depends    chosen_mask: 1    rejected_mask: 1
            # id:  25  chosen_token:          ▁on  rejected_token:          ▁on    chosen_mask: 1    rejected_mask: 1
            # id:  26  chosen_token:         ▁how  rejected_token:         ▁how    chosen_mask: 1    rejected_mask: 1
            # id:  27  chosen_token:         ▁big  rejected_token:         ▁big    chosen_mask: 1    rejected_mask: 1
            # id:  28  chosen_token:         ▁the  rejected_token:         ▁the    chosen_mask: 1    rejected_mask: 1
            # id:  29  chosen_token:       ▁child  rejected_token:       ▁child    chosen_mask: 1    rejected_mask: 1
            # id:  30  chosen_token:          ▁is  rejected_token:          ▁is    chosen_mask: 1    rejected_mask: 1
            # id:  31  chosen_token:            .  rejected_token:            .    chosen_mask: 1    rejected_mask: 1
            # id:  32  chosen_token:            ▁  rejected_token:            ▁    chosen_mask: 1    rejected_mask: 1
            # id:  33  chosen_token:          ▁If  rejected_token:          ▁If    chosen_mask: 1    rejected_mask: 1
            # id:  34  chosen_token:         ▁the  rejected_token:         ▁the    chosen_mask: 1    rejected_mask: 1
            # id:  35  chosen_token:        ▁baby  rejected_token:        ▁baby    chosen_mask: 1    rejected_mask: 1
            # id:  36  chosen_token:          ▁is  rejected_token:          ▁is    chosen_mask: 1    rejected_mask: 1
            # id:  37  chosen_token:         ▁the  rejected_token:         ▁the    chosen_mask: 1    rejected_mask: 1
            # id:  38  chosen_token:        ▁size  rejected_token:        ▁size    chosen_mask: 1    rejected_mask: 1
            # id:  39  chosen_token:          ▁of  rejected_token:          ▁of    chosen_mask: 1    rejected_mask: 1
            # id:  40  chosen_token:          ▁an  rejected_token:          ▁an    chosen_mask: 1    rejected_mask: 1
            # id:  41  chosen_token:     ▁average  rejected_token:     ▁average    chosen_mask: 1    rejected_mask: 1
            # id:  42  chosen_token:            ▁  rejected_token:            ▁    chosen_mask: 1    rejected_mask: 1
            # id:  43  chosen_token:            6  rejected_token:            6    chosen_mask: 1    rejected_mask: 1
            # id:  44  chosen_token:       ▁month  rejected_token:       ▁month    chosen_mask: 1    rejected_mask: 1
            # id:  45  chosen_token:         ▁old  rejected_token:         ▁old    chosen_mask: 1    rejected_mask: 1
            # id:  46  chosen_token:      ▁infant  rejected_token:      ▁infant    chosen_mask: 1    rejected_mask: 1
            # id:  47  chosen_token:            ,  rejected_token:            ,    chosen_mask: 1    rejected_mask: 1
            # id:  48  chosen_token:        ▁then  rejected_token:        ▁then    chosen_mask: 1    rejected_mask: 1
            # id:  49  chosen_token:        ▁they  rejected_token:        ▁they    chosen_mask: 1    rejected_mask: 1
            # id:  50  chosen_token:            ’  rejected_token:            ’    chosen_mask: 1    rejected_mask: 1
            # id:  51  chosen_token:           ll  rejected_token:           ll    chosen_mask: 1    rejected_mask: 1
            # id:  52  chosen_token:    ▁probably  rejected_token:    ▁probably    chosen_mask: 1    rejected_mask: 1
            # id:  53  chosen_token:        ▁need  rejected_token:        ▁need    chosen_mask: 1    rejected_mask: 1
            # id:  54  chosen_token:           ▁a  rejected_token:           ▁a    chosen_mask: 1    rejected_mask: 1
            # id:  55  chosen_token:        ▁play  rejected_token:        ▁play    chosen_mask: 1    rejected_mask: 1
            # id:  56  chosen_token:        ▁yard  rejected_token:        ▁yard    chosen_mask: 1    rejected_mask: 1
            # id:  57  chosen_token:        ▁that  rejected_token:        ▁that    chosen_mask: 1    rejected_mask: 1
            # id:  58  chosen_token:            ’  rejected_token:            ’    chosen_mask: 1    rejected_mask: 1
            # id:  59  chosen_token:            s  rejected_token:            s    chosen_mask: 1    rejected_mask: 1
            # id:  60  chosen_token:       ▁about  rejected_token:       ▁about    chosen_mask: 1    rejected_mask: 1
            # id:  61  chosen_token:            ▁  rejected_token:            ▁    chosen_mask: 1    rejected_mask: 1
            # id:  62  chosen_token:            2  rejected_token:            2    chosen_mask: 1    rejected_mask: 1
            # id:  63  chosen_token:        ▁feet  rejected_token:        ▁feet    chosen_mask: 1    rejected_mask: 1
            # id:  64  chosen_token:          ▁by  rejected_token:          ▁by    chosen_mask: 1    rejected_mask: 1
            # id:  65  chosen_token:            ▁  rejected_token:            ▁    chosen_mask: 1    rejected_mask: 1
            # id:  66  chosen_token:            3  rejected_token:            3    chosen_mask: 1    rejected_mask: 1
            # id:  67  chosen_token:        ▁feet  rejected_token:        ▁feet    chosen_mask: 1    rejected_mask: 1
            # id:  68  chosen_token:          ▁by  rejected_token:          ▁by    chosen_mask: 1    rejected_mask: 1
            # id:  69  chosen_token:            ▁  rejected_token:            ▁    chosen_mask: 1    rejected_mask: 1
            # id:  70  chosen_token:            2  rejected_token:            2    chosen_mask: 1    rejected_mask: 1
            # id:  71  chosen_token:        ▁feet  rejected_token:        ▁feet    chosen_mask: 1    rejected_mask: 1
            # id:  72  chosen_token:            .  rejected_token:            .    chosen_mask: 1    rejected_mask: 1
            # id:  73  chosen_token:            ▁  rejected_token:            ▁    chosen_mask: 1    rejected_mask: 1
            # id:  74  chosen_token:         ▁But  rejected_token:         ▁But    chosen_mask: 1    rejected_mask: 1
            # id:  75  chosen_token:          ▁if  rejected_token:          ▁if    chosen_mask: 1    rejected_mask: 1
            # id:  76  chosen_token:         ▁the  rejected_token:         ▁the    chosen_mask: 1    rejected_mask: 1
            # id:  77  chosen_token:        ▁baby  rejected_token:        ▁baby    chosen_mask: 1    rejected_mask: 1
            # id:  78  chosen_token:          ▁is  rejected_token:          ▁is    chosen_mask: 1    rejected_mask: 1
            # id:  79  chosen_token:        ▁much  rejected_token:        ▁much    chosen_mask: 1    rejected_mask: 1
            # id:  80  chosen_token:      ▁bigger  rejected_token:      ▁bigger    chosen_mask: 1    rejected_mask: 1
            # id:  81  chosen_token:            ,  rejected_token:            ,    chosen_mask: 1    rejected_mask: 1
            # id:  82  chosen_token:        ▁then  rejected_token:        ▁then    chosen_mask: 1    rejected_mask: 1
            # id:  83  chosen_token:          ▁we  rejected_token:          ▁we    chosen_mask: 1    rejected_mask: 1
            # id:  84  chosen_token:            ’  rejected_token:            ’    chosen_mask: 1    rejected_mask: 1
            # id:  85  chosen_token:           ll  rejected_token:           ll    chosen_mask: 1    rejected_mask: 1
            # id:  86  chosen_token:        ▁need  rejected_token:        ▁need    chosen_mask: 1    rejected_mask: 1
            # id:  87  chosen_token:           ▁a  rejected_token:           ▁a    chosen_mask: 1    rejected_mask: 1
            # id:  88  chosen_token:      ▁larger  rejected_token:      ▁larger    chosen_mask: 1    rejected_mask: 1
            # id:  89  chosen_token:        ▁play  rejected_token:        ▁play    chosen_mask: 1    rejected_mask: 1
            # id:  90  chosen_token:        ▁yard  rejected_token:        ▁yard    chosen_mask: 1    rejected_mask: 1
            # id:  91  chosen_token:            .  rejected_token:            .    chosen_mask: 1    rejected_mask: 1
            # id:  92  chosen_token:       <0x0A>  rejected_token:       <0x0A>    chosen_mask: 1    rejected_mask: 1
            # id:  93  chosen_token:       <0x0A>  rejected_token:       <0x0A>    chosen_mask: 1    rejected_mask: 1
            # id:  94  chosen_token:            H  rejected_token:            H    chosen_mask: 1    rejected_mask: 1
            # id:  95  chosen_token:         uman  rejected_token:         uman    chosen_mask: 1    rejected_mask: 1
            # id:  96  chosen_token:            :  rejected_token:            :    chosen_mask: 1    rejected_mask: 1
            # id:  97  chosen_token:        ▁That  rejected_token:        ▁That    chosen_mask: 1    rejected_mask: 1
            # id:  98  chosen_token:       ▁seems  rejected_token:       ▁seems    chosen_mask: 1    rejected_mask: 1
            # id:  99  chosen_token:        ▁kind  rejected_token:        ▁kind    chosen_mask: 1    rejected_mask: 1
            # id: 100  chosen_token:          ▁of  rejected_token:          ▁of    chosen_mask: 1    rejected_mask: 1
            # id: 101  chosen_token:       ▁small  rejected_token:       ▁small    chosen_mask: 1    rejected_mask: 1
            # id: 102  chosen_token:       ▁since  rejected_token:       ▁since    chosen_mask: 1    rejected_mask: 1
            # id: 103  chosen_token:        ▁they  rejected_token:        ▁they    chosen_mask: 1    rejected_mask: 1
            # id: 104  chosen_token:         ▁get  rejected_token:         ▁get    chosen_mask: 1    rejected_mask: 1
            # id: 105  chosen_token:      ▁bigger  rejected_token:      ▁bigger    chosen_mask: 1    rejected_mask: 1
            # id: 106  chosen_token:          ▁so  rejected_token:          ▁so    chosen_mask: 1    rejected_mask: 1
            # id: 107  chosen_token:        ▁fast  rejected_token:        ▁fast    chosen_mask: 1    rejected_mask: 1
            # id: 108  chosen_token:            .  rejected_token:            .    chosen_mask: 1    rejected_mask: 1
            # id: 109  chosen_token:       <0x0A>  rejected_token:       <0x0A>    chosen_mask: 1    rejected_mask: 1
            # id: 110  chosen_token:       <0x0A>  rejected_token:       <0x0A>    chosen_mask: 1    rejected_mask: 1
            # id: 111  chosen_token:          Ass  rejected_token:          Ass    chosen_mask: 1    rejected_mask: 1
            # id: 112  chosen_token:       istant  rejected_token:       istant    chosen_mask: 1    rejected_mask: 1
            # id: 113  chosen_token:            :  rejected_token:            :    chosen_mask: 1    rejected_mask: 1
            # id: 114  chosen_token:         ▁Yes  rejected_token:    ▁Actually    chosen_mask: 1    rejected_mask: 1       divergence_ind = 114
            # id: 115  chosen_token:            ,  rejected_token:           ▁I    chosen_mask: 1    rejected_mask: 1
            # id: 116  chosen_token:         ▁and  rejected_token:            ’    chosen_mask: 1    rejected_mask: 1
            # id: 117  chosen_token:          ▁it  rejected_token:            m    chosen_mask: 1    rejected_mask: 1
            # id: 118  chosen_token:         ▁can  rejected_token:      ▁afraid    chosen_mask: 1    rejected_mask: 1
            # id: 119  chosen_token:          ▁be  rejected_token:          ▁it    chosen_mask: 1    rejected_mask: 1
            # id: 120  chosen_token:   ▁confusing  rejected_token:          ▁is    chosen_mask: 1    rejected_mask: 1
            # id: 121  chosen_token:          ▁to  rejected_token:      ▁pretty    chosen_mask: 1    rejected_mask: 1
            # id: 122  chosen_token:        ▁have  rejected_token:       ▁small    chosen_mask: 1    rejected_mask: 1
            # id: 123  chosen_token:           ▁a  rejected_token:         ▁for    chosen_mask: 1    rejected_mask: 1
            # id: 124  chosen_token:        ▁play  rejected_token:           ▁a    chosen_mask: 1    rejected_mask: 1
            # id: 125  chosen_token:        ▁yard  rejected_token:        ▁baby    chosen_mask: 1    rejected_mask: 1
            # id: 126  chosen_token:        ▁that  rejected_token:        ▁that    chosen_mask: 1    rejected_mask: 1
            # id: 127  chosen_token:            ’  rejected_token:         ▁big    chosen_mask: 1    rejected_mask: 1
            # id: 128  chosen_token:            s  rejected_token:            .    chosen_mask: 1    rejected_mask: 1
            # id: 129  chosen_token:         ▁too  rejected_token:            ▁    chosen_mask: 1    rejected_mask: 1
            # id: 130  chosen_token:       ▁small  rejected_token:         ▁The    chosen_mask: 1    rejected_mask: 1
            # id: 131  chosen_token:          ▁or  rejected_token:     ▁average    chosen_mask: 1    rejected_mask: 1
            # id: 132  chosen_token:         ▁too  rejected_token:            ▁    chosen_mask: 1    rejected_mask: 1
            # id: 133  chosen_token:         ▁big  rejected_token:            6    chosen_mask: 1    rejected_mask: 1
            # id: 134  chosen_token:            ,  rejected_token:       ▁month    chosen_mask: 1    rejected_mask: 1
            # id: 135  chosen_token:     ▁because  rejected_token:         ▁old    chosen_mask: 1    rejected_mask: 1
            # id: 136  chosen_token:         ▁the  rejected_token:          ▁is    chosen_mask: 1    rejected_mask: 1
            # id: 137  chosen_token:        ▁baby  rejected_token:        ▁only    chosen_mask: 1    rejected_mask: 1
            # id: 138  chosen_token:         ▁won  rejected_token:            ▁    chosen_mask: 1    rejected_mask: 1
            # id: 139  chosen_token:            ’  rejected_token:            3    chosen_mask: 1    rejected_mask: 1
            # id: 140  chosen_token:            t  rejected_token:        ▁feet    chosen_mask: 1    rejected_mask: 1
            # id: 141  chosen_token:         ▁fit  rejected_token:        ▁tall    chosen_mask: 1    rejected_mask: 1
            # id: 142  chosen_token:        ▁well  rejected_token:            .    chosen_mask: 1    rejected_mask: 1
            # id: 143  chosen_token:         ▁and  rejected_token:            ▁    chosen_mask: 1    rejected_mask: 1
            # id: 144  chosen_token:         ▁may  rejected_token:          ▁If    chosen_mask: 1    rejected_mask: 1
            # id: 145  chosen_token:        ▁also  rejected_token:         ▁the    chosen_mask: 1    rejected_mask: 1
            # id: 146  chosen_token:          ▁be  rejected_token:        ▁baby    chosen_mask: 1    rejected_mask: 1
            # id: 147  chosen_token:        ▁able  rejected_token:          ▁is    chosen_mask: 1    rejected_mask: 1
            # id: 148  chosen_token:          ▁to  rejected_token:        ▁much    chosen_mask: 1    rejected_mask: 1
            # id: 149  chosen_token:        ▁clim  rejected_token:      ▁bigger    chosen_mask: 1    rejected_mask: 1
            # id: 150  chosen_token:            b  rejected_token:            ,    chosen_mask: 1    rejected_mask: 1
            # id: 151  chosen_token:         ▁out  rejected_token:        ▁then    chosen_mask: 1    rejected_mask: 1
            # id: 152  chosen_token:          ▁of  rejected_token:          ▁we    chosen_mask: 1    rejected_mask: 1
            # id: 153  chosen_token:          ▁it  rejected_token:            ’    chosen_mask: 1    rejected_mask: 1
            # id: 154  chosen_token:            .  rejected_token:           ll    chosen_mask: 1    rejected_mask: 1
            # id: 155  chosen_token:            ▁  rejected_token:        ▁need    chosen_mask: 1    rejected_mask: 1
            # id: 156  chosen_token:          ▁If  rejected_token:          ▁to    chosen_mask: 1    rejected_mask: 1
            # id: 157  chosen_token:         ▁the  rejected_token:      ▁double    chosen_mask: 1    rejected_mask: 1
            # id: 158  chosen_token:        ▁baby  rejected_token:       ▁those    chosen_mask: 1    rejected_mask: 1
            # id: 159  chosen_token:        ▁does  rejected_token:  ▁dimensions    chosen_mask: 1    rejected_mask: 1
            # id: 160  chosen_token:        ▁clim  rejected_token:           .<    chosen_mask: 1    rejected_mask: 1
            # id: 161  chosen_token:            b  rejected_token:            |    chosen_mask: 1    rejected_mask: 1
            # id: 162  chosen_token:         ▁out  rejected_token:          end    chosen_mask: 1    rejected_mask: 1
            # id: 163  chosen_token:          ▁of  rejected_token:           of    chosen_mask: 1    rejected_mask: 1
            # id: 164  chosen_token:         ▁the  rejected_token:         text    chosen_mask: 1    rejected_mask: 1
            # id: 165  chosen_token:        ▁play  rejected_token:            |    chosen_mask: 1    rejected_mask: 1
            # id: 166  chosen_token:        ▁yard  rejected_token:            >    chosen_mask: 1    rejected_mask: 1       r_ind - 1 = 166
            # id: 167  chosen_token:            ,  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0       r_ind = 167
            # id: 168  chosen_token:          ▁we  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 169  chosen_token:            ’  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 170  chosen_token:           ll  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 171  chosen_token:        ▁need  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 172  chosen_token:           ▁a  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 173  chosen_token:      ▁larger  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 174  chosen_token:         ▁one  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 175  chosen_token:            .  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 176  chosen_token:            ▁  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 177  chosen_token:         ▁But  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 178  chosen_token:           ▁a  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 179  chosen_token:      ▁really  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 180  chosen_token:       ▁large  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 181  chosen_token:        ▁play  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 182  chosen_token:        ▁yard  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 183  chosen_token:         ▁can  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 184  chosen_token:       ▁cause  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 185  chosen_token:    ▁problems  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 186  chosen_token:        ▁with  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 187  chosen_token:     ▁falling  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 188  chosen_token:            ,  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 189  chosen_token:          ▁so  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 190  chosen_token:          ▁it  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 191  chosen_token:            ’  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 192  chosen_token:            s  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 193  chosen_token:        ▁also  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 194  chosen_token:   ▁important  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 195  chosen_token:          ▁to  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 196  chosen_token:        ▁find  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 197  chosen_token:           ▁a  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 198  chosen_token:     ▁balance  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 199  chosen_token:     ▁between  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 200  chosen_token:         ▁too  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 201  chosen_token:       ▁large  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 202  chosen_token:         ▁and  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 203  chosen_token:         ▁too  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 204  chosen_token:       ▁small  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 205  chosen_token:           .<  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 206  chosen_token:            |  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 207  chosen_token:          end  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 208  chosen_token:           of  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 209  chosen_token:         text  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 210  chosen_token:            |  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0
            # id: 211  chosen_token:            >  rejected_token:        [PAD]    chosen_mask: 1    rejected_mask: 0       c_ind - 1 = 211
            # id: 212  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0       c_ind = end_ind = 212
            # id: 213  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 214  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 215  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 216  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 217  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 218  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 219  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 220  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 221  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 222  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 223  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 224  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 225  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 226  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 227  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 228  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 229  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 230  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 231  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 232  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 233  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 234  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 235  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 236  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 237  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 238  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 239  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 240  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 241  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 242  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 243  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 244  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 245  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 246  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 247  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 248  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 249  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 250  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 251  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 252  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 253  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 254  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 255  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 256  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 257  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 258  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 259  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 260  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 261  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 262  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 263  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 264  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 265  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 266  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 267  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 268  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 269  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 270  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 271  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 272  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 273  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 274  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 275  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 276  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 277  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 278  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 279  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 280  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 281  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 282  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 283  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 284  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 285  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 286  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 287  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 288  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 289  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 290  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 291  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 292  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 293  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 294  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 295  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 296  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 297  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 298  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 299  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 300  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 301  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 302  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 303  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 304  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 305  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 306  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 307  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 308  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 309  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 310  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 311  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 312  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 313  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 314  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 315  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 316  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 317  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 318  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 319  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 320  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 321  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 322  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 323  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 324  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 325  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 326  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 327  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 328  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 329  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 330  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 331  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 332  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 333  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 334  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 335  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 336  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 337  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 338  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 339  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 340  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 341  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 342  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 343  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 344  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 345  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 346  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 347  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 348  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 349  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 350  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 351  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 352  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 353  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 354  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 355  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 356  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 357  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 358  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 359  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 360  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 361  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 362  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 363  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 364  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 365  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 366  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 367  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 368  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 369  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 370  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 371  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 372  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 373  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 374  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 375  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 376  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 377  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 378  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 379  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 380  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 381  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 382  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 383  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 384  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 385  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 386  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 387  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 388  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 389  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 390  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 391  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 392  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 393  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 394  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 395  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 396  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 397  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 398  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 399  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 400  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 401  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 402  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 403  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 404  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 405  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 406  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 407  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 408  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 409  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 410  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 411  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 412  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 413  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 414  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 415  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 416  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 417  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 418  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 419  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 420  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 421  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 422  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 423  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 424  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 425  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 426  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 427  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 428  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 429  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 430  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 431  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 432  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 433  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 434  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 435  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 436  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 437  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 438  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 439  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 440  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 441  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 442  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 443  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 444  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 445  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 446  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 447  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 448  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 449  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 450  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 451  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 452  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 453  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 454  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 455  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 456  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 457  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 458  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 459  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 460  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 461  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 462  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 463  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 464  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 465  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 466  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 467  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 468  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 469  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 470  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 471  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 472  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 473  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 474  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 475  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 476  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 477  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 478  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 479  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 480  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 481  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 482  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 483  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 484  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 485  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 486  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 487  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 488  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 489  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 490  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 491  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 492  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 493  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 494  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 495  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 496  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 497  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 498  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 499  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 500  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 501  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 502  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 503  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 504  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 505  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 506  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 507  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 508  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 509  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 510  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0
            # id: 511  chosen_token:        [PAD]  rejected_token:        [PAD]    chosen_mask: 0    rejected_mask: 0

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False):

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)
        hidden_states = transformer_outputs[0]
        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }
