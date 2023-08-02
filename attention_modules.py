import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

dt = torch.bfloat16


class SaveAttention(nn.Module):
    def __init__(self, base, index, **kwargs):
        super().__init__()
        self.base_attn = base
        self.history = []

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        out = self.base_attn.forward(hidden_states, encoder_hidden_states, attention_mask, **cross_attention_kwargs)
        self.history.append(out.detach())
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except:
            return self.base_attn.__getattribute__(name)


class CorrespondingAttention(nn.Module):
    def __init__(self, base, index, **kwargs):
        super().__init__()
        self.base_attn = base
        self.index = index
        self.attn_transitions = kwargs["attn_transitions"]  # layer batch { }
        self.active_frames = kwargs["active_frames"]  # batch
        self.keyframe_attns = kwargs["keyframe_attns"]  # layer batch hw c
        self.tb_true_to_batch_index = kwargs["tb_true_to_batch_index"]  # batch
        self.history = None

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        my_layer_corr = self.attn_transitions[self.index]
        my_layer_keyframe_attns = self.keyframe_attns[self.index]  # batch hw c

        batch_before_key_frame_indexes = torch.tensor([self.tb_true_to_batch_index[c["before_frame"]] for c in my_layer_corr])
        batch_after_key_frame_indexes = torch.tensor([self.tb_true_to_batch_index[c["after_frame"]] for c in my_layer_corr])

        before_keyframe_attns = my_layer_keyframe_attns[batch_before_key_frame_indexes]
        after_keyframe_attns = my_layer_keyframe_attns[batch_after_key_frame_indexes]

        before_shuffled = torch.stack([b[c["before"]] for b, c in zip(before_keyframe_attns, my_layer_corr)])
        after_shuffled = torch.stack([a[c["after"]] for a, c in zip(after_keyframe_attns, my_layer_corr)])

        w = torch.tensor([(c["center_frame"] - c["before_frame"]) / (c["after_frame"] - c["before_frame"]) for c in my_layer_corr]).cuda().to(dt)

        out = torch.einsum("ijk,i->ijk", [before_shuffled, 1 - w]) + torch.einsum("ijk,i->ijk", [after_shuffled, w])

        self.history = out.detach()

        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return self.base_attn.__getattribute__(name)


class ExtendedAttention(nn.Module):
    def __init__(self, base, index):
        super().__init__()
        self.index = index
        self.base_attn = base
        self.history = None

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # we batch by polarity (positive / negative prompt), so positive and negative prompts don't cross-attend.

        b = hidden_states.shape[0] // 2
        hidden_states = rearrange(hidden_states, "(p b) hw c -> p (b hw) c", p=2)
        if encoder_hidden_states is not None:
            encoder_hidden_states = reduce(encoder_hidden_states, "(p b) t c -> p t c", "mean", p=2)
        out = self.base_attn.forward(hidden_states, encoder_hidden_states, attention_mask, **cross_attention_kwargs)
        out = rearrange(out, "p (b hw) c -> (p b) hw c", b=b, p=2)

        self.history = out.detach()  # out.chunk(2)[1].detach()
        return out

    def __getattr__(self, name):
        try:
            # we need this to search _modules, where torch moves otherwise-normal properties.
            # normally you can just do super().__getattr__(name) but that doesn't work here.
            # since .base_attn isn't in the __dict__, and will fallthrough to __getattr__.
            return super().__getattr__(name)
        except AttributeError:
            return self.base_attn.__getattribute__(name)


class NormalAttention(nn.Module):
    def __init__(self, base, index, **kwargs):
        super().__init__()
        self.base_attn = base
        self.history = None

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        out = self.base_attn.forward(hidden_states, encoder_hidden_states, attention_mask, **cross_attention_kwargs)
        self.history = out.detach()
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return self.base_attn.__getattribute__(name)
