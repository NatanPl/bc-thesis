import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from math import sqrt
from modelling.models.components.masking import ProbMask, TriangularCausalMask


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_attention = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attention_mask):
        """
        B = batch size
        L = length of queries
        S = length of keys/values
        D = model dimension
        H = number of attention heads
        Args:
            queries: [B, L, D]
            keys: [B, S, D]
            values: [B, S, D]
            attention_mask: [B, 1, L, S] or None
        Returns:
            out: [B, L, D]
            attention: [B, H, L, S] or None
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_attention(queries).reshape(B, L, H, -1)
        keys = self.key_projection(keys).reshape(B, S, H, -1)
        values = self.value_projection(values).reshape(B, S, H, -1)

        out, attention = self.inner_attention(
            queries, keys, values, attention_mask)

        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.reshape(B, L, -1)

        return self.out_projection(out), attention


class ProbSparseAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _sparsity_measurement(self, Query, Keys, sample_k, n_top):
        # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        BatchSize, NumHeads, KeysLength, KeysFeatures = Keys.shape
        _, _, QueryLength, _ = Query.shape

        # calculate the sampled Q_K
        K_expand = Keys.unsqueeze(-3).expand(BatchSize,
                                             NumHeads, QueryLength, KeysLength, KeysFeatures)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(KeysLength, (QueryLength, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            QueryLength).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Query.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), KeysLength)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Query[torch.arange(BatchSize)[:, None, None],
                         torch.arange(NumHeads)[None, :, None],
                         M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, Keys.transpose(-2, -1)
                           )  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, Values, L_Q):
        B, H, L_V, _ = Values.shape
        if not self.mask_flag:
            V_sum = Values.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H,
                                                 L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            context = Values.cumsum(dim=-2)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attention_mask):
        B, H, L_V, _ = V.shape

        if self.mask_flag:
            attention_mask = ProbMask(
                B, H, L_Q, index, scores, device=V.device)
            scores = scores.masked_fill(attention_mask.mask, -np.inf)

        attention = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attention, V).type_as(context_in)
        if self.output_attention:
            attentions = (torch.ones([B, H, L_V, L_V]) /
                          L_V).type_as(attention).to(attention.device)
            attentions[torch.arange(B)[:, None, None], torch.arange(H)[
                None, :, None], index, :] = attention
            return (context_in, attentions)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attention_mask):
        _, L_Q, _, D = queries.shape
        _, L_K, _, _ = keys.shape

        # Keep tensors in [B, L, H, D] format - transpose to [B, H, L, D] for internal computation
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._sparsity_measurement(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attention = self._update_context(
            context, values, scores_top, index, L_Q, attention_mask)

        # Return in original format [B, L, H, D]
        return context.transpose(2, 1).contiguous(), attention


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=False,
        scale=None,
        attention_dropout=0.1,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attention_mask):
        B, L, _, E = queries.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attention_mask is None:
                attention_mask = TriangularCausalMask(
                    B, L, device=queries.device)

        if attention_mask is not None:
            if hasattr(attention_mask, 'mask'):
                # Handle custom mask objects (TriangularCausalMask, ProbMask)
                mask_tensor: torch.Tensor = attention_mask.mask  # type: ignore
            else:
                # Handle tensor masks
                mask_tensor: torch.Tensor = attention_mask  # type: ignore
                if mask_tensor.dim() == 3:  # Need to expand heads dimension
                    mask_tensor = mask_tensor.unsqueeze(1)
            scores = scores.masked_fill(mask_tensor, -np.inf)

        A = torch.nan_to_num(self.dropout(
            torch.softmax(scale * scores, dim=-1)))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # Always return None for attention weights since we don't store output_attention in __init__
        return (V.contiguous(), None)
