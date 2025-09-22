import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# ------------------- Initialization functions -------------------
def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ------------------- Attention modules -------------------
def scaled_dot_product_attention_fallback(q, k, v, attn_mask=None, is_causal=False):
    """
    q, k, v: shape [bs, nh, seq_len, dh]
    attn_mask: [bs, nh, seq_len, seq_len] or [seq_len, seq_len]
    dropout_p: attention dropout 概率
    is_causal: 是否使用因果掩码（上三角为 -inf）
    """
    B, H, L, D = q.shape

    scale = D ** 0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, L, L]

    if is_causal:
        causal_mask = torch.triu(torch.ones(L, L, device=attn.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))

    # apply external mask
    if attn_mask is not None:
        attn = attn + attn_mask  # 注意：attn_mask 应为 log-space (即 -inf 表示屏蔽)

    # Softmax
    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)  # [B, H, L, D]
    return output

class QKVAttention(nn.Module):
    def __init__(self,
                 dim       :int,
                 qkv_bias  :bool  = False,
                 num_heads :int   = 8,
                 attn_drop :float = 0.,
                 proj_drop :float = 0.,
                 ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        # --------------- Basic parameters ---------------
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # --------------- Network parameters ---------------
        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor,):
        bs, c, h, w = x.shape

        # reshape [bs, c, h, w] -> [bs, hw, c]
        x_flatten = x.flatten(2).permute(0, 2, 1).contiguous()
        bs, seq_len, _ = x_flatten.shape

        # ----------------- Input proj -----------------
        qkv = self.qkv(x_flatten)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        ## [bs, seq_len, c] -> [bs, seq_len, nh, dh], c = nh x dh
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, seq_len, self.num_heads, self.head_dim)
        v = v.view(bs, seq_len, self.num_heads, self.head_dim)

        # [bs, seq_len, nh, dh] -> [bs, nh, seq_len, dh]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ----------------- Multi-head Attn -----------------
        try:
            x_flatten = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                is_causal=False,
            )
        except:
            x_flatten = scaled_dot_product_attention_fallback(
            q, k, v,
            attn_mask=None,
            is_causal=False
            )
        x_flatten = self.attn_drop(x_flatten)

        # ----------------- Output -----------------
        # [bs, nh, l, dh] -> [bs, l, nh, dh] -> [bs, l, c]
        x_flatten = x_flatten.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, -1)
        x_flatten = self.proj(x_flatten)
        x_flatten = self.proj_drop(x_flatten)
        x = x_flatten.permute(0, 2, 1).contiguous().view(bs, c, h, w)

        return x

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)

        return x


# ------------------- Resamplors -------------------
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor: float = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.proj_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Conv2d):
            # initialize weight
            init_w = weight_init(
                shape = module.weight.shape,
                mode = "kaiming_uniform",
                fan_in = self.in_channels*3*3,
                fan_out = self.out_channels*3*3,
                )
            init_w *= np.sqrt(1/ 3.0)
            module.weight = torch.nn.Parameter(init_w, requires_grad=True)

            # initialize bias
            if module.bias is not None:
                init_b = weight_init(
                    shape = module.bias.shape,
                    mode = "kaiming_uniform",
                fan_in = self.in_channels*3*3,
                fan_out = self.out_channels*3*3,
                    )
                init_b *= np.sqrt(1/ 3.0)
                module.bias = torch.nn.Parameter(init_b, requires_grad=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        x = self.proj_conv(x)

        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride: int = 2,):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.proj_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Conv2d):
            # initialize weight
            init_w = weight_init(
                shape = module.weight.shape,
                mode = "kaiming_uniform",
                fan_in = self.in_channels*3*3,
                fan_out = self.out_channels*3*3,
                )
            init_w *= np.sqrt(1/ 3.0)
            module.weight = torch.nn.Parameter(init_w, requires_grad=True)

            # initialize bias
            if module.bias is not None:
                init_b = weight_init(
                    shape = module.bias.shape,
                    mode = "kaiming_uniform",
                fan_in = self.in_channels*3*3,
                fan_out = self.out_channels*3*3,
                    )
                init_b *= np.sqrt(1/ 3.0)
                module.bias = torch.nn.Parameter(init_b, requires_grad=True)

    def forward(self, x):
        x = self.proj_conv(x)

        return x


# ------------------- Residual Module -------------------
class ResBlock(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        use_attention: bool = False,
        adaptive_scale: bool = True,
        channels_per_head: int = 64,
        dropout:float = 0,
    ):
        super().__init__()
        assert in_channels == out_channels, "in_channels: {} | out_channels: {}".format(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.use_attention = use_attention
        self.channels_per_head = channels_per_head
        self.dropout = dropout
        self.adaptive_scale = adaptive_scale

        self.norm0 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv0 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = True,
            )

        self.affine = nn.Linear(
            in_features = emb_channels,
            out_features = out_channels * (2 if adaptive_scale else 1),
            )

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv1 = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = True,
            )

        if use_attention:
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
            self.qkv_attn = QKVAttention(
                dim = out_channels,
                qkv_bias = True,
                num_heads = min(out_channels // channels_per_head, 1),
                attn_drop = dropout,
                proj_drop = dropout,
            )

        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Conv2d):
            # initialize weight
            init_w = weight_init(
                shape = module.weight.shape,
                mode = "kaiming_uniform",
                fan_in = self.in_channels*3*3,
                fan_out = self.out_channels*3*3,
                )
            init_w *= np.sqrt(1/ 3.0)
            module.weight = torch.nn.Parameter(init_w, requires_grad=True)

            # initialize bias
            if module.bias is not None:
                init_b = weight_init(
                    shape = module.bias.shape,
                    mode = "kaiming_uniform",
                fan_in = self.in_channels*3*3,
                fan_out = self.out_channels*3*3,
                    )
                init_b *= np.sqrt(1/ 3.0)
                module.bias = torch.nn.Parameter(init_b, requires_grad=True)

        if isinstance(module, nn.Linear):
            # initialize weight
            init_w = weight_init(
                shape = module.weight.shape,
                mode = "kaiming_uniform",
                fan_in = self.emb_channels,
                fan_out = self.out_channels * (2 if self.adaptive_scale else 1),
                )
            init_w *= np.sqrt(1/ 3.0)
            module.weight = torch.nn.Parameter(init_w, requires_grad=True)

            # initialize bias
            if module.bias is not None:
                init_b = weight_init(
                    shape = module.bias.shape,
                    mode = "kaiming_uniform",
                fan_in = self.emb_channels,
                fan_out = self.out_channels * (2 if self.adaptive_scale else 1),
                    )
                init_b *= np.sqrt(1/ 3.0)
                module.bias = torch.nn.Parameter(init_b, requires_grad=True)

    def forward(self, x, emb):
        inp = x
        # -------- conv block 1 --------
        x = F.silu(self.norm0(x))
        x = self.conv0(x)

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = F.silu(self.norm1(x + params))

        # -------- conv block 2 --------
        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
        x = x + inp

        if self.use_attention:
            inp = x
            x = self.norm2(x)
            x = self.qkv_attn(x)
            x = x + inp

        return x

class ResStage(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        use_attention: bool = False,
        channels_per_head: int = 64,
        dropout:float = 0,
        adaptive_scale: bool = True,
        num_blocks: int = 1,
    ):
        super().__init__()
        assert in_channels == out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.channels_per_head = channels_per_head
        self.dropout = dropout

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                ResBlock(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    emb_channels = emb_channels,
                    use_attention = use_attention,
                    adaptive_scale = adaptive_scale,
                    channels_per_head = channels_per_head,
                    dropout = dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, emb):
        for block in self.blocks:
            x = block(x, emb)
        return x


# ------------------- Simple UNet -------------------
class UNet(torch.nn.Module):
    def __init__(self,
        img_resolution: int,      # Image resolution at input/output.
        in_channels: int,         # Number of color channels at input.
        out_channels: int,        # Number of color channels at output.
        label_dim: int = 0,       # Number of class labels, 0 = unconditional.
        augment_dim: int = 0,     # Augmentation label dimensionality, 0 = no augmentation.

        model_channels: int = 192,               # Base multiplier for the number of channels.
        channel_mult: List = [1, 2, 3, 4],       # Per-resolution multipliers for the number of channels.
        channel_mult_emb:int = 4,                # Multiplier for the dimensionality of the embedding vector.
        channel_mult_noise  = 1,                 # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        num_blocks: int = 3,                     # Number of residual blocks per resolution.
        attn_resolutions: List = [32, 16, 8],    # List of resolutions with self-attention.
        dropout: float = 0.10,                   # List of resolutions with self-attention.
        label_dropout: float = 0,                # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.label_dropout = label_dropout

        emb_channels = model_channels * channel_mult_emb
        self.emb_channels = emb_channels
        noise_channels = model_channels * channel_mult_noise
        self.noise_channels = noise_channels

        # Mapping.
        self.map_noise   = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        self.map_label   = nn.Linear(label_dim, emb_channels, bias=False) if label_dim else None
        self.map_augment = nn.Linear(augment_dim, noise_channels * 2, bias=False) if augment_dim else None
        self.map_layer0  = nn.Linear(noise_channels * 2, emb_channels)
        self.map_layer1  = nn.Linear(emb_channels, emb_channels)

        # Encoder.
        self.enc = nn.ModuleDict()
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = in_channels
                cout = model_channels * mult
                self.enc[f'enc_conv_l{level}'] = nn.Conv2d(cin, cout, kernel_size= 3, padding=1, stride=1, bias=True)
                self.enc[f'enc_stage_l{level}'] = ResStage(cout, cout, emb_channels=emb_channels, use_attention = (res in attn_resolutions), channels_per_head=64, dropout=dropout, num_blocks=num_blocks)
            else:
                cin = cout
                cout = model_channels * mult
                self.enc[f'enc_down_l{level}'] = Downsample(cin, cout, stride=2)
                self.enc[f'enc_stage_l{level}'] = ResStage(cout, cout, emb_channels=emb_channels, use_attention = (res in attn_resolutions), channels_per_head=64, dropout=dropout, num_blocks=num_blocks)

        # Decoder.
        self.dec = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            lid = len(channel_mult) - 1 - level
            if level == len(channel_mult) - 1:
                self.dec[f'dec_l{lid}'] = ResStage(cout, cout, emb_channels=emb_channels, use_attention = (res in attn_resolutions), channels_per_head=64, dropout=dropout, num_blocks=num_blocks)
            else:
                cin = cout
                cout = model_channels * mult
                self.dec[f'dec_up_l{lid}'] = Upsample(cin, cout, scale_factor=2.0)
                self.dec[f'dec_stage_l{lid}'] = ResStage(cout, cout, emb_channels=emb_channels, use_attention = (res in attn_resolutions), channels_per_head=64, dropout=dropout, num_blocks=num_blocks)

        # output projection
        self.out_conv = nn.Conv2d(cout, out_channels, kernel_size=3, padding=1, stride=1)

        # initialize nn.Conv2d
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Conv2d):
            # initialize weight
            weight_shape = module.weight.shape
            init_w = weight_init(
                shape = module.weight.shape,
                mode = "kaiming_uniform",
                fan_in = weight_shape[1]*weight_shape[2]*weight_shape[3],
                fan_out = weight_shape[0]*weight_shape[2]*weight_shape[3],
                )
            init_w *= np.sqrt(1/ 3.0)
            module.weight = torch.nn.Parameter(init_w, requires_grad=True)

            # initialize bias
            if module.bias is not None:
                init_b = weight_init(
                    shape = module.bias.shape,
                    mode = "kaiming_uniform",
                    fan_in = weight_shape[1]*weight_shape[2]*weight_shape[3],
                    fan_out = weight_shape[0]*weight_shape[2]*weight_shape[3],
                    )
                init_b *= np.sqrt(1/ 3.0)
                module.bias = torch.nn.Parameter(init_b, requires_grad=True)

    def forward(self, x, time_steps, aug_cond=None, class_labels=None):
        augment_labels = aug_cond

        assert type(time_steps) is tuple and len(time_steps) == 2, "time_steps must be a tuple of (t, h) where t is the current time step and h is another time condiion."
        t, h = time_steps

        # Mapping.
        emb_t = self.map_noise(t)
        emb_t = emb_t.reshape(emb_t.shape[0], 2, -1).flip(1).reshape(*emb_t.shape) # swap sin/cos
        emb_h = self.map_noise(h)
        emb_h = emb_h.reshape(emb_h.shape[0], 2, -1).flip(1).reshape(*emb_h.shape) # swap sin/cos
        emb = torch.cat([emb_t, emb_h], dim=1)

        # Augment embeddings
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = F.silu(self.map_layer0(emb))
        emb = F.silu(self.map_layer1(emb))

        # class condition embeddings
        if self.map_label is not None and class_labels is not None:
            # raise NotImplementedError
            tmp = class_labels  # shape of [batch_size, num_classes]
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
            emb = F.silu(emb)

        # apply encoder layers.
        skips = []
        for block in self.enc.values():
            if isinstance(block, ResStage):
                x = block(x, emb)
                skips.append(x)
            else:
                x = block(x)

        # apply decoder layers.
        x = skips.pop()
        for block in self.dec.values():
            if isinstance(block, ResStage):
                x = block(x, emb)
            else:
                x = block(x)
                x = x + skips.pop()

        # final conv
        x = self.out_conv(F.silu(x))

        return x


if __name__ == "__main__":
    # setup model
    model = UNet(
        img_resolution = 32,
        in_channels = 3,
        out_channels = 3,
        model_channels = 64,
        channel_mult = [1, 2, 3, 4],
        channel_mult_emb = 4,
        channel_mult_noise = 2,
        num_blocks = 3,
        attn_resolutions = [8, 4],
        dropout = 0.1,
        label_dropout = 0.0,
    )
    model.eval()

    # random input data
    x = torch.randn(5, 3, 32, 32)
    t = torch.randn(5,).view(-1, 1, 1, 1)
    r = torch.randn(5,).view(-1, 1, 1, 1)
    h = t - r
    aug_cond = None

    # model inference
    out = model(x, (t.view(-1,), r.view(-1)), aug_cond)
    print(out.shape)
