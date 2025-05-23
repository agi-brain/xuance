import numpy as np
import torch

from torch import nn
try:
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange
except:
    pass
from torch.nn import ModuleList

from xuance.common import Sequence
from xuance.torch import Module

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(Module):
    def __init__(self, *,
                 input_shape,
                 image_patch_size,
                 frame_patch_size,
                 final_dim,
                 embedding_dim,
                 depth,
                 heads,
                 FFN_dim,
                 pool = 'mean',
                 channels = 1,
                 dim_head = 64,
                 dropout = 0.,
                 emb_dropout = 0.,
                 ):
        super().__init__()
        image_size = input_shape[0]
        frames = input_shape[-1]
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(embedding_dim, depth, heads, dim_head, FFN_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, final_dim)
        )

    def forward(self, video):
        video = video.unsqueeze(1)
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

# process the input observations with stacks of 3D-ViT layers
class Basic_ViT(Module):
    def __init__(self,input_shape: Sequence[int],
                 image_patch_size: int,
                 frame_patch_size: int,
                 final_dim: int,
                 embedding_dim: int,
                 depth: int,
                 heads: int,
                 FFN_dim: int,
                 pool='mean',
                 channels=1,
                 dim_head=16,
                 dropout=0.,
                 emb_dropout=0.,
                 device='cpu',
                 **kwargs):
        super(Basic_ViT, self).__init__()
        self.output_shapes = {'state': (final_dim,)}
        self.device = device
        self.model = ViT(input_shape = input_shape,image_patch_size=image_patch_size, frame_patch_size=frame_patch_size, final_dim=final_dim,
                           embedding_dim=embedding_dim, depth=depth, heads=heads, FFN_dim=FFN_dim, pool=pool,
                           channels=channels, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout).to(self.device)
    def forward(self, observations: np.ndarray):
        observations = observations / 255.0
        tensor_observation = torch.as_tensor(observations, dtype=torch.float32,
                                             device=self.device).permute((0, 3, 1, 2))
        return {'state': self.model(tensor_observation)}