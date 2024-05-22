import torch.nn as nn
from .mlp import MLPBase


class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=recurrent_N)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                if use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs):
        self.rnn.flatten_parameters()
        x, hxs = self.rnn(x, hxs)
        x = self.norm(x)
        return x, hxs[0, :, :]


class RNNBase(MLPBase):
    def __init__(self, args, inputs_dim):
        super(RNNBase, self).__init__(args, inputs_dim)

        self._recurrent_N = args.recurrent_N

        self.rnn = RNNLayer(
            self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal
        )

    def forward(self, x, hxs):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        if self._use_conv1d:
            batch_size = x.size(0)
            x = x.view(batch_size, self._stacked_frames, -1)
            x = self.conv(x)
            x = x.view(batch_size, -1)

        x = self.mlp(x)

        x, hxs = self.rnn(x, hxs)

        return x, hxs
