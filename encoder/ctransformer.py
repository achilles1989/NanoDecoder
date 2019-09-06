"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

import onmt
from onmt.encoders.encoder import EncoderBase
# from onmt.utils.misc import aeq
from encoder.resnet_encoder import BasicBlock,ResNet
from onmt.modules.position_ffn import PositionwiseFeedForward
from encoder.transformer import TransformerEncoderLayer


class CTransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, cnn_kernel_width, num_layers, d_model, heads, d_ff,
                 dropout, embeddings):
        super(CTransformerEncoder, self).__init__()

        self.cnn = ResNet(
            BasicBlock, cnn_kernel_width, num_classes=d_model
        )
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.d_model = d_model
        self.transformer = nn.ModuleList(
           [TransformerEncoderLayer(d_model, heads, d_ff, dropout,d_model)
            for _ in range(num_layers)])
        # self.transformer = nn.ModuleList(
        #     [TransformerEncoderLayer(input_size, heads, d_ff, dropout)
        #      for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.linear = nn.Linear(input_size, d_model)

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        t, batch_size, nfft = src.size()

        if self.embeddings:
            emb = self.embeddings(src)
        else:
            # emb = self.linear(src)
            emb_remap = src.transpose(0, 1).transpose(1, 2).contiguous().view(batch_size, nfft, -1, t)
            src = self.cnn(emb_remap)
            emb = src


        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        #padding_idx = self.embeddings.word_padding_idx
        padding_idx = 0
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths
