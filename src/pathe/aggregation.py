"""
Modules for aggregating representations of arbitrary size.

"""
from typing import Union

import torch
import torch_scatter
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def simple_aggregation(embed_heads: Tensor, embed_tails: Tensor,
                       aggregator: nn.Module):
    """
    Concatenate head and tail embeddings horizontally and perform aggregation.

    Args:
        embed_heads (Tensor): Tensor containing the embeddings of the heads.
        embed_tails (Tensor): Tensor containing the embeddings of the tails.
        aggregator (nn.Module): Aggregator module to perform the aggregation.

    Returns:
        Tensor: Tensor containing the aggregated embeddings.
    """
    all_embeddings = torch.cat([embed_heads, embed_tails], dim=1)
    all_embeddings = aggregator(all_embeddings)
    return all_embeddings


class ContextualAggregator(nn.Module):
    """
    # 1. From num_embeddings to 1 embedding vector applying stage-1 aggregation
    # 2. From 1 embedding we attend to all the other embeddings with stage-2 to yield the context vector
    # 3. Finally, we sum the stage-1 embedding vector with the context vector
    """

    def __init__(self, embedding_dim, aggregator: str, context_heads=0,
                 context_dropout=.0, **kwargs):
        super().__init__()
        if aggregator == "recurrent":
            aggregator = RecurrentAggregator(
                embedding_dim=embedding_dim, **kwargs)
        elif aggregator == "transformer":
            aggregator = TransformerAggregator(
                embedding_dim=embedding_dim, **kwargs)
        elif aggregator == "single_query":
            aggregator = SingleQueryAggregator(
                embedding_dim=embedding_dim, **kwargs)
        elif aggregator == "laf":
            aggregator = LAFAggregator(
                embedding_dim=embedding_dim, **kwargs)
        else:
            raise ValueError("Invalid aggregator type")
        self.aggregator = aggregator  # Stage-1 aggregator
        # Stage-2 aggregators, for the contextualisation embeddings

        self.context_heads = context_heads
        if context_heads > 0 and aggregator != "transformer":
            self.h2tails_contextualiser = torch.nn.MultiheadAttention(
                embed_dim=embedding_dim, num_heads=context_heads,
                dropout=context_dropout, batch_first=True)
            self.t2heads_contextualiser = torch.nn.MultiheadAttention(
                embed_dim=embedding_dim, num_heads=context_heads,
                dropout=context_dropout, batch_first=True)

    def forward(self, embed_heads: Tensor, embed_tails: Tensor):
        # Stage-1 aggregation
        heads_agg = self.aggregator(embed_heads)
        tails_agg = self.aggregator(embed_tails)
        # print(heads_agg.shape, tails_agg.shape)
        # print(embed_heads.shape, embed_tails.shape)
        if self.context_heads > 0:
            # Stage-2 contextualisation
            heads_contextualised, _ = self.h2tails_contextualiser(
                heads_agg.unsqueeze(1), embed_tails, embed_tails)
            tails_contextualised, _ = self.t2heads_contextualiser(
                tails_agg.unsqueeze(1), embed_heads, embed_heads)
            # Sum the stage-1 embeddings with the context vectors
            heads_agg = heads_agg + heads_contextualised.squeeze(1)
            tails_agg = tails_agg + tails_contextualised.squeeze(1)

        return heads_agg, tails_agg


class RecurrentAggregator(nn.Module):
    """
    Implements an encoder as a bidirectional reccurrent network to agrgegate a
    sequence of embedding vectors into a single representation.
    """

    def __init__(self, embedding_dim, num_agg_layers=1, **kwargs):

        super().__init__()
        self.embed_dim = embedding_dim
        self.recurrent_net = nn.LSTM(input_size=embedding_dim,
                                     hidden_size=embedding_dim,
                                     num_layers=num_agg_layers,
                                     batch_first=True,
                                     bidirectional=True,
                                     dropout=0.0)

    def forward(self, sequence_embed: Union[list, Tensor]) -> Tensor:
        if isinstance(sequence_embed, list):  # listed batch of tensors
            sequence_lens = [len(s) for s in sequence_embed]
            sequence_embed = pad_sequence(sequence_embed, batch_first=True)
        else:  # single tensor with padding
            sequence_lens = (torch.abs(sequence_embed).sum(dim=-1) != 0).sum(
                dim=-1).to('cpu')
        # Pack the sequence to remove padding and feed to the recurrent model
        # assert (sequence_lens == 0).nonzero().size()[0] == 0, \
        #     ("the sequence lengths cannot be zero")
        sequence_embed = pack_padded_sequence(
            sequence_embed, batch_first=True,
            enforce_sorted=False, lengths=sequence_lens)
        # Feed the sequence batch to the recurrent model and retrieve last items
        outputs, (hn, _) = self.recurrent_net(sequence_embed)
        outputs, input_sizes = pad_packed_sequence(outputs, batch_first=True)
        last_seq_idxs = torch.LongTensor([x - 1 for x in input_sizes])
        last_seq_items = outputs[range(outputs.shape[0]), last_seq_idxs, :]
        # Bididrectional outputs and hidden state are aggregated by sum
        if self.recurrent_net.bidirectional:
            aggregation = last_seq_items[:, :self.embed_dim] + \
                          last_seq_items[:, self.embed_dim:]
        else:  # Unidirectional outputs and hidden state are the same
            aggregation = last_seq_items
        # context = torch.sum(hn, dim=0)
        return aggregation


class TransformerAggregator(nn.Module):
    """
    Implements a sequence aggregator by appending an embedding vector to the
    sequence and applying N transformer encoder layers to contextually add
    information. The aggregated vector is the last embedding after the encoder.
    """

    def __init__(self, embedding_dim, num_agg_layers=2, num_agg_heads=1,
                 dropout=.1, activation=F.relu, **kwargs):
        super().__init__()
        # Uses an embedding CLS-alike vector to hold the aggregation
        self.num_agg_layers = num_agg_layers
        self.num_agg_heads = num_agg_heads
        self.agg_cls = nn.Parameter(
            torch.empty(embedding_dim), requires_grad=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_agg_heads,
            dim_feedforward=embedding_dim, dropout=dropout,
            activation=activation, batch_first=True)
        self.sa_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_agg_layers, norm=None)
        # Initialise the aggregation vector with normal distribution
        nn.init.normal_(self.agg_cls, mean=0.0, std=1.00)

    def device(self):
        return self.agg_cls.device

    def forward(self, sequence_embed: Tensor) -> Tensor:
        """
        This method takes a batch of sequences of embeddings and performs:
        1. Appends the aggregation vector to each sequence in the batch.
        2. Creates a mask that hides the last item (the aggregation vector).
        3. Applies the transformer encoder to the sequence, ensuring that
           padded items (0 embedding vectors) are not attended to.
        4. Returns the last item of the aggregated sequence.

        Note: The aggregation vector is obtained from the self.agg_cls attribute.

        """
        agg = self.agg_cls.reshape(1, 1, -1).repeat(sequence_embed.size(0), 1,
                                                    1)
        sequence_embed = torch.cat((sequence_embed, agg), dim=1)
        # Create padding and aggreagation mask for the transformer encoder
        mask_pad = (torch.abs(sequence_embed).sum(
            dim=-1) != 0)  # assume 0 for padding
        mask_agg = torch.ones(sequence_embed.size(0) * self.num_agg_heads,
                              sequence_embed.size(1), sequence_embed.size(1),
                              device=self.device())
        mask_agg[:, :-1, -1] = 0  # the last item is the aggregation vector
        aggregation = self.sa_encoder(sequence_embed, mask=~mask_agg.bool(),
                                      src_key_padding_mask=~mask_pad)
        return aggregation[:, -1, :]


class SingleQueryAggregator(nn.Module):
    """
    A specialised implementation of `TransformerAggregator`` when using a single
    aggregation layer and the learned aggregation embedding as a query vector.
    """

    def __init__(self, embedding_dim, num_agg_heads=1, **kwargs):
        super().__init__()
        # Uses an embedding CLS-alike vector to hold the aggregation
        self.agg_cls = nn.Parameter(
            torch.empty(embedding_dim), requires_grad=True)
        self.mha_layer = nn.MultiheadAttention(
            embedding_dim, num_heads=num_agg_heads, batch_first=True)
        # Initialise the aggregation vector with normal distribution
        nn.init.normal_(self.agg_cls, mean=0.0, std=1.00)

    def forward(self, sequence_embed: Tensor) -> Tensor:
        """
        This method takes a batch of sequences of embeddings and performs a
        multi-head attention step by using the learnable query to attend the
        given embedding sequence. It returns the attended aggregation vector.
        """
        # sequence_embed is of shape (batch_size, seq_length, embedding_dim)
        mask_pad = (sequence_embed.sum(dim=-1) != 0)  # assume 0 for padding
        # Perform MHA using self.agg_cls to attend the sequence_embed
        query = self.agg_cls.reshape(1, 1, -1).repeat(sequence_embed.size(0), 1,
                                                      1)
        aggregation, _ = self.mha_layer(query, sequence_embed, sequence_embed,
                                        key_padding_mask=~mask_pad)
        # print(aggregation.shape)
        return aggregation[:, 0, :]


class LAFAggregator(nn.Module):
    """
    Uses a Learnable Aggregation Function (LAF) to aggregate a sequence of
    embeddings into a single representation. As each LAF unit returns a single
    aggregation, the final projection layer is used to combine the units into
    a single embedding vector of the same dimension as the input embeddings.
    """

    def __init__(self, embedding_dim, laf_units: int = 1,
                 kernel_initializer: str = "random_normal", **kwargs):
        super().__init__()
        self.laf_layer = LAFLayer(units=laf_units,
                                  kernel_initializer=kernel_initializer)
        self.projector = nn.Linear(embedding_dim * laf_units, embedding_dim)

    def forward(self, sequence_embed: Tensor) -> Tensor:
        """
        Aggregates a sequence of embeddings via a Learnable Aggregation Function
        """
        # Infer the sequence length from the data (assuming that the padding is 0)
        seq_lens = (sequence_embed.sum(dim=-1) != 0).sum(dim=-1)
        seq_idxs = torch.arange(seq_lens.size(0),
                                device=seq_lens.device).repeat_interleave(
            seq_lens)
        # Flatten the tensor to a 2D tensor by removing the padded values
        flattened = sequence_embed.view(-1, sequence_embed.size(-1)) \
            [sequence_embed.sum(dim=-1).view(-1) != 0]
        # Apply the LAF layer to the flattened sequence
        laf_output = self.laf_layer(flattened, seq_idxs)
        laf_output = laf_output.view(laf_output.size(0), -1)
        # Project the output to the original embedding dimension
        aggregated = self.projector(laf_output)
        return aggregated


class AbstractLAFLayer(nn.Module):
    """
    Implements an abstract Learnable Aggregation Function (LAF)
    Code adapted from https://github.com/alessandro-t/laf/tree/main
    """

    def __init__(self, **kwargs):
        super(AbstractLAFLayer, self).__init__()
        assert 'units' in kwargs or 'weights' in kwargs

        if 'kernel_initializer' in kwargs.keys():
            assert kwargs['kernel_initializer'] in [
                'random_normal',
                'glorot_normal',
                'he_normal',
                'random_uniform',
                'glorot_uniform',
                'he_uniform']
            self.kernel_initializer = kwargs['kernel_initializer']
        else:
            self.kernel_initializer = 'random_normal'

        if 'weights' in kwargs.keys():
            self.weights = nn.Parameter(kwargs['weights'], requires_grad=True)
            self.units = self.weights.shape[1]
        else:
            self.units = kwargs['units']
            params = torch.empty(12, self.units)
            if self.kernel_initializer == 'random_normal':
                torch.nn.init.normal_(params)
            elif self.kernel_initializer == 'glorot_normal':
                torch.nn.init.xavier_normal_(params)
            elif self.kernel_initializer == 'he_normal':
                torch.nn.init.kaiming_normal_(params)
            elif self.kernel_initializer == 'random_uniform':
                torch.nn.init.uniform_(params)
            elif self.kernel_initializer == 'glorot_uniform':
                torch.nn.init.xavier_uniform_(params)
            elif self.kernel_initializer == 'he_uniform':
                torch.nn.init.kaiming_uniform_(params)
            self.weights = nn.Parameter(params, requires_grad=True)
        e = torch.tensor([1, -1, 1, -1], dtype=torch.float32)
        self.e = nn.Parameter(e, requires_grad=False)
        num_idx = torch.tensor([1, 1, 0, 0], dtype=torch.float32).view(1, 1, -1,
                                                                       1)
        self.num_idx = nn.Parameter(num_idx, requires_grad=False)
        den_idx = torch.tensor([0, 0, 1, 1], dtype=torch.float32).view(1, 1, -1,
                                                                       1)
        self.den_idx = nn.Parameter(den_idx, requires_grad=False)


class LAFLayer(AbstractLAFLayer):
    """
    Implements a Learnable Aggregation Function (LAF) layer
    Code adapted from https://github.com/alessandro-t/laf/tree/main

    See also
    --------
    https://arxiv.org/abs/2012.08482

    """

    def __init__(self, eps=1e-7, **kwargs):
        super(LAFLayer, self).__init__(**kwargs)
        self.eps = eps

    def forward(self, data, index, dim=0, **kwargs):
        eps = self.eps
        sup = 1.0 - eps
        e = self.e

        x = torch.clamp(data, eps, sup)
        x = torch.unsqueeze(x, -1)
        e = e.view(1, 1, -1)

        exps = (1. - e) / 2. + x * e
        exps = torch.unsqueeze(exps, -1)
        exps = torch.pow(exps, torch.relu(self.weights[0:4]))

        scatter = torch_scatter.scatter_add(exps, index.view(-1), dim=dim)
        scatter = torch.clamp(scatter, eps)

        sqrt = torch.pow(scatter, torch.relu(self.weights[4:8]))
        alpha_beta = self.weights[8:12].view(1, 1, 4, -1)
        terms = sqrt * alpha_beta

        num = torch.sum(terms * self.num_idx, dim=2)
        den = torch.sum(terms * self.den_idx, dim=2)

        multiplier = 2.0 * torch.clamp(torch.sign(den), min=0.0) - 1.0

        den = torch.where((den < eps) & (den > -eps), multiplier * eps, den)

        res = num / den
        return res


class LAFLayerFast(AbstractLAFLayer):
    def __init__(self, eps=1e-3, **kwargs):
        super(LAFLayerFast, self).__init__(**kwargs)
        self.eps = eps

    def forward(self, data, **kwargs):
        eps = self.eps
        sup = 1.0 - eps
        e = self.e

        x = torch.clamp(data, eps, sup)
        x = torch.unsqueeze(x, -1)
        e = e.view(1, 1, -1)

        exps = (1. - e) / 2. + x * e
        exps = torch.unsqueeze(exps, -1)
        exps = torch.pow(exps, torch.relu(self.weights[0:4]))

        # exps has dim batch_size, seq_length, feature_dim
        scatter = torch.sum(exps, dim=1)
        scatter = torch.clamp(scatter, eps)

        sqrt = torch.pow(scatter, torch.relu(self.weights[4:8]))
        alpha_beta = self.weights[8:12].view(1, 1, 4, -1)
        terms = sqrt * alpha_beta

        num = torch.sum(terms * self.num_idx, dim=2)
        den = torch.sum(terms * self.den_idx, dim=2)

        multiplier = 2.0 * torch.clamp(torch.sign(den), min=0.0) - 1.0

        den = torch.where((den < eps) & (den > -eps), multiplier * eps, den)

        res = num / den
        return res


# Generate some tests for all the aggregators using sample data
if __name__ == "__main__":
    # Generate some sample data
    batch_size = 3
    seq_length = 5
    embedding_dim = 4
    sequence_embed = torch.rand((batch_size, seq_length, embedding_dim))
    sequence_embed[0, 3:, :] = 0  # simulate padding
    sequence_embed[1, 4:, :] = 0  # simulate padding
    print(sequence_embed.shape)

    # Test the RecurrentAggregator
    print("\nTesting RecurrentAggregator")
    recurrent_agg = RecurrentAggregator(embedding_dim=embedding_dim)
    aggregation = recurrent_agg(sequence_embed)
    assert aggregation.shape == (batch_size, embedding_dim), \
        "Aggregation shape does not match batch_size * embedding_dim"
    print(aggregation.shape)

    # Test the TransformerAggregator
    print("\nTesting TransformerAggregator")
    transformer_agg = TransformerAggregator(embedding_dim=embedding_dim)
    aggregation = transformer_agg(sequence_embed)
    assert aggregation.shape == (batch_size, embedding_dim), \
        "Aggregation shape does not match batch_size * embedding_dim"
    print(aggregation.shape)

    # Test the SingleQueryAggregator
    print("\nTesting SingleQueryAggregator")
    single_query_agg = SingleQueryAggregator(embedding_dim=embedding_dim)
    aggregation = single_query_agg(sequence_embed)
    assert aggregation.shape == (batch_size, embedding_dim), \
        "Aggregation shape does not match batch_size * embedding_dim"
    print(aggregation.shape)

    # Test the LAFAggregator
    print("\nTesting LAFAggregator")
    laf_agg = LAFAggregator(embedding_dim=embedding_dim, laf_units=2)
    aggregation = laf_agg(sequence_embed)
    assert aggregation.shape == (batch_size, embedding_dim), \
        "Aggregation shape does not match batch_size * embedding_dim"
    print(aggregation.shape)

    # Test the ContextualAggregator
    print("\nTesting ContextualAggregator")
    for aggregator in ["recurrent", "transformer", "single_query", "laf"]:
        print(f"Testing {aggregator} aggregator")
        contextual_agg = ContextualAggregator(embedding_dim=embedding_dim,
                                              aggregator=aggregator)
        heads_final, tails_final = contextual_agg(sequence_embed,
                                                  sequence_embed)
        assert heads_final.shape == (batch_size, embedding_dim), \
            "Aggregation shape does not match batch_size * embedding_dim"
        print(heads_final.shape)
        assert tails_final.shape == (batch_size, embedding_dim), \
            "Aggregation shape does not match batch_size * embedding_dim"
        print(tails_final.shape)
