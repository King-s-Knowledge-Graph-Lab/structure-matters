"""
PathER : the Path Embedder

This module wraps different architectures for the path embedder, ranging from
self-attention networks inspired by NLP models (BART, GPT, etc.) to denoising
architectures attempting to reconstruct an input sequence or classify the type
of transformation(s) operated on the (discrete) signal.
"""
import math
import logging
import warnings
from typing import Optional, Any, Union, Callable, List, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .aggregation import ContextualAggregator, simple_aggregation

logger = logging.getLogger(__name__)


def init_transformer_params(module):
    """
    Initialise parameters in a transformer model.
    """
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def init_bert_params(module):
    """
    Initialise the weights specific to the BERT Model.
    This overrides the default init depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).

    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, nn.MultiheadAttention):
        if module._qkv_same_embed_dim:  # torch adpt
            normal_(module.in_proj_weight.data)
        else:  # this is as done in BART (FAIR)
            normal_(module.q_proj_weight.data)
            normal_(module.k_proj_weight.data)
            normal_(module.v_proj_weight.data)


class PositionalEncoding(nn.Module):
    """
    Vanilla Positional Encoding as introduced in the Transformer. Code borrows
    from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Positional encodings here are implemented as a transformation expecting the
    input sequence to be already embedding before positional info is injected.
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1,
                 max_len: int = 5000, batch_first: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Create mockup sequence to serve as additive tensor
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2)
                             * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            A sequence of shape ``[seq_len, bsz, embedding_dim]`` assuming that
            `batch_first = False` else ``[bsz, seq_len, embedding_dim]``
        """
        pos = self.pe[:x.size(0)] if not self.batch_first \
            else torch.swapaxes(self.pe[:x.size(1)], 0, 1)
        x = x + pos

        return self.dropout(x)


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module serves as an alternative to vanilla positional encodings, by
    serving positional embeddings up to a fixed maximum size. For convenience,
    this is already provided as a specialised embedding layer.

    Parameters
    ----------
    num_embeddings : int
        This should correspond to the size of the longest sequence to embed.
    embedding_dim : int
        The size of each embedding vector.
    padding_idx : int
        Index of padding tokens to initialiase the embedding table.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """
        Parameters
        ----------
        x: Tensor
            A sequence of shape ``[bsz, seq_len]`` to embed.
        past_key_values_length : int
            Acts as a positional offset to this sequence, which is supposed to
            be a continuation of another sequence previosuly fed.
        """
        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len,
            dtype=torch.int, device=self.weight.device,
        ).expand(bsz, -1)

        return super().forward(positions)


class RelationPredictorNNModel(nn.Module):
    """
    A FC layer that performs relation prediction

    Parameters
    ----------
    d_input : int
        The number of expected features in the input as embedding dim.
    output_dim : int
        The number of output neurons (predicted classes - in our case
        relations).
    dim_layers: List[int]
        A list with the dimensions of the hidden layers- default 1 layer of
        512 neurons
    dropout : float
        Dropout value to use in the encoder model (default=0.1).
    activation : Union[str, Callable[[Tensor], Tensor]]
        The activation function of the intermediate layer, can be a string
        ("relu" or "gelu") or a unary callable. Default: relu
    """

    def __init__(self,
                 output_dim: int,
                 d_input: int = None,
                 dim_layers: List = None,
                 dropout: float = 0.1,
                 activation: Union[str, Callable] = F.relu) -> None:
        super().__init__()
        self.dim_layers = dim_layers if dim_layers is not None else [512]
        self.d_input = d_input
        self.dropout_p = dropout
        self.output_dim = output_dim
        self.activation_f = activation
        self.model = nn.ModuleList()
        self.init_model()
        self.model.apply(self.init_weights)

    def init_model(self):
        """
        A function that initializes the MLP layers
        @return: None
        @rtype: None
        """

        # if the input dimension is given initialize the layers
        # according to the dimensions provided
        if self.d_input:
            current_dim = self.d_input
            self.model = nn.ModuleList()
            for hdim in self.dim_layers:
                layer = nn.Linear(current_dim, hdim)
                block = nn.Sequential(layer, nn.Dropout(self.dropout_p))
                self.model.append(block)
                current_dim = hdim
            layer = nn.Linear(current_dim, self.output_dim)
            self.model.append(layer)
        # if the input is missing use Lazy layers that will initialize on
        # first forward pass
        else:
            self.model = nn.ModuleList()
            for hdim in self.dim_layers:
                block = nn.Sequential(nn.LazyLinear(hdim),
                                      nn.Dropout(self.dropout_p))
                self.model.append(block)
            self.model.append(nn.LazyLinear(self.output_dim))

    def init_weights(self, module: nn.Module, initialization: Union[str,
    Callable] =
    nn.init.xavier_uniform) -> None:
        """
        Initialisation taken from:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        for the default.
        Parameters
        ----------
        module: nn.Module
            The module to initialize
        initialization: Union[str, Callable]
            The initialization function to use for the layer weights.
        """
        if initialization == 'default':
            initrange = 0.1
            module.bias.data.zero_()
            module.weight.data.uniform_(-initrange, initrange)
        else:
            initialization(module)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : torch.tensor
            A batch of inputs of shape ``[bsz, input dim]``

        Returns
        -------
        logits : torch.tensor
            A tensor of shape ``[bsz, num_relations]`` with predictions

        """
        for layer in self.model[:-1]:
            x = layer(x)
            x = self.activation_f(x)
        out = F.softmax(self.layers[-1](x))
        return out


class AdaptiveNodeEncoder(nn.Module):
    """
    A FC layer that performs relation prediction

    Parameters
    ----------
    d_input : int
        The number of expected features in the input as embedding dim.
    output_dim : int
        The number of output neurons (predicted classes - in our case
        relations).
    dim_layers: List[int]
        A list with the dimensions of the hidden layers- default 1 layer of
        512 neurons
    dropout : float
        Dropout value to use in the encoder model (default=0.1).
    activation : Union[str, Callable[[Tensor], Tensor]]
        The activation function of the intermediate layer, can be a string
        ("relu" or "gelu") or a unary callable. Default: relu
    """

    def __init__(self,
                 encodings: torch.tensor,
                 d_output: int,
                 edge_features: torch.tensor = None,
                 dropout: float = 0.1,
                 projector: str = "GCN",
                 activation: Union[str, Callable] = F.leaky_relu) -> None:
        super().__init__()
        self.mlp2 = None
        self.mlp1 = None
        self.projector = projector
        self.d_input = d_output
        self.dropout_p = dropout
        self.incoming_gcn = None
        self.outgoing_gcn = None
        self.activation_f = activation
        self.init_model(encodings, edge_features)

    def init_model(self, encodings, edge_features):
        """
        A function that initializes the MLP layers
        @return: None
        @rtype: None
        """
        encoding_inc = encodings[1]
        encoding_out = encodings[2]
        if self.projector == "GCN":
            self.incoming_gcn = EdgeGCN(encodings=encoding_inc,
                                        edge_features=edge_features,
                                        d_out=self.d_input)
            self.outgoing_gcn = EdgeGCN(encodings=encoding_out,
                                        edge_features=edge_features,
                                        d_out=self.d_input)
        else:
            self.incoming_gcn = DummyProjector(encodings=encoding_inc,
                                        d_out=self.d_input)
            self.outgoing_gcn = DummyProjector(encodings=encoding_out,
                                        d_out=self.d_input)
        self.mlp1 = nn.Linear(
            self.d_input * 2,
            self.d_input * 2)
        self.mlp2 = nn.Linear(
            self.d_input * 2,
            self.d_input)

    def set_encodings(self, encodings):
        """
        Replaces the relational-context counts of the projectors with those of
        another graph (the output of ``encode_relcontext_freqs``); the number
        of relations must be the same, the number of entities may differ.
        """
        for projector, encoding in ((self.incoming_gcn, encodings[1]),
                                    (self.outgoing_gcn, encodings[2])):
            if encoding.shape[1] != projector.encodings.shape[1]:
                raise ValueError("Relational contexts over different relations")
            projector.encodings = encoding.to(projector.encodings.device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : torch.tensor
            A batch of inputs of shape `[num_paths, path length]'

        Returns
        -------
        logits : torch.tensor
            A tensor of shape ``[num paths, path length, embedding dimension]``
            with embeddings

        """
        if x.ndim < 2:
            x.unsqueeze(0)
        inc_encodings = self.incoming_gcn(x)
        out_encodings = self.outgoing_gcn(x)
        logits = self.mlp1(torch.cat((inc_encodings, out_encodings), dim=1))
        logits = self.activation_f(logits)
        logits = self.mlp2(logits)
        embeddings = self.activation_f(logits)
        embeddings = embeddings.reshape((x.size()[0], x.size()[1], -1))
        return embeddings


class EdgeGCN(nn.Module):
    """
    A FC layer that performs relation prediction

    Parameters
    ----------
    d_input : int
        The number of expected features in the input as embedding dim.
    output_dim : int
        The number of output neurons (predicted classes - in our case
        relations).
    dim_layers: List[int]
        A list with the dimensions of the hidden layers- default 1 layer of
        512 neurons
    dropout : float
        Dropout value to use in the encoder model (default=0.1).
    activation : Union[str, Callable[[Tensor], Tensor]]
        The activation function of the intermediate layer, can be a string
        ("relu" or "gelu") or a unary callable. Default: relu
    """

    def __init__(self,
                 encodings: torch.tensor,
                 d_out: int,
                 edge_features: torch.tensor = None,
                 dropout: float = 0.1,
                 activation: Union[str, Callable] = F.tanh) -> None:
        super().__init__()
        self.index_tensor = None
        self.attention2 = None
        self.attention = None
        # non-persistent buffers: they follow the module's device and are not
        # part of the state dict (the released checkpoints do not contain them)
        self.register_buffer("encodings", encodings, persistent=False)
        self.edge_features = edge_features
        self.gcn_weights = None
        if isinstance(edge_features, nn.Embedding):
            self.d_out = d_out
            self.register_buffer("embedding_indices", torch.arange(
                edge_features.num_embeddings, dtype=torch.int32),
                persistent=False)
        elif isinstance(edge_features, torch.Tensor):
            self.d_out = d_out
        elif not edge_features:
            self.d_out = d_out
            self.edge_features = nn.Embedding(encodings.size()[1], d_out,
                                              padding_idx=0)
            self.register_buffer("embedding_indices", torch.arange(
                encodings.size()[1], dtype=torch.int32), persistent=False)
        else:
            raise Exception("The edge features are not of a supported dtype. "
                            "Use torch Tensor or nn.Embedding.")
        self.dropout_p = dropout
        self.activation_f = activation
        self.init_model()
        # self.model.apply(self.init_weights)

    def init_model(self):
        """
        A function that initializes the MLP layers
        @return: None
        @rtype: None
        """

        # self.gcn_weights = nn.Parameter(torch.rand((self.d_out,
        #                                             self.d_out),
        #                                            dtype=torch.float32))
        self.gcn_weights = nn.Embedding(self.d_out, self.d_out)
        self.index_tensor = nn.Parameter(torch.arange(self.d_out,
                                                      dtype=torch.int32),
                                         requires_grad=False)
        self.attention = nn.Linear(self.encodings.size()[1],
                                   3 * self.encodings.size()[1])
        self.attention2 = nn.Linear(3 * self.encodings.size()[1],
                                    self.encodings.size()[1])

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : torch.tensor
            A batch of inputs of shape `[num_paths, path length]'

        Returns
        -------
        logits : torch.tensor
            A tensor of shape ``[num_paths * path length, embedding dim]``

        """
        enc = self.encodings[x, :]
        enc_flattened = enc.reshape(enc.size()[0] * enc.size()[1], -1)
        attention = self.attention(enc_flattened.float())
        attention = self.activation_f(attention)
        attention = self.attention2(attention)
        attention_weights = torch.nn.functional.softmax(attention, dim=1)
        # attention_weights = enc_flattened
        if isinstance(self.edge_features, nn.Embedding):
            edge_features = self.edge_features(self.embedding_indices)
            node_features = torch.mm(attention_weights, edge_features)
        else:
            node_features = torch.mm(attention_weights, self.edge_features)
        node_features = torch.mm(node_features, self.gcn_weights(self.index_tensor))
        return node_features

class DummyProjector(nn.Module):
    """
    A FC layer that performs relation prediction

    Parameters
    ----------
    d_input : int
        The number of expected features in the input as embedding dim.
    output_dim : int
        The number of output neurons (predicted classes - in our case
        relations).
    dim_layers: List[int]
        A list with the dimensions of the hidden layers- default 1 layer of
        512 neurons
    dropout : float
        Dropout value to use in the encoder model (default=0.1).
    activation : Union[str, Callable[[Tensor], Tensor]]
        The activation function of the intermediate layer, can be a string
        ("relu" or "gelu") or a unary callable. Default: relu
    """

    def __init__(self,
                 encodings: torch.tensor,
                 d_out: int,
                 activation: Union[str, Callable] = F.tanh) -> None:
        super().__init__()
        self.model = None
        self.register_buffer("encodings", encodings, persistent=False)
        self.activation_f = activation

        self.d_out = d_out
        self.init_model()
        # self.model.apply(self.init_weights)

    def init_model(self):
        """
        A function that initializes the MLP layers
        @return: None
        @rtype: None
        """

        # self.model = nn.Linear(self.encodings.size()[1],
        #           self.d_out)
        self.model = nn.Sequential(
                                    nn.Dropout(0.5),
                                   nn.Linear(self.encodings.size()[1],
                                             self.d_out))
        # self.model = nn.Sequential(nn.Linear(self.encodings.size()[1],
        #                                      self.encodings.size()[1]),
        #                            nn.ReLU(),
        #                            nn.Linear(self.encodings.size()[1],
        #                                      self.d_out),
        #                            # nn.ReLU()
        #                            )



    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : torch.tensor
            A batch of inputs of shape `[num_paths, path length]'

        Returns
        -------
        logits : torch.tensor
            A tensor of shape ``[num_paths * path length, embedding dim]``

        """
        enc = self.encodings[x, :]
        enc_flattened = enc.reshape(enc.size()[0] * enc.size()[1], -1)
        node_features = self.model(enc_flattened)
        # node_features = nn.functional.relu(node_features)
        return node_features


class EmbeddingInjector(nn.Module):
    """
    A model that injects embeddings into pathe

    Parameters
    ----------
    node_indices : torch.tensor
        the offseted indices of the nodes that will be replaced.
    num_dim : int
        The number of dimensions of the output.
    num_nodes: int
        the total number of nodes in the KG
    device : Any
       The device of the model.
    embeddings : torch.tensor
        The tensor of embeddings that correspond to the node_indices
    """

    def __init__(self,
                 node_indices: torch.tensor,
                 num_dim: int,
                 num_nodes: int,
                 device,
                 projector: bool,
                 fusion: str,
                 embeddings: torch.tensor=None):
        super().__init__()
        self.model = None
        self.device = device
        self.projector = projector
        self.embeddings = embeddings
        self.node_indices = node_indices + 2
        self.num_nodes = num_nodes
        self.mapper = None
        self.num_dim = num_dim
        if fusion in ["avg", "sum", 'mult', 'conc', 'replace']:
            self.fusion_type = fusion
        else:
            warnings.warn(f'fusion type {fusion} not supported, '
                          f'defaulting to replacement ')
            self.fusion_type = 'replace'

        self.init_model()
        # self.model.apply(self.init_weights)

    def init_model(self):
        """
        A function that initializes the model
        @return: None
        @rtype: None
        """
        if self.embeddings is None:
            self.mapper = torch.zeros((self.num_nodes + 2), dtype=torch.int32,
                                      requires_grad=False, device=self.device) - 1
            # starts from 1 and goes to num+1 because position 0 will be
            # padding
            node_assignments = torch.arange(1, self.node_indices.size()[0] + 1,
                                            dtype=torch.int32,
                                            requires_grad=False, device=self.device)
            self.mapper[self.node_indices] = node_assignments
            self.model = nn.Embedding(self.node_indices.size()[0]+1,
                                      self.num_dim, 0, device=self.device)
        else:
            # initialize a vector of zeros to store which indices in the
            # embeddings map to which nodes
            self.mapper = torch.zeros((self.num_nodes+2), dtype=torch.int32,
                                      requires_grad=False,device=self.device) - 1
            # create a vector from 0->number of nodes that have pretrained
            # embeddings
            node_assignments = torch.arange(0, self.node_indices.size()[0],
                                            dtype=torch.int32,
                                            requires_grad=False, device=self.device)
            # replace the zeros with the corresponding index in the mapper
            self.mapper[self.node_indices] = node_assignments
            self.mapper = nn.Parameter(self.mapper, requires_grad=False)
            # initialize embedding table (non trainable) and the projector
            self.embeddings = nn.Parameter(self.embeddings,
                                           requires_grad=False)
            if (self.embeddings.size()[1] != self.num_dim) or self.projector:
                self.model = nn.Sequential(nn.Linear(self.embeddings.size()[1],
                                                     self.embeddings.size()[
                                                         1],device=self.device),
                                           nn.ReLU(),
                                           nn.Linear(self.embeddings.size()[1],
                                                     self.num_dim, device=self.device))
        # separate approaches depending on fusion strategy
        if self.fusion_type in ["avg", "sum", 'mult', 'conc']:
            # create a trainable vector corresponding to the unknown entity
            unknown_entity_vec = torch.rand((1,self.num_dim),
                                              device=self.device,
                                              dtype=torch.float32)
            self.unknown_entity = nn.Parameter(unknown_entity_vec,
                                               requires_grad=True)
        else:
            self.unknown_entity = None
        # if the fusion strategy is concatenation we need to half the
        # embedding dimension to match PathE
        if self.fusion_type == 'conc':
            self.dimension_down_projector = nn.Sequential(nn.Linear(
                                                    self.num_dim * 2,
                                                    self.num_dim * 2,
                                                    device=self.device),
                                           nn.ReLU(),
                                           nn.Linear(self.num_dim * 2,
                                                     self.num_dim,
                                                     device=self.device))






    def forward(self, embed_paths: Tensor, entity_paths: Tensor) -> Tensor:
        """
        Parameters
        ----------
        embed_paths : torch.tensor
            A batch of encoded entity paths `[num_paths, path length,
            embed_dim]'
        ent_paths : torch.tensor
            A batch of entity paths with the entities offseted for special
            tokens `[num_paths, path length]'


        Returns
        -------
        logits : torch.tensor
            A tensor of shape ``[num_paths * path length, embedding dim]``

        """
        indices_to_replace = (self.mapper[entity_paths] >= 0)
        unknown_indices = (self.mapper[entity_paths] < 0)
        mapped_indices = self.mapper[entity_paths]
        if self.embeddings is None:
            coordinates = torch.nonzero(indices_to_replace)
            embed_paths[coordinates[:,0],coordinates[:,1]] = self.model[
                                        coordinates[:,0], coordinates[:,1]]
        else:
            # a = mapped_indices[indices_to_replace]
            coordinates = torch.nonzero(indices_to_replace)
            unknown_coordinates = torch.nonzero(unknown_indices)
            # print(coordinates)
            # print(torch.max(self.mapper))
            # new_emb = self.embeddings[mapped_indices[indices_to_replace]]
            # if (self.embeddings.size()[1] != self.num_dim) or self.projector:
            #     new_emb = self.model(new_emb)
            # a = mapped_indices[coordinates[:,0], coordinates[:,1]]
            # b =self.embeddings[mapped_indices[coordinates[:,0], coordinates[
            #                                                      :,1]]]
            # c = self.model(b)
            if self.fusion_type == "avg":
                # first average the pretrained embeddings
                embed_paths[coordinates[:, 0], coordinates[:,
                                                 1]] = torch.div(
                    torch.add(self.model(self.embeddings[mapped_indices[coordinates[:, 0],
                                                   coordinates[:, 1]]]),
                        embed_paths[coordinates[:, 0], coordinates[:, 1]]),2.0)
                # then average the remaining with the trainable unknown tensor
                embed_paths[unknown_coordinates[:, 0], unknown_coordinates[:,
                                                 1]] = torch.div(
                    torch.add(self.unknown_entity,
                        embed_paths[unknown_coordinates[:, 0],
                        unknown_coordinates[:, 1]]),2.0)

            elif self.fusion_type == "sum":
                # first sum the pretrained embeddings
                embed_paths[coordinates[:, 0], coordinates[:,
                                               1]] = torch.add(self.model(
                        self.embeddings[mapped_indices[coordinates[:, 0],
                        coordinates[:, 1]]]),
                              embed_paths[
                                  coordinates[:, 0], coordinates[:, 1]])
                # then sum the remaining with the trainable unknown tensor
                embed_paths[unknown_coordinates[:, 0], unknown_coordinates[:,
                                                       1]] = torch.add(self.unknown_entity,
                              embed_paths[unknown_coordinates[:, 0],
                              unknown_coordinates[:, 1]])
            elif self.fusion_type == "mult":
                # first multiply the pretrained embeddings
                embed_paths[coordinates[:, 0], coordinates[:,
                                               1]] = torch.mul(self.model(
                    self.embeddings[mapped_indices[coordinates[:, 0],
                    coordinates[:, 1]]]),
                    embed_paths[
                        coordinates[:, 0], coordinates[:, 1]])
                # then multiply the remaining with the trainable unknown tensor
                embed_paths[unknown_coordinates[:, 0], unknown_coordinates[:,
                                                       1]] = torch.mul(
                    self.unknown_entity,
                    embed_paths[unknown_coordinates[:, 0],
                    unknown_coordinates[:, 1]])
            elif self.fusion_type == "conc":
                test = torch.cat((self.model(
                    self.embeddings[mapped_indices[coordinates[:, 0],
                    coordinates[:, 1]]]),
                           embed_paths[
                               coordinates[:, 0], coordinates[:, 1]]), dim=1)
                # first concat and down project the pretrained embeddings
                embed_paths[coordinates[:, 0], coordinates[:,
                                               1]] = (
                    self.dimension_down_projector(torch.cat((self.model(
                    self.embeddings[mapped_indices[coordinates[:, 0],
                    coordinates[:, 1]]]),
                    embed_paths[
                        coordinates[:, 0], coordinates[:, 1]]),dim=1)))
                # then concat and down project the remaining with the trainable
                # unknown tensor
                embed_paths[unknown_coordinates[:, 0], unknown_coordinates[:,
                                                       1]] = (
                    self.dimension_down_projector(torch.cat((self.model(
                    self.embeddings[mapped_indices[unknown_coordinates[:, 0],
                    unknown_coordinates[:, 1]]]),
                    embed_paths[
                        unknown_coordinates[:, 0], unknown_coordinates[:,
                                                   1]]),dim=1)))
            else:
                embed_paths[coordinates[:,0],coordinates[:,1]] = self.model(
                    self.embeddings[mapped_indices[coordinates[:,0], coordinates[
                                                                     :,1]]])
        return embed_paths


################################################################################
# PathE models
################################################################################

class PathEModel(nn.Module):
    """
    Parameters
    ----------
    vocab_size : int
        The number of token types that can be expected in input sequences
    relcontext_graph : tuple
        ...
    padding_index : int
        The index of the token used to denote padding in the sequences
    d_model : int
        Number of expected features in the encoder/decoder inputs (default=512)
    nhead : int
        Number of heads in the multiheadattention models (default=8)
    num_encoder_layers : int
        Number of sub-encoder-layers in the encoder (default=6)
    dim_feedforward : int
        Dimension of the feedforward network model (default=2048)
    dropout : float
        The dropout value (default=0.1)
    activation: str
        Activation function of encoder/decoder intermediate layer (default=gelu)
    layer_norm_eps : float
        The eps value in layer normalization components (default=1e-5)
    batch_first : bool
        If ``True``, then the input and output tensors are provided
        as (batch, seq, feature); default=True (batch, seq, feature)
    norm_first : bool
        If ``True``, encoder and decoder layers will perform LayerNorms before
        other attention and feedforward ops, otherwise after (default=false)
    max_seqlen : int
        If provided, it will allocate ``max_seqlen`` positional encodings for
        Entity-Relation paths, and ``math.ceil(max_seqlen / 2)`` for R paths.
    """

    def __init__(self,
                 vocab_size: int,
                 relcontext_graph: tuple,
                 padding_idx: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5,
                 ent_aggregation : str = "avg",
                 num_agg_heads : int = 1,
                 num_agg_layers : int = 1,
                 laf_units : int = 1,
                 context_heads : int = 1,
                 batch_first: bool = True,
                 norm_first: bool = False,
                 node_projector: str = "GCN",
                 injector_params: Dict=None,
                 max_seqlen: int = None,
                ) -> None:

        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.batch_first = batch_first
        assert node_projector in ["GCN", "dummy"]
        self.node_projector = node_projector
        self.ent_aggregation = ent_aggregation

        self.injector_params = injector_params
        
        if ent_aggregation == "avg":
            logger.info("Using average aggregation for entities")
        else:  # assume contextual aggregation
            self.aggregator = ContextualAggregator(
                embedding_dim=d_model, aggregator=ent_aggregation,
                num_agg_heads=num_agg_heads, num_agg_layers=num_agg_layers,
                laf_units=laf_units, context_heads=context_heads,
            )

        max_seqlen = vocab_size if max_seqlen is None else max_seqlen
        # Relational Embedding layers are shared between encoder and decoder;
        # in contrast to vanilla Parth, positional embeddings are not shared.
        self.rel_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        self.pred_link_embeddings = nn.Embedding(
            num_embeddings=vocab_size - 2,
            embedding_dim=d_model,
        )
        self.rel_index_tensor = nn.Parameter(
            torch.arange(vocab_size - 2), requires_grad=False)
        self.pos_embeddings = nn.Embedding(
            num_embeddings=max_seqlen,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        # Transformer layers: encoder and decoder needed for representations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation, layer_norm_eps, batch_first, norm_first)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)
        # Relational context project for init entities
        self.rcontext_projector = AdaptiveNodeEncoder(
            encodings=relcontext_graph, edge_features=None,
            d_output=d_model, dropout=dropout, projector=self.node_projector,
            # activation=  # TODO custom act
        )

        # embedding injector
        if self.injector_params is not None:
            self.injector = EmbeddingInjector(
                node_indices=self.injector_params["node_indices"], num_dim=
                self.d_model, num_nodes=self.injector_params["num_nodes"],
                device=self.device(), projector=True,
                embeddings=self.injector_params["embeddings"],
                fusion=self.injector_params["fusion"])
        self.has_injector = True if self.injector_params is not None else \
            False

        # Relation prediction head from averaged entities
        # self.relpredict_head_avg1 = nn.Linear(d_model * 2, d_model)  # FIXME
        # self.relpredict_head_avg2 = nn.Linear(d_model, vocab_size-2)
        #
        self.relpredict_head_avg = nn.Linear(d_model * 2, vocab_size - 2)


        ###################################################
        # Relation-based agrgeagtion and link prediction 
        # self.relpredict_head_l1 = nn.Parameter(torch.rand((
        #     vocab_size - 2, d_model * 2), dtype=torch.float32))
        #
        # self.gating_layer = nn.Parameter(torch.rand((
        #     vocab_size - 2, d_model * 2), dtype=torch.float32))
        #
        self.link_predict_head1 = nn.Linear(d_model * 3, d_model)
        self.link_predict_head2 = nn.Linear(d_model, 1)
        # XXX Apply custom initialisation
        init_transformer_params(self.encoder)
        # self.apply(init_bert_params)

    def device(self):
        return self.rel_embeddings.weight.device

    def link_prediction(self, heads, tails, relation_scores):
        relation_embeddings = self.pred_link_embeddings(
            self.rel_index_tensor).unsqueeze(0)
        relation_weights = nn.functional.softmax(relation_scores)
        relations_emb = torch.matmul(relation_weights,
                                     relation_embeddings).squeeze()
        heads_emb, tails_emb = self.mha_aggregate(heads, tails, relations_emb)
        triples = torch.cat((heads_emb, relations_emb, tails_emb), dim=1)
        logits = nn.functional.relu(self.link_predict_head1(triples))
        logits = self.link_predict_head2(logits)
        return logits

    def mha_aggregate(self, heads, tails, relations):
        expanded_relation_emb = relations.unsqueeze(1)
        heads_aggregated, _ = self.head_aggregator(
            query=expanded_relation_emb, key=heads,
            value=heads, need_weights=False)
        tails_aggregated, _ = self.tail_aggregator(
            query=expanded_relation_emb, key=tails,
            value=tails, need_weights=False)
        return heads_aggregated.squeeze(), tails_aggregated.squeeze()

    def average_aggregate(self, heads, tails):
        unique_heads, counts_h = torch.unique(heads.nonzero()[:, 0],
                                              return_counts=True)
        unique_tails, count_tails = torch.unique(tails.nonzero()[:, 0],
                                                 return_counts=True)
        count_h, count_t = (counts_h.float() / heads.size()[2],
                            count_tails.float() / tails.size()[2])
        heads_emb = torch.sum(heads, dim=1) / count_h.unsqueeze(1)
        tails_emb = torch.sum(tails, dim=1) / count_t.unsqueeze(1)
        return heads_emb, tails_emb

    def predict_relations_and_average(self, head_emb, tail_embed, ppt):
        head_tail = torch.cat((head_emb, tail_embed), dim=1)
        predictions = self.relpredict_head_avg1(head_tail)
        predictions = nn.functional.relu(predictions)
        predictions = self.relpredict_head_avg2(predictions)
        triple_preds = torch.split(predictions, ppt.tolist(), dim=0)
        padded_triples = nn.utils.rnn.pad_sequence(triple_preds,
                                                  batch_first=True)
        # unique_triples, counts = torch.unique(padded_triples.nonzero()[:, 0],
        #                                       return_counts=True)
        # counts = counts.float() / padded_triples.size()[2]
        averaged_scores = torch.sum(padded_triples, dim=1) / ppt.unsqueeze(1)
        return averaged_scores

    def predict_links_and_average(self, head_emb, tail_embed, ppt):
        head_tail = torch.cat((head_emb, tail_embed), dim=1)
        predictions = self.linkpred_head_avg1(head_tail)
        predictions = nn.functional.relu(predictions)
        predictions = self.linkpred_head_avg2(predictions)
        triple_preds = torch.split(predictions, ppt.tolist(), dim=0)
        padded_triples = nn.utils.rnn.pad_sequence(triple_preds,
                                                   batch_first=True)
        # unique_triples, counts = torch.unique(padded_triples.nonzero()[:, 0],
        #                                       return_counts=True)
        # counts = counts.float() / padded_triples.size()[2]
        averaged_scores = torch.sum(padded_triples, dim=1) / ppt.unsqueeze(1)
        return averaged_scores

    def average_embeddings(self, embeddings, ppt, idxs):
        """
        A function that averages the embeddings in a table for each triple
        Parameters
        ----------
        embeddings : The emebeddings of shape (num_entities, dim)
        ppt : The paths per triple
        idxs : The idxs of the entities in each path

        Returns
        -------
        A single embedding table of shape (num triples, dim)
        """
        # hack so that the only indexes that are negative are the non-existent
        idxs_local= idxs + 1
        # split the idxs for each triple (numtriple, numpath)
        split_idxs = torch.split(idxs_local, ppt.tolist(), dim=0)
        padded_idxs = nn.utils.rnn.pad_sequence(split_idxs, batch_first=True)
        # get the positive indexes (true ones)
        actual_idxs = (padded_idxs > 0).nonzero()
        # get their number per triple
        unique, counts = torch.unique(actual_idxs[:,0], return_counts=True)

        # split the embeddings based on the existing number of entities per
        # triple
        triple_embeds = torch.split(embeddings, counts.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(triple_embeds,
                                                   batch_first=True)
        # average ignoring padding
        averaged_embs = torch.sum(padded_embs, dim=1) / counts.unsqueeze(1)
        return averaged_embs

    def sum_embeddings(self, embeddings, ppt, idxs):
        """
        A function that averages the embeddings in a table for each triple
        Parameters
        ----------
        embeddings : The emebeddings of shape (num_entities, dim)
        ppt : The paths per triple
        idxs : The idxs of the entities in each path

        Returns
        -------
        A single embedding table of shape (num triples, dim)
        """
        # hack so that the only indexes that are negative are the non-existent
        idxs_local= idxs + 1
        # split the idxs for each triple (numtriple, numpath)
        split_idxs = torch.split(idxs_local, ppt.tolist(), dim=0)
        padded_idxs = nn.utils.rnn.pad_sequence(split_idxs, batch_first=True)
        # get the positive indexes (true ones)
        actual_idxs = (padded_idxs > 0).nonzero()
        # get their number per triple
        unique, counts = torch.unique(actual_idxs[:,0], return_counts=True)

        # split the embeddings based on the existing number of entities per
        # triple
        triple_embeds = torch.split(embeddings, counts.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(triple_embeds,
                                                   batch_first=True)
        # average ignoring padding
        summed_embs = torch.sum(padded_embs, dim=1)
        return summed_embs

    def predict_relation_from_ht(self, head_emb, tail_embed):
        head_tail = torch.cat((head_emb, tail_embed), dim=1)
        predictions = self.relpredict_head_avg(head_tail)
        # predictions = self.relpredict_head_avg1(head_tail)
        # predictions = nn.functional.relu(predictions)
        # predictions = self.relpredict_head_avg2(predictions)
        return predictions

    def link_predict_from_ht(self, head_emb, tail_embed, targets):
        relations_emb = self.pred_link_embeddings(targets)
        triples = torch.cat((head_emb, relations_emb, tail_embed), dim=1)
        predictions = self.link_predict_head1(triples)
        predictions = nn.functional.relu(predictions)
        predictions = self.link_predict_head2(predictions)
        return predictions

    def interleave_embeddings(self, ent_embeddings, rel_embeddings):
        # print(f"Rel embed {rel_embed.shape}\nEnt embed {ent_embed.shape}")
        # Interleave rels and ents and injecting positional information
        # path_embed = torch.stack((ent_embed, rel_embed), dim=2).view(2, 6, 5)
        path_embed = torch.zeros((ent_embeddings.size()[0],  # bsz
                                  ent_embeddings.size()[1] + rel_embeddings.size()[1],  # np
                                  ent_embeddings.size()[2]), device=self.device())

        ent_ind = torch.arange(0, ent_embeddings.size()[1] + rel_embeddings.size()[1],
                               2, device=self.device())
        rel_ind = torch.arange(1, ent_embeddings.size()[1] + rel_embeddings.size()[1],
                               2, device=self.device())
        path_embed[:, ent_ind, :], path_embed[:, rel_ind, :] = \
            ent_embeddings, rel_embeddings
        return path_embed

    def select_separated_head_and_tail_embeddings(self,memory, head_idxs,
                                                  tail_idxs, entity_origin):
        # Isolate the embeddings for head and tails only from the triples
        # head_embed is (num_paths, d_model) after squeezing
        # print(f"Memory {memory.shape}")
        # head and tail selection with variable num paths for each
        head_emb = memory[torch.arange(memory.shape[0]), head_idxs, :]
        tail_emb = memory[torch.arange(memory.shape[0]), tail_idxs, :]
        # when the number of tail and head embeddings is not the same
        # and the entity_origin is available
        head_idxs_mask = ~entity_origin.bool()
        # head_indices = head_idxs[head_idxs_mask]
        tail_idxs_mask = ~head_idxs_mask
        # tail_indices = tail_idxs[tail_idxs_mask]
        head_embed = head_emb[head_idxs_mask, :]
        tail_embed = tail_emb[tail_idxs_mask, :]
        return head_embed, tail_embed


    def select_head_and_tail_embeddings(self, memory, head_idxs,
                                                  tail_idxs):
        # Isolate the embeddings for head and tails only from the triples
        # head_embed is (num_paths, d_model) after squeezing
        # print(f"Memory {memory.shape}")
        # head and tail selection with variable num paths for each
        head_emb = memory[torch.arange(memory.shape[0]), head_idxs, :]
        tail_embed = memory[torch.arange(memory.shape[0]), tail_idxs, :]
        # when the number of tail and head embeddings is not the same
        head_indices = (head_idxs >= 0).nonzero().squeeze()
        head_embed = head_emb[head_indices, :]
        tail_indices = (tail_idxs >= 0).nonzero().squeeze()
        tail_embed = tail_embed[tail_indices, :]
        return head_embed, tail_embed

    def average_embeddings_with_origin(self, embeddings, origin, target):
        """
        A function that averages the embeddings in a table for each triple
        Parameters
        ----------
        embeddings : The emebeddings of shape (num_entities, dim)
        ppt : The paths per triple
        idxs : The idxs of the entities in each path

        Returns
        -------
        A single embedding table of shape (num triples, dim)
        """

        output, counts = torch.unique_consecutive(origin, return_counts=True)
        count_selection_mask = (output == target)
        paths_per_entity = counts[count_selection_mask]
        # split the embeddings based on the existing number of paths per entity
        entity_embeds = torch.split(embeddings, paths_per_entity.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(entity_embeds, batch_first=True)
        # average ignoring padding
        averaged_embs = torch.sum(padded_embs, dim=1) / paths_per_entity.unsqueeze(1)
        return averaged_embs

    def split_and_pad_embeddings_with_origin(self, embeddings, origin, target):
        """
        A function that averages the embeddings in a table for each triple
        Parameters
        ----------
        embeddings : The emebeddings of shape (num_entities, dim)
        ppt : The paths per triple
        idxs : The idxs of the entities in each path

        Returns
        -------
        A single embedding table of shape (num triples, dim)
        """

        output, counts = torch.unique_consecutive(origin, return_counts=True)
        count_selection_mask = (output == target)
        paths_per_entity = counts[count_selection_mask]
        # split the embeddings based on the existing number of paths per entity
        entity_embeds = torch.split(embeddings, paths_per_entity.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(entity_embeds, batch_first=True)
        return padded_embs

    def sum_embeddings_with_origin(self, embeddings, origin, target):
        """
        A function that averages the embeddings in a table for each triple
        Parameters
        ----------
        embeddings : The emebeddings of shape (num_entities, dim)
        ppt : The paths per triple
        idxs : The idxs of the entities in each path

        Returns
        -------
        A single embedding table of shape (num triples, dim)
        """

        output, counts = torch.unique_consecutive(origin, return_counts=True)
        count_selection_mask = (output == target)
        paths_per_entity = counts[count_selection_mask]
        # split the embeddings based on the existing number of paths per entity
        entity_embeds = torch.split(embeddings, paths_per_entity.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(entity_embeds, batch_first=True)
        # average ignoring padding
        # summed_embs = torch.sum(padded_embs, dim=1)
        summed_embs, _ = torch.max(padded_embs, dim=1)
        return summed_embs

    def query_head_embeddings(self, embeddings, origin):
        output, counts = torch.unique_consecutive(origin, return_counts=True)
        count_selection_mask = (output == 0)
        paths_per_entity = counts[count_selection_mask]
        entity_embeds = torch.split(embeddings, paths_per_entity.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(entity_embeds, batch_first=True)
        query_vector = self.head_query_vector(
            torch.tensor([0], device=self.device()))
        expanded_query = query_vector.unsqueeze(0).expand(
            padded_embs.size()[0],-1, -1)
        result, _ = self.entity_fusion_head(
            query=expanded_query, key=padded_embs,
            value=padded_embs, need_weights=False)
        return result.squeeze()

    def query_tail_embeddings(self, embeddings, origin):
        """
        A function that aggregates the tail embeddings by using a specialized
         tail query vector and multi head attention
        Parameters
        ----------
        embeddings : the embeddings to aggregate (num paths, embeddings
        dimension)
        origin : the vector denoting which entity head - 0  or tail - 1,
        each embedding belongs to (num paths, 1)

        Returns
        -------
        A single vector for each tail of each triple

        """
        output, counts = torch.unique_consecutive(origin, return_counts=True)
        count_selection_mask = (output == 1)
        paths_per_entity = counts[count_selection_mask]
        entity_embeds = torch.split(embeddings, paths_per_entity.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(entity_embeds, batch_first=True)
        query_vector = self.tail_query_vector(
            torch.tensor([0], device=self.device()))
        expanded_query = query_vector.unsqueeze(0).expand(
            padded_embs.size()[0],-1, -1)
        result, _ = self.entity_fusion_tail(
            query=expanded_query, key=padded_embs,
            value=padded_embs, need_weights=False)
        return result.squeeze()

    def embed_nodes(self, ent_paths: Tensor,
                    rel_paths: Tensor,
                    idxs: Tensor,
                    pos: Tensor,
                    ppe: Optional[Tensor] = None,
                    source_mask: Optional[Tensor] = None
                    ):
        assert self.batch_first == True, "Dev. code in batch-first mode"
        # FIXME remove after testing
        # injector = EmbeddingInjector(torch.arange(1,100,
        #                                                dtype=torch.int32,
        #                                           device=self.device()),
        #                                   64, 14505, self.device())
        rel_embed = self.rel_embeddings(rel_paths)
        ent_embed = self.rcontext_projector(ent_paths)
        if self.has_injector:
            ent_embed = self.injector(ent_embed, ent_paths)
        # test = injector.forward(ent_embed, ent_paths) # FIXME remove after
        #  testing
        # interleave embeddings to create path embeddings
        path_embed = self.interleave_embeddings(ent_embeddings=ent_embed,
                                                rel_embeddings=rel_embed)
        path_embed += self.pos_embeddings(pos)
        # Masking and feeding the entity-relation path to the encoder
        # print(f"Input {path_embed.shape}")
        source_pmask = None  # FIXME
        memory = self.encoder(path_embed,
                              mask=source_mask,
                              src_key_padding_mask=source_pmask)
        # Isolate the embeddings for head and tails only from the triples
        # head_embed is (num_paths, d_model) after squeezing
        # head and tail selection with variable num paths for each
        ent_emb = memory[torch.arange(memory.shape[0]), idxs, :]

        if self.ent_aggregation != "avg":

            entity_embeds = torch.split(ent_emb, ppe.tolist(),
                                        dim=0)
            # padded_embs = nn.utils.rnn.pad_sequence(entity_embeds,
            #                                         batch_first=True)

            return entity_embeds
        else:  # use the average pooling otherwise
            # split the embeddings based on the existing number of paths per entity
            entity_embeds = torch.split(ent_emb, ppe.tolist(),
                                        dim=0)
            # convert list tp tensor by padding
            padded_embs = nn.utils.rnn.pad_sequence(entity_embeds,
                                                    batch_first=True)
            # average ignoring padding
            averaged_embs = torch.sum(padded_embs,
                                      dim=1) / ppe.unsqueeze(1) #FIXME summation
            # averaged_embs = torch.sum(padded_embs, dim=1)
            # averaged_embs, _ = torch.max(padded_embs, dim=1)
            return averaged_embs

    def mha_aggregate_heads(self, head_emb):
        query_vector = self.head_query_vector(
            torch.tensor([0], device=self.device()))
        expanded_query = query_vector.unsqueeze(0).expand(
            head_emb.size()[0], -1, -1)
        result, _ = self.entity_fusion_head(
            query=expanded_query, key=head_emb,
            value=head_emb, need_weights=False)
        return result.squeeze()

    def mha_aggregate_tails(self,tail_emb):
        query_vector = self.tail_query_vector(
            torch.tensor([0], device=self.device()))
        expanded_query = query_vector.unsqueeze(0).expand(
            tail_emb.size()[0], -1, -1)
        result, _ = self.entity_fusion_head(
            query=expanded_query, key=tail_emb,
            value=tail_emb, need_weights=False)
        return result.squeeze()



    def predict_links(self, head_emb, tail_emb, targets):
        if self.ent_aggregation != "avg":
            head_emb, tail_emb = self.aggregator(
            head_emb, tail_emb)
        logits = self.link_predict_from_ht(head_emb, tail_emb, targets)
        return logits

    def predict_relations(self, head_emb, tail_emb):
        if self.ent_aggregation != "avg":
            head_emb, tail_emb = self.aggregator(
            head_emb, tail_emb)
        logits = self.predict_relation_from_ht(head_emb, tail_emb)
        logits = nn.functional.softmax(logits, dim=1)
        return logits


    def forward(self,
                ent_paths: Tensor,
                rel_paths: Tensor,
                head_idxs: Tensor,
                tail_idxs: Tensor,
                pos: Tensor,
                entity_origin: Tensor,
                targets: Tensor,
                ppt: Optional[Tensor] = None,
                source_mask: Optional[Tensor] = None):

        assert self.batch_first == True, "Dev. code in batch-first mode"

        rel_embed = self.rel_embeddings(rel_paths)
        ent_embed = self.rcontext_projector(ent_paths)

        # If the model contains an injector pass the embeddings through the
        # injector first
        if self.has_injector:
            ent_embed = self.injector(ent_embed, ent_paths)
        # interleave embeddings to create path embeddings
        path_embed = self.interleave_embeddings(ent_embeddings=ent_embed,
                                                rel_embeddings=rel_embed)
        path_embed += self.pos_embeddings(pos)
        # Masking and feeding the entity-relation path to the encoder
        # print(f"Input {path_embed.shape}")
        source_pmask = None  # FIXME
        memory = self.encoder(path_embed,
                              mask=source_mask,
                              src_key_padding_mask=source_pmask)
        # Isolate the embeddings for head and tails only from the triples
        head_embed, tail_embed = (
            self.select_separated_head_and_tail_embeddings(memory=memory,
                                                           head_idxs=head_idxs,
                                                           tail_idxs=tail_idxs,
                                                           entity_origin=entity_origin))
        if self.ent_aggregation != "avg":
            head_padded = self.split_and_pad_embeddings_with_origin(
                head_embed, origin=entity_origin, target=0)
            tail_padded = self.split_and_pad_embeddings_with_origin(
                tail_embed, origin=entity_origin, target=1)
            head_emb, tail_emb = self.aggregator(
                head_padded, tail_padded)
            # head_emb = self.query_head_embeddings(
            #     head_embed, origin=entity_origin)
            # tail_emb = self.query_tail_embeddings(
            #     tail_embed, origin=entity_origin)
        else:  # use the average pooling otherwise
            tail_emb = self.average_embeddings_with_origin(
                tail_embed, origin=entity_origin, target=1)
            head_emb = self.average_embeddings_with_origin(
                head_embed, origin=entity_origin, target=0)
            # tail_emb = self.sum_embeddings_with_origin(
            #     tail_embed, origin=entity_origin, target=1)
            # head_emb = self.sum_embeddings_with_origin(
            #     head_embed, origin=entity_origin, target=0)

        head_emb = head_emb.unsqueeze(0) if head_emb.ndim < 2 else head_emb
        tail_emb = tail_emb.unsqueeze(0) if tail_emb.ndim < 2 else tail_emb

        logits_rp = self.predict_relation_from_ht(head_emb, tail_emb)
        logits_link = self.link_predict_from_ht(head_emb, tail_emb, targets)

        return logits_rp, logits_link

# FIXME romove after testing
# embeddings = torch.rand((4,10))
# node_indices= torch.tensor([1,0,3,10])
# num_dim = 10
# num_nodes = 50
# model = EmbeddingInjector(embeddings=embeddings, node_indices=node_indices,
#                           num_dim=num_dim,
#                           num_nodes=num_nodes,
#                           projector=True,
#                           device='cpu')