"""
Denoising datasets for the reconstruction task of PathER, where logic is moved to
collate function in order to streamline data loading operations from arbitrary
files. Collation is used to create batch by adding the perturbations.

"""
import math
import logging
import itertools
from collections import Counter
from typing import Dict, List, Iterator

from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import RandomApply, RandomChoice, RandomOrder

from . import data_utils as du
from . import path_lib as plib
from .utils import sample_or_repeat
from .corruption import generate_negative_triples, CorruptLinkGenerator

logger = logging.getLogger(__name__)


class InconsistentVocabulary(Exception):
    pass  # raised when the input vocab is not consistent with the data


def make_rpred_tensor(entities_idx, entities_pos, rpred_dict,
                      no_relt=torch.tensor([-1])):
    """
    Create a tensor for a relation prediction task by creating all combinations
    of the entities provided, and indexing them based on their position.

    Note: mechanism for balancing out the number of no-rel predictions as a way
          to counter the class imbalance of the classification task.

    Parameters
    ----------
    entities_idx : torch.tensor
        The sequence of entities to combine for relation prediction.
    entities_pos : torch.tensor
        The position of the CLS tokens for each entity in ``entities_idx``.
    rpred_dict : dict
        A nested dictionary mapping heads, to tails, to their relations.
    no_rel : torch.tensor
        The torch tensor corresponding to the "no-relation" token.

    Returns
    -------
    rprediction_t : torch.tensor
        The relation prediction tensor, containing all possible relations among
        the given entities. Columns are organised as: [0] head_cls_idx,
        [1] tail_cls_idx, [2] relation_id (dataset encoding from a PathDataset),
        [3] head_id (original), [4] tail_id (original).
    """
    assert len(entities_idx) == len(entities_pos)

    entities_list = list(entities_idx.numpy())
    enpos = dict(zip(entities_list, entities_pos))

    choice = lambda x: x[torch.randperm(x.numel())[0]]
    rprediction_t = [[enpos[h], enpos[t], choice(rpred_dict[h].get(t, no_relt)),
                      h, t]  # to account for the original node ids
                     for h in entities_list for t in entities_list
                     if h in rpred_dict and h != t]

    rprediction_t = torch.tensor(rprediction_t, dtype=entities_idx.dtype)
    if no_relt == -1:  # remove negative triples if norelt token not given
        # Keep all rows whose relation col (-1, the last one) does not have -1,
        # which denotes the default no-relation token (no_relt)
        rprediction_t = rprediction_t[rprediction_t[:, 2] != -1]

    return rprediction_t


def sample_relational_context(in_relcontext, out_relcontext, num_samples,
                              merge=False):
    """
    Randomly sample from the incoming and the outgoing relational contexts,
    either separately (two tensors are returned) or from both of them.

    Parameters
    ----------
    in_relcontext : torch.tensor
        The incoming relational context, given as a tensor of relation ids.
    out_relcontext : torch.tensor
        The outgoing relatioonal context, given as a tensor of relation ids.
    num_samples : int
        The number of unique tokens to sample from the contexts.
    merge : bool
        Whether sampling is done on the merged tensor or not.
    
    Note: we may want to parameterise the sampling function below, to minimise
          the impact of 1-to-many relations.
    """

    def sample(sampled_tensor: torch.tensor, num_samples: int, unique=True):
        # sampled_tensor = torch.tensor(sampled_tensor)  # make sure its a tensor
        sampled_tensor = sampled_tensor[torch.randperm(len(sampled_tensor))]
        sampled_tensor = torch.unique(
            sampled_tensor) if unique else sampled_tensor
        # Now we are safe to return min(``num_samples``, len(``sampled_tensor``)
        return sampled_tensor[:num_samples]

    if merge:  # return a single sample from both the inc and outg contexts
        return sample(torch.cat((in_relcontext, out_relcontext)), num_samples)
    in_relcontext = sample(in_relcontext, num_samples)
    out_relcontext = sample(out_relcontext, num_samples)
    return in_relcontext, out_relcontext


def sample_relational_context_with_oversampling(in_relcontext, out_relcontext,
                                                num_samples,
                                                merge=False):
    """
    Randomly sample from the uniques of the incoming and the outgoing
    relational contexts, either separately (two tensors are returned)
     or from both of them and oversample relation based on their frequency

    Parameters
    ----------
    in_relcontext : torch.tensor
        The incoming relational context, given as a tensor of relation ids.
    out_relcontext : torch.tensor
        The outgoing relatioonal context, given as a tensor of relation ids.
    num_samples : int
        The number of unique tokens to sample from the contexts.
    merge : bool
        Whether sampling is done on the merged tensor or not.

    Note: we may want to parameterise the sampling function below, to minimise
          the impact of 1-to-many relations.
    """

    def sample(sampled_tensor: torch.tensor, num_samples: int, unique=True):
        # get unique relations and their counts
        sampled_tensor, counts = torch.unique(sampled_tensor,
                                              return_counts=True)
        # get sorted indices of the counts (same for relations), to sample
        # preferentially the higher frequency relations
        _, sorted_counts = torch.sort(counts, descending=True)
        # sort the tensors and sample
        sampled_tensor = sampled_tensor[sorted_counts][:num_samples]
        counts = (counts.float() / torch.sum(counts))
        counts = counts[sorted_counts][:num_samples]

        # calculate the number of times each relations should be repeated
        # oversampling = ((counts.float() / torch.sum(counts)) * (
        #         2*num_samples)).round().to(torch.int32)
        oversampling = torch.ceil(counts * (2 * num_samples)).to(torch.int32)
        # repeat each relation
        sampled_tensor = torch.repeat_interleave(sampled_tensor, oversampling)
        # Now we are safe to return min(``num_samples``, len(``sampled_tensor``)
        return sampled_tensor

    if merge:  # return a single sample from both the inc and outg contexts
        return sample(torch.cat((in_relcontext, out_relcontext)), num_samples)
    in_relcontext = sample(in_relcontext, num_samples)
    out_relcontext = sample(out_relcontext, num_samples)
    return in_relcontext, out_relcontext


def create_vocabulary(paths: List[List[int]], xtokens: List[str]):
    """
    Create a vocabulary based on the relations retrieved from the path store.
    This will provide an encoding for both the relations and the special tokens,
    which will be prepended in the mapping (they offset the rel. mapping). Note
    that the PADding token is always included to extend the ``xtokens``.

    Parameters
    ----------
    paths : List[List[int]]
        A list containing relational paths, encoded as a list of integers.
    xtokens : List[str]
        The special tokens to add to the vocabulary, which will offset idxs.

    Note: at the moment, it also provides statistics to potentially filter out
        less frequent relations, but this is not yet reflected in the encoding.
    """
    # Counting relation occurrences across all paths and relative freqs
    relation_cnt = Counter()
    for path in tqdm(paths):
        relation_cnt.update(path)
    relation_tot = relation_cnt.total()
    relation_pct = {k: v / relation_tot for k, v in relation_cnt.items()}
    # self.start_nodes_vocab = sorted(np.unique(sample_df['start_node']))
    # self.end_nodes_vocab = sorted(np.unique(sample_df['end_node']))
    relation_vocab = sorted(relation_cnt)
    min_relation_idx, max_relation_idx = \
        min(relation_vocab), max(relation_vocab)
    expected_relation_vocab = list(
        range(min_relation_idx, max_relation_idx + 1))
    if relation_vocab != expected_relation_vocab:
        missing = set(expected_relation_vocab) - set(relation_vocab)
        logger.warn("Relation set may not be complete, sample more paths before"
                    f" training! Removing unseen relations for now: {missing}")
    print(f"Relation size: {len(relation_vocab)} min {min_relation_idx}")
    # Saving token encoder and decoder: from tokens to indices and v.v.
    tokens_to_idxs = {t: idx for idx, t in enumerate(
        ["PAD"] + xtokens + relation_vocab)}

    return relation_vocab, tokens_to_idxs


def check_vocabulary(paths: List[List[int]], vocabulary: List):
    """
    Check whether a vocabulary subsumes the one implied by a list of paths.
    
    Parameters
    ----------
    paths : List[List[int]]
        A list containing relational paths, encoded as a list of integers.
    vocabulary : List[int]
        A vocabulary of tokens (including special tokens) to sanity check.

    """
    for path in tqdm(paths, desc="Vocabulary check"):
        if not all([r in vocabulary for r in path]):
            return False
    # Vocabulary is good at this stage
    return True


class PathDataset(Dataset):

    def __init__(self, path_store: str, seed: int = 46, transform=None,
                 xtokens: List[str] = ["MSK", "CLS", "SEP", "UKN"],
                 tokens_to_idxs: Dict[int, int] = None):
        """
        Generalisation to describe datasets of network paths given in a CSV file
        and encoded as sequence of relations, given by their index. Although the
        encoding is re-initialised to start from 0, relation indexes are assumed
        to be contiguous: spanning all integers from a certain range. If this
        does not happen, the code assumes that more paths should be sampled.

        Extra indices can be used on request to denote special tokens such as:
        - [PAD] used for creating batches with same-length sequences
        - [MSK] used for masked sequence modelling (input sequences only)
        - [CLS] used for additional learning strategies at the subsequence level
        - [SEP] used to denote sub-sequence separators (e.g. full stop in text)
        - [UKN] for encoding rare, uncommon, or unformatted observations

        Parameters
        ----------
        path_store : str
            Path to the CSV file storing Knowledge Graph paths, where the first
            two columns denote the indexes of the starting and the ending node
            of each path, whereas the third column holds the sequence of crossed
            relation indexes in the path, separated by a space.
        transform : fn, optional
            A transformation to apply to paths before they are returned.
        xtokens : List[str]
            A list of extra/special tokens whose index will preceed the mapping.

        """
        sample_df = pd.read_csv(path_store)
        print(f"Found {len(sample_df)} paths from {path_store}")
        # Split the path string on spaces and convert to list of int (edge ids)
        # and compute path len (which can help sorting sequences prior to batch)
        sample_df['relation_path'] = sample_df['relation_path'].apply(
            lambda x: [int(i) for i in x.split()])
        sample_df['path_len'] = sample_df['relation_path'].apply(
            lambda x: len(x))

        if tokens_to_idxs is not None:
            logger.info(f"Using given encoding: {len(tokens_to_idxs)} tokens")
            self.tokens_to_idxs = tokens_to_idxs  # TODO Sanity check
            assert all([xt in self.tokens_to_idxs for xt in xtokens])
            self.no_relations = len(tokens_to_idxs) - len(xtokens) - 1
        else:  # otherwise, recompute all the encoding and vocabulary
            logger.info(f"Creating new vocabulary from {path_store}")
            relation_vocab, self.tokens_to_idxs = create_vocabulary(
                list(sample_df['relation_path']), xtokens=xtokens)
        # Decoding map: from idxs back to the original sequence tokens
        self.idxs_to_tokens = {idx: t for t, idx in self.tokens_to_idxs.items()}

        self.epoch = None
        self.transform = transform
        self.paths = sample_df.iloc[:, 2].values
        self.vocab_size = len(self.tokens_to_idxs)
        self.seed = seed if seed is not None else torch.seed()

    def __len__(self):
        return len(self.paths)

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def __getitem__(self, idx):
        tokens = self.paths[idx]  # path sequence - List of int edge ids
        tokens = torch.LongTensor([self.tokens_to_idxs[t] for t in tokens])
        source, target = tokens, tokens.clone()

        if self.transform is not None:
            with du.numpy_seed(self.seed, self.epoch, idx):
                source = self.transform(source)  # w.r.t. perturb strategy

        return {"id": idx, "source": source, "target": target}


class UnrolledPathDataset(Dataset):

    def __init__(self, path_store: str,
                 relneighb_store: str,
                 relcontext_store: str,
                 relcontext_size: int,
                 tokens_to_idxs: Dict[int, int] = None,
                 transform_relcontext=None,
                 transform_relations=None,
                 transform_entities=None,
                 seed: int = 46, ):
        """
        A PathDatest where paths hold the relational context of nodes traversed.

        This dataset use a predefined set of special tokens:
        - [PAD] used for creating batches with same-length sequences
        - [MSK] used for masked sequence modelling (input sequences only)
        - [CLS] used for entities embeddings w.r.t the relational context
        - [SEP] to separate and denote bridging relations from context relations

        Parameters
        ----------
        path_store : str
            Path to the CSV file storing paths, where the first column contains
            the relations in the path, and the second holds the traversed nodes.
        relneighb_store : str
            Path to the CSV file holding the relations that occur between every
            pair of entity in the knowledhe graph (a summary of local info).
        relcontext_store : str
            Path to the CSV file storing the relational context for each entity,
            where `node_id` denotes the entity, `direction` holds the type of
            context (either `in` or `out` for incoming and outgoing relations),
            and `edges` contains the whole list of relations (with repetitions).
        transform : fn, optional
            A transformation to apply to paths before they are returned.
        xtokens : List[str]
            A list of extra/special tokens whose index will preceed the mapping.

        """
        xtokens = ["MSK", "CLS", "SEP"]  # NR
        pathstore_df = pd.read_csv(path_store)
        relcontext_df = pd.read_csv(relcontext_store)
        relneighb_df = pd.read_csv(relneighb_store)
        # Step 1: Read paths and convert them to plain lists
        print(f"Found {len(pathstore_df)} paths from {path_store}")
        du.listify_columns(pathstore_df)  # space-separated rels --> lists
        du.listify_columns(relcontext_df, "edges")  # same for rel context
        du.listify_columns(relneighb_df, "Relations")  # and same for neighbours

        # Step 2: Load or build the vocabulary using the relational context
        if tokens_to_idxs is not None:
            logger.info(f"Using given encoding: {len(tokens_to_idxs)} tokens")
            if not check_vocabulary(relcontext_df["edges"],
                                    tokens_to_idxs.keys()):
                raise InconsistentVocabulary("Missing tokens in vocabulary")
            self.tokens_to_idxs = tokens_to_idxs
        else:  # otherwise, recompute all the encoding and vocabulary
            logger.info(f"Creating new vocabulary from {path_store}")
            relation_vocab, self.tokens_to_idxs = create_vocabulary(
                list(relcontext_df["edges"]), xtokens=xtokens)
        # Decoding map: from idxs back to the original sequence tokens
        self.idxs_to_tokens = {idx: t for t, idx in self.tokens_to_idxs.items()}

        # Step 3: Encoding and building tensors for relations and entities
        encode = lambda x: torch.LongTensor([self.tokens_to_idxs[r] for r in x])
        self.relation_paths = [encode(p) for p in pathstore_df["relation_path"]]
        self.entity_paths = [torch.LongTensor(p)  # not encoded!
                             for p in pathstore_df["entity_path"]]

        # Step 4:  Convert relational context to a tensor dict
        relcontexts = {}  # {node_id: {direction: torch.tensor(relations)}}
        for i, record in relcontext_df.iterrows():
            if record["node_id"] not in relcontexts:
                relcontexts[record["node_id"]] = {}
            relcontexts[record["node_id"]][record["direction"]] = \
                encode(record["edges"])  # needed for sampling

        # Step 5: Convert relational neighbour to a tensor dict
        relneighbours = {}  # {node_id: {node_id: torch.tensor(relations)}}
        for i, record in relneighb_df.iterrows():
            if record["Head"] not in relneighbours:
                relneighbours[record["Head"]] = {}
            relneighbours[record["Head"]][record["Tail"]] = \
                encode(record["Relations"])  # needed for sampling

        self.epoch = None
        self.relcontexts = relcontexts
        self.relneighbours = relneighbours
        self.relcontext_size = relcontext_size
        self.vocab_size = len(self.tokens_to_idxs)
        self.no_relations = self.vocab_size - len(xtokens) - 1  # PAD
        self.transform_rc = transform_relcontext
        self.transform_re = transform_relations
        self.transform_en = transform_entities
        self.seed = seed if seed is not None else torch.seed()

    def __len__(self):
        return len(self.relation_paths)

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def create_unrolled_path(self, relations, entities, relcontext_size,
                             local_seed=1, perturb=True):
        """
        Create and assemble an unrolled path given relations and entities.
        """
        # FIXME Understand if we can just encode these for all records
        clst = torch.tensor([self.tokens_to_idxs["CLS"]], dtype=entities.dtype)
        sept = torch.tensor(self.tokens_to_idxs["SEP"], dtype=clst.dtype)
        mskt = torch.tensor(self.tokens_to_idxs["MSK"], dtype=clst.dtype)
        empt = torch.tensor([], dtype=clst.dtype)  # for nodes with no rel
        mcon = torch.tensor([mskt, clst, mskt],
                            dtype=clst.dtype)  # mskd context

        relational_contexts = {}  # node_id → relational context
        for node in entities:  # sampling relational contexts for nodes
            node_relcontext = self.relcontexts[node.item()]  # full RC
            # logger.debug(f"Node {node} full RC: {node_relcontext}")
            in_relcontext, out_relcontext = sample_relational_context(
                node_relcontext.get("in", empt),
                node_relcontext.get("out", empt),
                relcontext_size,
            )
            # TODO Check that we have at least a context available
            if perturb and self.transform_rc is not None:  # local perturbations
                with du.numpy_seed(self.seed, self.epoch, local_seed):
                    in_relcontext = self.transform_rc(in_relcontext)
                    out_relcontext = self.transform_rc(out_relcontext)
            # XXX This would fail if the path contains loops (entity rep)
            relational_contexts[node.item()] = torch.concat(
                (in_relcontext, clst, out_relcontext))

        source_relations = relations.clone()
        if perturb and self.transform_re is not None:  # relation perturbations
            with du.numpy_seed(self.seed, self.epoch, local_seed):
                source_relations = self.transform_re(source_relations)

        source_entities = entities.clone()
        if perturb and self.transform_en is not None:  # entity perturbations
            with du.numpy_seed(self.seed, self.epoch, local_seed):
                source_entities = self.transform_en(source_entities)

        # So far, no support for deletion/insertion
        assert source_entities.numel() == entities.numel()
        assert source_relations.numel() == relations.numel()
        unrolled_path = []  # assembling the unrolled sequence
        unrolled_pos = []  # holding the true positions of tokens
        # Assembling the unrolled path via concatenation of RCs and bridges
        for i, (node, rel) in enumerate(zip(source_entities, source_relations)):
            rc_con = relational_contexts[node.item()] if node != mskt else mcon
            unrolled_path.append(rc_con)  # relational or masked context
            unrolled_path.append(torch.LongTensor([sept, rel, sept]))
            unrolled_pos += [i * 2] * len(rc_con) + [i * 2 + 1] * 3
        # Appending information of the last relational context in the sequence
        unrolled_path.append(relational_contexts[source_entities[-1].item()])
        unrolled_path = torch.concat(unrolled_path)
        unrolled_pos += [unrolled_pos[-1] + 1] * \
                        (len(unrolled_path) - len(unrolled_pos))
        # Creating torch tensors for the positionals and start from 1
        unrolled_pos = torch.tensor(unrolled_pos, dtype=unrolled_path.dtype) + 1

        return unrolled_path, unrolled_pos, source_entities

    def __getitem__(self, idx):
        # Retrieve the relational and entity paths from record index
        relations, entities = self.relation_paths[idx], self.entity_paths[idx]
        unrolled_path, unrolled_pos, source_entities = \
            self.create_unrolled_path(relations, entities,
                                      self.relcontext_size, local_seed=idx)
        # Final round: preparing the relation prediction tensor for CLS task
        clst = torch.tensor([self.tokens_to_idxs["CLS"]], dtype=entities.dtype)
        # FIXME This should also remove the MSKed relational context otw fails
        cls_pos = (unrolled_path == clst).argwhere().flatten()
        relpred = make_rpred_tensor(source_entities, cls_pos,
                                    self.relneighbours,
                                    torch.tensor([-2]))  # FIXME!!!

        return {"id": idx, "pos": unrolled_pos, "relpred": relpred,
                "source": unrolled_path, "target": relations}

    # Decoder expects the correct, original sequence of relations
    # Encoder wants the relation between each node for CLS objective
    # Encoder wants the full sequence with separators, etc.
    # Permutations need to remember which relation connects two nodes


class TriplePathDataset(Dataset):
    """
    A dataset for representing triples as paths: where heads and tails are
    replaced by an incoming and an outgoing path, respectively. The current
    implementation reuses a ``PathDataset`` to reuse path construction logic.

    Parameters
    ----------
    triplestore : torch.tensor
        A matrix holding (h, r, t) triples, following the original raw encoding.
    pathdataset : UnrolledPathDataset
        The unrolled path dataset from which paths will be drawn for concat.
    relcontext_size : int
        The size of the relational contexts to sample for entities.
    trim_context_len : int
        If provided, context (entity-relation) paths are trimmed to the given
        length; for example, if set to 5, it trims 3 entities and 2 relations.
    """

    def __init__(self, triplestore, pathdataset, relcontext_size,
                 trim_context_len=101, num_negatives=0, triple_corruptor=None):

        if trim_context_len % 2 == 0:  # expecting N entities + (N-1) relations
            raise ValueError("Context path length can only be odd")

        path_index = {"in": {}, "out": {}}
        for i in tqdm(range(len(pathdataset)), desc="H-T path indexing"):
            # Retrieving start and end node from entity paths
            start, end = pathdataset.entity_paths[i][[0, -1]]
            start, end = start.item(), end.item()
            # Populating path index for incoming paths
            if end not in path_index["in"]:
                path_index["in"][end] = [i]
            else:  # node already in dict
                path_index["in"][end].append(i)
            # Populating path index for outgoing paths
            if start not in path_index["out"]:
                path_index["out"][start] = [i]
            else:  # node already in dict
                path_index["out"][start].append(i)

        self.num_positive, self.num_negatives = len(triplestore), num_negatives
        if num_negatives > 0:  # This triggers the creation of negative samples
            logger.info(f"Creating {num_negatives} neg per {self.num_positive}")
            corruptions = triple_corruptor.get_filtered_corrupted_triples(
                triplestore, num_negatives)  # generate full corruption tensor
            positive_triples = corruptions[:, 0, :]
            negative_triples = corruptions[:, 1:, :]
            negative_triples = negative_triples.reshape(-1, 3)
            assert len(negative_triples) == self.num_positive * num_negatives \
                   or len(
                negative_triples) == self.num_positive * num_negatives * 2
            assert torch.all(positive_triples.unique(dim=0) == triplestore)
            triplestore = torch.concat([positive_triples, negative_triples])

        self.triplstore = triplestore
        self.path_index = path_index
        self.pathdataset = pathdataset
        self.relcontext_size = relcontext_size
        self.rel_cut = trim_context_len // 2
        self.ent_cut = self.rel_cut + 1
        self.dtype = pathdataset.entity_paths[0].dtype

    def __len__(self):
        return len(self.triplstore)

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def __getitem__(self, idx):

        head, relation, tail = self.triplstore[idx]
        # Encoding the relation using the pathdataset mapping
        mskt = torch.tensor(self.pathdataset.tokens_to_idxs["MSK"],
                            dtype=self.dtype)
        relation_enc = self.pathdataset.tokens_to_idxs[relation.item()]
        relation_enc = torch.tensor(relation_enc, dtype=mskt.dtype)
        # print(f"Triple: {head}, {relation}({relation_enc}), {tail}")

        # First step is to assemble the template of the path
        choice = lambda x: np.random.choice(x, 1)[0]
        entities, relations = [], []
        # Retrieving an incoming path, if present, or use head
        if head.item() in self.path_index["in"]:
            # print(self.path_index["in"][head.item()])
            in_path_idx = choice(self.path_index["in"][head.item()])
            entities += self.pathdataset.entity_paths[in_path_idx][
                        -self.ent_cut:]
            relations += self.pathdataset.relation_paths[in_path_idx][
                         -self.rel_cut:]
        else:  # just include the node
            entities.append(head)

        relations.append(relation_enc)
        relation_loc, head_loc = len(relations) - 1, len(entities) - 1
        # Retrieving an outgoing path, if present, or use tail
        if tail.item() in self.path_index["out"]:
            out_path_idx = choice(self.path_index["out"][tail.item()])
            entities += self.pathdataset.entity_paths[out_path_idx][
                        :self.ent_cut]
            relations += self.pathdataset.relation_paths[out_path_idx][
                         :self.rel_cut]
        else:  # just include the node
            entities.append(tail)

        entities = torch.tensor(entities, dtype=mskt.dtype)
        relations = torch.tensor(relations, dtype=mskt.dtype)
        relations_tmask = relations.clone()
        relations_tmask[relation_loc] = mskt

        unrolled_path, unrolled_pos, _ = self.pathdataset.create_unrolled_path(
            relations_tmask, entities, self.relcontext_size, perturb=False)

        clst = torch.tensor(self.pathdataset.tokens_to_idxs["CLS"],
                            dtype=mskt.dtype)
        cls_pos = (unrolled_path == clst).argwhere().flatten()

        relpred = torch.tensor([[cls_pos[head_loc], cls_pos[head_loc + 1],
                                 relation_enc, head, tail]], dtype=mskt.dtype)

        return {"id": idx, "pos": unrolled_pos, "relpred": relpred,
                "source": unrolled_path, "target": relations, "ents": entities}


class NegativeTripleSampler(Sampler[int]):
    """
    Samples positive and negative triples from a TriplePathDataset.

    Parameters
    ----------
    num_positives : int
        Total number of positive triples in the dataset.
    k : int
        Number of negative triples (corruptions) per positive.
    """

    def __init__(self, num_positives, k) -> None:
        self.num_positives = num_positives
        self.k = k  # negatives per positive

    def __iter__(self) -> Iterator[int]:
        # Create interleaved triple indexes: [p_i, n_i1, ... n_ik]
        indexes = [[i] + list(range(self.num_positives + self.k * i,
                                    self.num_positives + self.k * i + self.k))
                   for i in range(self.num_positives)]

        yield from itertools.chain.from_iterable(indexes)

    def __len__(self) -> int:
        return self.num_positives * (self.k + 1)


################################################################################
# Multi-path datasets
# ------------------------------------------------------------------------------

class SimplePathDataset(Dataset):

    def __init__(self,
                 path_store: str,
                 relcontext_store: str,
                 triple_store: torch.tensor,
                 context_triple_store: torch.tensor = None,
                 tokens_to_idxs: Dict[int, int] = None,
                 maximum_triple_paths = 50,
                 num_negatives: int = 0,
                 triple_corruptor = None,
                 seed: int = 46,
                 parallel = False,
                 neg_triple_store = None,
                 ):
        """
        A base class for path-based datasets, wrapping the basic functionalities
        including pre-processing, vocabulary generation, path indexing, encoding
        and negative generation.

        Parameters
        ----------
        path_store : str
            Path to the CSV file storing paths, where the first column contains
            the relations in the path, and the second holds the traversed nodes.
        relcontext_store : str
            Path to the CSV file storing the relational context for each entity,
            where `node_id` denotes the entity, `direction` holds the type of
            context (either `in` or `out` for incoming and outgoing relations),
            and `edges` contains the whole list of relations (with repetitions).
        triplestore : torch.tensor
            A matrix holding (h, r, t) triples, following the original encoding.
        context_triple_store : torch.tensor
            If provided, these triples will be used for constructing paths,
            rather than using (as default option) those in ``triplestore``. 
        tokens_to_idxs : Dict[int, int]
            Mapping dictionary for encoding tokens to proper idxs.
        maximum_triple_paths : int
            The maximum number of paths per triple that will be returned.
        seed : int
            A seed to control the sthocasticity of the dataset.

        """
        xtokens = ["MSK"]  # the special tokens that will be reserved
        relcontext_df = pd.read_csv(relcontext_store)
        du.listify_columns(relcontext_df, "edges") # space-sep rels --> lists

        if isinstance(path_store, str):
            # Step 1: Read paths and convert them to plain lists
            pathstore_df = pd.read_csv(path_store)
            print(f"Found {len(pathstore_df)} paths from {path_store}")
            du.listify_columns(pathstore_df)  # space-separated vals --> lists
            # Step 2: Encoding and building tensors for relations and entities
            to_tensor = lambda x: [torch.IntTensor(t) for t in x]
            self.relation_paths = to_tensor(pathstore_df["relation_path"])
            self.entity_paths =  to_tensor(pathstore_df["entity_path"])
            self.path_index = plib.create_path_indexing(self.entity_paths)
        else:  # this assumes that the pathstore is a tuple
            self.relation_paths, self.entity_paths, self.path_index = path_store

        # Step 2: Load or build the vocabulary using the relational context
        if tokens_to_idxs is not None:
            logger.info(f"Using given encoding: {len(tokens_to_idxs)} tokens")
            if not check_vocabulary(relcontext_df["edges"], tokens_to_idxs.keys()):
                raise InconsistentVocabulary("Missing tokens in vocabulary")
            self.tokens_to_idxs = tokens_to_idxs
        else:  # otherwise, recompute all the encoding and vocabulary
            logger.info(f"Creating new vocabulary from {path_store}")
            relation_vocab, self.tokens_to_idxs = create_vocabulary(
                list(relcontext_df["edges"]), xtokens=xtokens)
        # Decoding map: from idxs back to the original sequence tokens
        self.idxs_to_tokens = {idx: t for t, idx in self.tokens_to_idxs.items()}

        self.num_pos, self.num_neg = len(triple_store), num_negatives
        if num_negatives > 0:  # triggers the extension of the triple store
            if neg_triple_store is None:  # create new negatives
                triple_store = generate_negative_triples(
                    triple_store, num_negatives,
                    triple_corruptor=triple_corruptor, parallel=parallel)
            else:  # attempting to reuse precomputed negatives
                triple_store = neg_triple_store  # FIXME
                expected_dim = (self.num_pos * 2) * (num_negatives + 1)
                assert neg_triple_store.shape[0] == expected_dim, \
                    "Corrupted triplestore dim does not match the triplestore:"\
                    f" got {neg_triple_store.shape[0]}, expected {expected_dim}"

        self.epoch = None
        self.xtokens = xtokens
        self.triplestore = triple_store
        self.max_paths = maximum_triple_paths
        self.vocab_size = len(self.tokens_to_idxs)
        self.context_triple_store = context_triple_store \
              if context_triple_store is not None else triple_store
        self.no_relations = self.vocab_size - len(xtokens) - 1  # 1 for PAD
        self.seed = seed if seed is not None else torch.seed()

    def __len__(self):
        return len(self.triplestore)

    def set_epoch(self, epoch, **unused):
        # This is kept for compatibility with other path datasets
        self.epoch = epoch

    def encode_relations(self, relations):
        # Follows the encoding computed at construction time or reused
        if torch.is_tensor(relations):
            relations =  relations.tolist()
        return torch.IntTensor([self.tokens_to_idxs[r] for r in relations]) \

    def encode_entities(self, entities):
        # Recreates the encoding of relations for entities, although we are not
        # really encoding nodes; but rather prepare a consistent mapping
        if torch.is_tensor(entities):
            entities =  entities.tolist()
        return torch.IntTensor([r + len(self.xtokens) + 1 for r in entities])
    
    def fetch_path_context(self, index: int, context_type: str):
        """
        Returns the entity and relation path associated to a path template.

        Parameters
        ----------
        index : int
            Index of the path that will be fetched from the pathstore.
        context_type : str
            Type of context, either ``triple`` or ``path``.

        """
        if context_type == "path":
            entities = self.entity_paths[index]
            relations = self.relation_paths[index]
        elif context_type == "triple":
            triple = self.context_triple_store[index]
            entities = triple[[0, -1]]
            relations = triple[[1]]
        else:  # context type is not path nor triple, hence not supported
            raise ValueError(f"{context_type} not a supported context type")

        return entities, relations

    def __getitem__(self, index) -> dict:
        raise NotImplementedError("Abstract class needs extension")

    def dump_negatives(self, fname: str):
        """Dumps the negative triples to disk, depending on corruption type."""
        negative_start = self.num_pos * 2 \
            if self.triplestore[0] == self.triplestore[1] else self.num_pos
        negatives = self.triplestore[negative_start:]
        torch.save(negatives, fname)
    
    def _load_negatives(self, negatives: torch.tensor, corruptor):
        if isinstance(corruptor, CorruptLinkGenerator):
            self.triplestore = self.triplestore.repeat_interleave(2)
        self.triple_store = torch.vstack([self.triple_store, negatives])


class MultiPathDataset(SimplePathDataset):

    def __init__(self,
                 path_store: str,
                 relcontext_store: str,
                 triple_store: torch.tensor,
                 context_triple_store: torch.tensor = None,
                 tokens_to_idxs: Dict[int, int] = None,
                 maximum_triple_paths = 50,
                 num_negatives: int = 0,
                 triple_corruptor = None,
                 seed: int = 46,
                 parallel=False,
                 neg_triple_store = None,
                ):
        """
        Adds methods and attributes to deal with multiple paths.
        """
        super().__init__(path_store,relcontext_store, triple_store,
                         context_triple_store, tokens_to_idxs,
                         maximum_triple_paths, num_negatives,
                         triple_corruptor, seed, parallel, neg_triple_store)
        # Distributing path budget across entities (H, T) and contexts (in, out)
        self.ppe = maximum_triple_paths // 2  # paths per entity
        self.ppc = self.ppe // 2  # paths per entity context (in/out)

    def _create_inout_contextpaths(self, entity):
        """
        Creates contextualised paths by combining incoming and outgoing paths
        for a given entity, and updating positional informatioon.
        """
        et = lambda: torch.IntTensor([])
        in_paths, out_paths = plib.create_contextpaths(
            entity, self.path_index, self.context_triple_store)
        # Get maximum posible num paths and sample from each context
        max_ppe = min(self.ppe, max(len(in_paths), len(out_paths)))
        path_sample_a = sample_or_repeat(in_paths, max_ppe)
        path_sample_b = sample_or_repeat(out_paths, max_ppe)
        # assert len(path_sample_b) == len(path_sample_a)
        ent_paths, rel_paths, ht_idxs, er_pos = [], [], [], []
        for in_context, out_context in zip(path_sample_a, path_sample_b):

            if in_context is not None:
                in_ents, in_rels = self.fetch_path_context(*in_context)
                entity = in_ents[-1]
                in_ents = in_ents[:-1]
            else:  # no incoming context is available
                in_ents, in_rels = et(), et()

            if out_context is not None:
                out_ents, out_rels = self.fetch_path_context(*out_context)
                entity = out_ents[0]
                out_ents = out_ents[1:]
            else:  # no outgoing context is available
                out_ents, out_rels = et(), et()

            entix = len(in_ents)  # position of the entity in the path
            entities = torch.concat([in_ents, entity.unsqueeze(0), out_ents])
            relations = torch.concat([in_rels, out_rels])
            assert len(entities) == len(relations) + 1

            ht_idxs.append(entix)
            ent_paths.append(self.encode_entities(entities))
            rel_paths.append(self.encode_relations(relations))
            er_pos.append(plib.get_entfocused_positionals(
                entix*2, len(entities)*2-1) + 1)  # +1 to avoid PAD

        return ent_paths, rel_paths, ht_idxs, er_pos


class ProtopathDataset(SimplePathDataset):

    def __init__(self,
                 path_store: str,
                 relcontext_store: str,
                 triple_store: torch.tensor,
                 context_triple_store: torch.tensor = None,
                 tokens_to_idxs: Dict[int, int] = None,
                 maximum_triple_paths = 50,
                 num_negatives: int = 0,
                 triple_corruptor = None,
                 seed: int = 46,
                ):
        """
        One triple, several paths traversing the same triple for context. These
        are constructed by considering the various combinations of incoming and
        outgoing paths (or triples, if paths are not available) with respect to
        the head and tail, respectively. Rather than explicitly storing paths,
        only the actual indexes are kept in memory for efficiency.
        """
        super().__init__(path_store,relcontext_store, triple_store,
                         context_triple_store, tokens_to_idxs,
                         maximum_triple_paths, num_negatives,
                         triple_corruptor, seed)
        # Optionally creates protopaths if caching is enabled
        self.protopath_caching = False # FIXME to parameterise
        if self.protopath_caching:
            self._create_triple_protopaths()

    def _create_triple_protopaths(self):
        # Create protopaths only for the unique combinations of heads-tails
        self.protopath_map = {}  # indexed by head-tail
        for i in tqdm(range(self.triplestore.shape[0]), desc="Protopaths gen"):
            # Retrieving the triple and creating
            head, rel, tail = self.triplestore[i]
            protopath_hash = f"{head}_{tail}"
            if protopath_hash not in self.protopath_map:
                self.protopath_map[protopath_hash] = plib.create_protopaths(
                    head, tail, self.path_index, self.context_triple_store)

    def instantiate_protopath(self, triple, propath):
        """
        Generate an actual entity-relation path given a protopath, indexing the
        incoming and the outgoing context, if available; or use triple only.

        Parameters
        ----------
        triple : torch.tensor
            A triple to contextualise/extend to form a path.
        propath : tuple
            A tuple specifying the idx and type of incoming and outgoing context

        """
        to_tensor = lambda x: torch.tensor(x, dtype=triple.dtype)
        # Fetch the incoming (head) and the outgoing (tail) context, if avail
        entities_in, relations_in = self.fetch_path_context(
            propath[0], propath[1]) if propath[0] is not None \
                else (to_tensor([triple[0]]), to_tensor([]))
        entities_out, relations_out = self.fetch_path_context(
            propath[2], propath[3]) if propath[2] is not None \
                else (to_tensor([triple[-1]]), to_tensor([]))

        # print(entities_in, relations_in)
        # print(entities_out, relations_out)
        assert entities_in[-1] == triple[0] and entities_out[0] == triple[-1]
        head_idx = len(entities_in) - 1  # tail_idx = head_idx + 1
        # Merging incoming with outgoing contexts with the bridging triple
        ents = torch.concat([entities_in[:-1], triple[[0,-1]], entities_out[1:]])
        rels = torch.concat([relations_in, triple[[1]], relations_out])

        return ents, rels, head_idx

    def __getitem__(self, index) -> dict:
        """
        Get a multi-paths contextualisation of the i-th triple in the dataset.
        This implementation either reuses protopath caching, or creates otf. 
        """
        triple = self.triplestore[index]
        head, rel, tail = triple
        protopaths = self.protopath_map[f"{head}_{tail}"] \
            if self.protopath_caching else plib.create_protopaths(
                head, tail, self.path_index, self.context_triple_store)

        with du.local_seed(self.seed, self.epoch, index):
            # Sampling max N before translating protopaths
            protopaths = [protopaths[i] for i in
                          torch.randperm(len(protopaths))[:self.max_paths]]

        ent_paths, rel_paths, head_idxs = [], [], []
        for protopath in protopaths:
            ents, rels, head_idx = self.instantiate_protopath(triple, protopath)
            # Encode both relations and entities, and mask the bridge relation
            rels = self.encode_relations(rels)
            rels[head_idx] = self.tokens_to_idxs["MSK"]
            rel_paths.append(rels)
            ent_paths.append(self.encode_entities(ents))
            head_idxs.append(head_idx)

        return {
            "id": index, "ent_paths": ent_paths, "rel_paths": rel_paths,
            "head_indexes": head_idxs, "relation": self.tokens_to_idxs[rel.item()],
        }


class TripleEntityMultiPathDataset(MultiPathDataset):

    def __init__(self,
                 path_store: str,
                 relcontext_store: str,
                 triple_store: torch.tensor,
                 context_triple_store: torch.tensor = None,
                 tokens_to_idxs: Dict[int, int] = None,
                 maximum_triple_paths = 50,
                 num_negatives: int = 0,
                 triple_corruptor = None,
                 seed: int = 46,
                 parallel=False,
                 neg_triple_store = None,
                ):
        """
        A triple-centric dataset where the given head and tail are used to grab
        paths traversing such entities, individually. The triple item is then
        defined as a balanced set of paths starting or ending from/at heads and
        tails; without introducing any particular contraint. This comes at the
        benefit of avoiding combinatorial issues of the TripleMultiPathDataset.
        """
        super().__init__(path_store,relcontext_store, triple_store,
                         context_triple_store, tokens_to_idxs,
                         maximum_triple_paths, num_negatives,
                         triple_corruptor, seed, parallel, neg_triple_store)

    def _getitem_separate(self, index) -> dict:
        """
        Get a multi-paths contextualisation of the i-th triple in the dataset,
        where incoming and outgoing paths are kept independent.
        """
        triple = torch.as_tensor(self.triplestore[index])
        head, rel, tail = triple
        # Retrieving incoming and outgoing paths/triples per entity
        in_hpaths, out_hpaths = plib.create_contextpaths(
            head, self.path_index, self.context_triple_store)
        in_tpaths, out_tpaths = plib.create_contextpaths(
            tail, self.path_index, self.context_triple_store)
        # Record the origin of each path (0 for H, 1 for T) capped at ppc
        max_enpaths = lambda x: min(self.ppc, len(x))
        # Saving path origins and directionality
        path_ori = [0] * (max_enpaths(in_hpaths) + max_enpaths(out_hpaths)) + \
                   [1] * (max_enpaths(in_tpaths) + max_enpaths(out_tpaths))
        path_dir = [0] * max_enpaths(in_hpaths) + [1] * max_enpaths(out_hpaths) + \
                   [0] * max_enpaths(in_tpaths) + [1] * max_enpaths(out_tpaths)

        # Sample paths for each entity context using local seed
        with du.local_seed(self.seed, self.epoch, index):
            context_paths = []  # holding all the context paths
            for context in [in_hpaths, out_hpaths, in_tpaths, out_tpaths]:
                context_paths += [context[i] for i in
                          torch.randperm(len(context))[:self.ppc]]

        ent_paths, rel_paths, head_idxs, tail_idxs, pos = [], [], [], [], []
        for i, context_path in enumerate(context_paths):
            sel = max if path_dir[i] == 0 else min  # for self-loop
            # Instantiate context paths to retrieve entities and relations
            ents, rels = self.fetch_path_context(*context_path)
            heix = sel((ents == head).nonzero()).item() if head in ents else -1
            teix = sel((ents == tail).nonzero()).item() if tail in ents else -1
            # Creates a custom positional encoding centered around the entity
            entix = heix if path_ori[i] == 0 else teix
            pos.append(plib.get_entfocused_positionals(
                entix*2, len(ents)*2-1) + 1)  # adding 1 to avoid padding
            # FIXME Mask the relation to predict if the triple occurs in paths
            # Encode both relations and entities, and mask the bridge relation
            rels = self.encode_relations(rels)
            # rels[head_idx] = self.tokens_to_idxs["MSK"]
            rel_paths.append(rels)
            ent_paths.append(self.encode_entities(ents))
            head_idxs.append(heix)
            tail_idxs.append(teix)

        return {
            "id": index, "ent_paths": ent_paths, "rel_paths": rel_paths,
            "head_indexes": head_idxs, "tail_indexes": tail_idxs,
            "relation": self.tokens_to_idxs[rel.item()],
            "pos": pos, "ori_triple": triple, "path_origins": path_ori,
        }

    def _getitem_combined(self, index) -> dict:
        """
        Get a multi-path contextualisation of the i-th triple in the dataset.
        Paths are constructed using both incoming and outgoing context.
        """
        triple = torch.as_tensor(self.triplestore[index])
        head, rel, tail = triple
        # Retrieving combined incoming and outgoing paths per entity
        with du.local_seed(self.seed, self.epoch, index):
            h_epaths, h_rpaths, h_idxs, h_erpos = \
                self._create_inout_contextpaths(head)
            t_epaths, t_rpaths, t_idxs, t_erpos = \
                self._create_inout_contextpaths(tail)
        no_hpaths, no_tpaths = len(h_epaths), len(t_epaths)
        # Record the origin of each path (0 for H, 1 for T) and update entixs
        path_ori = [0] * no_hpaths + [1] * no_tpaths
        h_idxs = h_idxs + [-1] * no_tpaths
        t_idxs = [-1] * no_hpaths + t_idxs

        return {
            "id": index,
            "pos": h_erpos + t_erpos, 
            "ent_paths": h_epaths + t_epaths,
            "rel_paths": h_rpaths + t_rpaths,
            "head_indexes": h_idxs, "tail_indexes": t_idxs,
            "relation": self.tokens_to_idxs[rel.item()],
            "ori_triple": triple, "path_origins": path_ori,
        }

    def __getitem__(self, index) -> dict:
        """
        Get a multi-paths contextualisation of the i-th triple in the dataset.
        """
        # self._getitem_separate(index)
        return self._getitem_combined(index)


class EntityMultiPathDataset(MultiPathDataset):

    def __init__(self,
                 path_store: str,
                 relcontext_store: str,
                 triple_store: torch.tensor,
                 unique_entities: torch.tensor,
                 paths_per_entity = 20,
                 seed: int = 46,
                ):
        """
        An entity-centric dataset where entities are defined from their paths.
        """
        super().__init__(path_store, relcontext_store, triple_store,
                         context_triple_store=triple_store,
                         maximum_triple_paths=paths_per_entity*2,
                         tokens_to_idxs=None, num_negatives=0,
                         triple_corruptor=None, seed=seed)
        # Recording path budget and spreading it across the in-out context
        self.unique_entities = unique_entities

    def __len__(self):
        return len(self.unique_entities)

    def _getitem_separate(self, index: int) -> dict:
        """
        Get a multi-path contextualisation of the i-th entity in the dataset.
        """
        ent = self.unique_entities[index]

        # Retrieving incoming and outgoing paths/triples per entity
        in_paths, out_paths = plib.create_contextpaths(
            ent, self.path_index, self.context_triple_store)
        # Record the origin of each path (0 for H, 1 for T) capped at ppc
        max_enpaths = lambda x: min(self.ppc, len(x))
        # Saving path directionality: 0 incoming, 1 outgoing
        path_dir = [0] * max_enpaths(in_paths) + [1] * max_enpaths(out_paths)

        # Sample paths for each entity context using local seed
        with du.local_seed(self.seed, index):
            context_paths = []  # holding all the context paths
            for context in [in_paths, out_paths]:
                context_paths += [context[i] for i in
                          torch.randperm(len(context))[:self.ppc]]

        ent_paths, rel_paths, ent_idxs, pos = [], [], [], []
        for i, context_path in enumerate(context_paths):
            sel = max if path_dir[i] == 0 else min  # for self-loop
            # Instantiate context paths to retrieve entities and relations
            ents, rels = self.fetch_path_context(*context_path)
            entix = sel((ents == ent).nonzero()).item() if ent in ents else -1
            # Creates a custom positional encoding centered around the entity
            pos.append(plib.get_entfocused_positionals(
                entix*2, len(ents)*2-1) + 1)  # adding 1 to avoid padding
            # Encode both relations and entities, and mask the bridge relation
            rels = self.encode_relations(rels)
            # rels[head_idx] = self.tokens_to_idxs["MSK"]
            rel_paths.append(rels)
            ent_paths.append(self.encode_entities(ents))
            ent_idxs.append(entix)

        return {
            "id": index, "ent_paths": ent_paths, "rel_paths": rel_paths,
            "head_indexes": ent_idxs, "pos": pos, 
        }

    def _getitem_combined(self, index) -> dict:
        """
        Get a multi-path contextualisation of the i-th entity in the dataset.
        """
        ent = self.unique_entities[index]
        # Retrieving combined incoming and outgoing paths per entity
        with du.local_seed(self.seed, index):
            ent_paths, rel_paths, ent_idxs, pos = \
                self._create_inout_contextpaths(ent)

        return {
            "id": index, "ent_paths": ent_paths, "rel_paths": rel_paths,
            "head_indexes": ent_idxs, "pos": pos, 
        }

    def __getitem__(self, index) -> dict:
        # return self._getitem_separate(index)
        return self._getitem_combined(index)


################################################################################
# Sequence perturbation strategies
# ------------------------------------------------------------------------------

def permute_tokens(sequence, p=1.0):
    """
    Permutes the elements of a given sequence depending on a permutation ratio.
    The original sequence is retained, whereas a new one is returned. Note that
    if `num_to_permute` is equal to 1, no permutation is performed, because 2
    tokens at least are required for an actual permutation.

    Parameters
    ----------
    sequence : torch.tensor
        A sequence of tokens (assumed to be 1D for now)
    p : float
        Ratio of tokens to be permuted in each sequence.

    Note: this can also be done using spans of variable length
    """
    permuted = sequence.clone()
    num_tokens = permuted.size(0)
    # Select number and tokens to permute in the sequence
    num_to_permute = math.ceil((num_tokens * 2 * p) / 2.0)
    # NB: from here on working with indexes
    substitutions = torch.randperm(num_tokens)[:num_to_permute]
    # Select the target substitutions for each token(index)
    ordering = torch.arange(0, num_tokens)  # create the initial ordering
    # Create random ordering of indexes equal to the no. of elements to permute,
    # get the matching indexes from the indexes to be permuted (substitutions)
    # and overwrite them in the ordering
    ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]
    # -- end of index manipulation
    # swap the tokens according to the permuted ordering
    permuted = permuted[ordering]
    return permuted


def add_insertion_noise(sequence, p, random_ratio, mask_idx, vocab_size,
                        low_idx=1, keep_context=False):
    """
    Augment the input sequence with additional tokens, which can either be: a
    mask token, or a random token; where the insertion probability controls the
    amount of extension, whereas a random ratio balances masking vs randomness.

    Parameters
    ----------
    sequence : torch.tensor
        A sequence of tokens (assumed to be 1D for now)
    p : float
        The probability with which to (randomly) insert noise (replace with one
        or more mask or random tokens)
    random_ratio : float
        The probability with which to (randomly) replace a token by a random one
    mask_idx : int
        The index associated to the mask token in the current vocabulary
        (positive int)
    vocab_size : int
        The number of token types (assumed to start from 0, and contiguous)
    low_idx : int
        The lowest index of legit token that can be generated for insertion
    keep_context=False : bool
        Whether the context (start and end tokens) should be kept intact

    Note: to define a probability of insertion even when the size of the input
          tensor/sequence is 0 (to address insertion of empty sequence).
    """
    if p == 0.0 or len(sequence) == 0:  # disabled
        return sequence

    num_tokens = len(sequence)
    num_noise = int(math.ceil(num_tokens * p))
    # Keep context: to not corrupt the start and end token subtract two to
    # avoid the last index (after you add 1 to avoid the first token)
    noise_indices = torch.randperm(num_tokens + num_noise - 2)[:num_noise] + 1 \
        if keep_context else torch.randperm(num_tokens + num_noise)[:num_noise]
    noise_mask = torch.zeros(size=(num_tokens + num_noise,), dtype=torch.bool)
    noise_mask[noise_indices] = 1  # indices to corrupt set to 1 in the mask
    result = torch.LongTensor(num_tokens + num_noise).fill_(-1)

    # Calculate the number of token that will be replaced with random tokens
    num_random = int(math.ceil(num_noise * random_ratio))
    # Get all the indices in the noise mask after the
    # number of random ones and mask them
    result[noise_indices[num_random:]] = mask_idx
    # Get number of indices equal to the number of random
    # ones and swap them with random relation ids
    result[noise_indices[:num_random]] = torch.randint(
        low=low_idx, high=vocab_size, size=(num_random,), dtype=torch.long, )
    # Add in the result the original tokens in the places where
    # the noise mask is false (uncorrupted indices)
    result[~noise_mask] = sequence
    assert (result >= 0).all()
    return result


def add_rolling_noise(sequence):
    """
    Cut a sequence in a random point and swap the order of tokens so that the
    second subsequence precedes the former while keeping start and end tokens.

    Parameters
    ----------
    sequence : torch.tensor
        A sequence of tokens (assumed to be 1D for now)

    """
    # Random selection of an index where the sequence will be rolled
    offset = np.random.randint(1, max(1, sequence.size(-1) - 1) + 1)
    # Invert order of sequence at the split point while preserving start-end
    rotated = torch.cat(
        (sequence[0:1], sequence[offset:-1],
         sequence[1:offset], sequence[-1:]),
        dim=0,
    )
    return rotated


def delete_tokens(sequence, p: float, keep_context=False):
    """
    Randomly select tokens to delete in the sequence, bounded to the total
    number of tokens - 2 to make sure that leading and trailing elements are not
    discarded from the original sequence (they act as a context).

    Parameters
    ----------
    sequence : torch.tensor
        A sequence of tokens (assumed to be 1D for now)
    p : float
        The probability to delete a token in the sequence

    """
    # Num context is the no. of tokens that will not be affected by the perturb
    # Offset is considered from the end of the sequence (1 for keep last)
    num_context, offset = (2, 1) if keep_context else (0, 0)

    num_tokens = len(sequence)
    if p == 0.0 or num_tokens < num_context + 1:
        return sequence  # unchanged

    num_deletions = min(int(math.ceil(p * num_tokens)),
                        num_tokens - num_context)
    # Select indexes of tokens that will be deleted
    deletion_indexes = torch.randperm(num_tokens - num_context) + offset
    deletion_indexes = deletion_indexes[:num_deletions]
    # This is just to delete tokens as a numpy.delete operation
    indices_to_keep = [i for i in range(num_tokens) if
                       i not in deletion_indexes]
    sequence = sequence[torch.LongTensor(indices_to_keep)]

    return sequence


def add_token_mask(sequence, p, mask_idx, random_ratio, vocab_size, low_idx=1,
                   span_distribution=None, delete_masked=False,
                   keep_context=False):
    """
    Randomly mask tokens in a sequence depending on a masking probability. If a
    span distribution is provided, spanning lengths are sampled from it.

    Parameters
    ----------
    sequence : torch.tensor
        A sequence of tokens (assumed to be 1D for now)
    p : float
        The probability of masking tokens in the sequence
    mask_idx : int
        The index associated to the masking token [MASK]
    random_ratio : float
        The proportion of random token to randomise instead of masking
    vocab_size : int
        The number of token types that can be observed in sequences
    span_distribution : torch.Distribution
        A categorical distribution to sample span lengths from
    delete_masked : bool
        Whether the masked tokens will be considered as sequence deletions

    Note: This implementation can also generalise deletion (with spans as well).

    """
    sequence_length = sequence.size(0)
    # To control the keep_context scenario (no masking for leading-trailing)
    num_context, offset = (2, 1) if keep_context else (0, 0)
    # Based on the scenario, select the actual number of tokens to mask
    num_to_mask = max(int(math.ceil((sequence_length - num_context) * p)), 0)
    if num_to_mask == 0: return sequence  # stop!
    sequence = sequence.clone()

    num_inserts = 0  # will count the no. of insertions to compensate masking
    # This denotes which tokens can be masked (1s) and which cannot (0s)
    is_token_start = torch.ones(sequence.size(), dtype=torch.long)
    if keep_context:  # exclude first-last tokens as framing context
        is_token_start[0], is_token_start[-1] = 0, 0
    # Assume the vanilla BERT masking first: 1-span
    lengths = torch.ones((num_to_mask,), dtype=torch.long)

    if span_distribution is not None:  # or alternatively sample spans
        # Sample from the Poisson a span-length for each token to mask
        lengths = span_distribution.sample(sample_shape=(num_to_mask,)).long()
        # Make sure we have enough to mask: add spans to cover `num_to_mask`
        cum_length = torch.cumsum(lengths, 0)  # sum of sampled span lenghts
        while cum_length[-1] < num_to_mask:
            lengths = torch.cat(
                [
                    lengths,
                    span_distribution.sample(sample_shape=(num_to_mask,)),
                ],
                dim=0,
            )
            cum_length = torch.cumsum(lengths, 0)
        # Trim to masking budget (num_to_mask) to avoid excessive masking
        i = torch.argmax((cum_length >= num_to_mask).long()).item()
        lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
        num_to_mask = i + 1  # because i starts from 0 (it is an index)
        lengths = lengths[:num_to_mask]  # the span for each token to mask
        # Handle 0-length mask (inserts) separately: using a separate strategy
        # XXX Currently, they receive an insertion noise that is proportional to
        # the number of 0-length masks; although the reason is not fully clear
        lengths = lengths[lengths > 0]
        num_inserts = num_to_mask - lengths.size(0)
        num_to_mask -= num_inserts
        if num_to_mask == 0:
            return add_insertion_noise(
                sequence, num_inserts / sequence.size(0), random_ratio=1,
                mask_idx=mask_idx, low_idx=low_idx, vocab_size=vocab_size)
        assert (lengths > 0).all(), "No 0-length masking at this stage!"

    # Select the tokens to be masked + create random mask accordingly
    token_starts = is_token_start.nonzero(as_tuple=False)  # idx of token start
    indices = token_starts[
        torch.randperm(token_starts.size(0))[:num_to_mask]].squeeze(1)
    mask_random = torch.FloatTensor(num_to_mask).uniform_() < random_ratio
    # print(f"Indices: {indices} {indices.dtype}\n"
    #       f"Lengths: {lengths} {lengths.dtype}\n"
    #       f"Masks: {mask_random} {mask_random.dtype}")
    assert sequence_length - offset not in indices, \
        "Should not mask last token when keeping context unmasked"
    assert lengths.size() == indices.size()  # equal to num_to_mask

    # Now expand/extend the indices of tokens to mask based on their span
    # For example token 5 with span 2 should take indices 5 and 6
    all_indices = torch.LongTensor(list(itertools.chain.from_iterable(
        [list(range(i, min(j + i, sequence_length - offset)))  # offset 0 or 1
         for i, j in zip(indices, lengths)]
    )))
    assert sequence_length - offset not in indices
    lengths = lengths - torch.nn.functional.relu((lengths + indices)
                                                 - (sequence_length - offset))
    all_mask = torch.repeat_interleave(mask_random, lengths)
    # print(f"AllIndices: {all_indices} {all_indices.dtype}\n"
    #       f"AllMask: {all_mask} {all_mask.dtype}")

    # Just assume that all tokens will be kept rather than deleted
    to_keep = torch.ones(sequence_length, dtype=torch.bool)  # 1 for keep
    if delete_masked:  # flag out tokens as deletions (do not keep) if needed
        to_keep[all_indices] = 0
    else:  # keep index, but replace it with [MASK] (FIXME replace 1)
        sequence[all_indices] = mask_idx
        sequence[all_indices[all_mask]] = torch.randint(
            low_idx, vocab_size, size=(all_mask.sum(),), dtype=torch.long)

    sequence = sequence[to_keep]
    return sequence


################################################################################
# Transformations (wrapping the atomic perturbation strategies above)
# ------------------------------------------------------------------------------

class TokenPermutation(torch.nn.Module):
    """
    Shuffle elements of a sequence through permutation.

    Parameters
    ----------
    permutation_ratio : float
        Ratio of sentences to be permuted in each document. If equal to 1 (the
        maximum ratio), a random permutation for the whole sequence is produced;
        if equal to 0, the sequence is left unchanged.
    """

    def __init__(self, permutation_ratio):
        super().__init__()
        self.permutation_ratio = permutation_ratio

    def forward(self, sequence):
        return permute_tokens(sequence, p=self.permutation_ratio)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pratio={self.permutation_ratio})"


class SequenceInsertionNoise(torch.nn.Module):
    """
    Sequence augmentation through masking and noise.
    This transformation augments a given sequence with additional tokens. Based
    on the parametrisation, new tokens can either be masks or uniformly sampled.

    Parameters
    ----------
    insertion_ratio : float
        The probability with which to (randomly) insert noise in a sequence
    random_ratio : float
        The probability with which to (randomly) replace a token by a random one
        among all the new insertions (the remaining ones will be masked)
    mask_idx : int
        The index associated to the mask token in the current vocabulary
    vocab_size : int
        The number of token types (assumed to start from 0, and contiguous)
    """

    def __init__(self, insertion_ratio, random_ratio, mask_idx, vocab_size,
                 low_idx=1, keep_context=False):
        super().__init__()
        self.insertion_ratio = insertion_ratio
        self.random_ratio = random_ratio
        self.vocab_size = vocab_size
        self.mask_idx = mask_idx
        self.low_idx = low_idx
        self.keep_context = keep_context

    def forward(self, sequence):
        return add_insertion_noise(
            sequence,
            p=self.insertion_ratio,
            random_ratio=self.random_ratio,
            vocab_size=self.vocab_size,
            mask_idx=self.mask_idx,
            low_idx=self.low_idx,
            keep_context=self.keep_context,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}" \
               f"(insertion-ratio={self.insertion_ratio}," \
               f" random-ratio={self.random_ratio})"


class SequenceRolling(torch.nn.Module):
    """
    Rotate a sequence over a (random) split point, while preserving the start
    and the end tokens. The second sub-sequence will preceed the first.
    """

    def __init__(self):
        super().__init__()

    def forward(self, sequence):
        return add_rolling_noise(sequence)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class TokenDeletion(torch.nn.Module):
    """
    Delete randomly selected tokens in a sequence according to a given ration.
    The first and the last tokens in the sequence are always preserved.

    Parameters
    ----------
    deletion_ratio : float
        The probability to delete a token in a sequence
    """

    def __init__(self, deletion_ratio):
        super().__init__()
        self.deletion_ratio = deletion_ratio

    def forward(self, sequence):
        return delete_tokens(sequence, p=self.deletion_ratio)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dratio={self.deletion_ratio})"


class SequenceMasking(torch.nn.Module):
    """
    Apply sequence masking using either vanilla BERT or SpanBERT strategies.
    This behaviour is controlled by an optional span distribution, and can also
    be used to apply sequence deletion on consecutive elements.

    Parameters
    ----------
    masking_ratio : float
        The probability of masking tokens in the sequence (1 mask all tokens,
        whereas 0 does not apply any masking as extreme cases)
    mask_idx : int
        The index associated to the masking token [MASK] in the vocabulary
    random_ratio : float
        The proportion of tokens to randomise instead of masking them
    vocab_size : int
        The number of token types that can be observed in sequences
    span_distribution : torch.Distribution
        If provided, the categorical distribution is used to sample span lengths
        for each starting token to mask; if not provided, 1-len spans are used
    delete_masked : bool
        Whether the masked tokens will be considered as sequence deletions
    """

    def __init__(self, masking_ratio, mask_idx, random_ratio, vocab_size,
                 low_idx=1, span_distribution=None, delete_masked=False):
        super().__init__()
        self.masking_ratio = masking_ratio
        self.random_ratio = random_ratio
        self.mask_idx = mask_idx
        self.vocab_size = vocab_size
        self.low_idx = low_idx
        self.span_distribution = span_distribution
        self.delete_masked = delete_masked

    def forward(self, sequence):
        return add_token_mask(
            sequence,
            p=self.masking_ratio,
            mask_idx=self.mask_idx,
            random_ratio=self.random_ratio,
            vocab_size=self.vocab_size,
            low_idx=self.low_idx,
            span_distribution=self.span_distribution,
            delete_masked=self.delete_masked
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}" \
               f"(masking-ratio={self.masking_ratio}," \
               f" random-ratio={self.random_ratio}," \
               f" delete-masked={self.delete_masked}," \
               f" span-distrib={self.span_distribution})"


def get_data_transforms(strategy,
                        mask_idx: int,
                        low_idx: int,
                        vocab_size: int,
                        permutation_ratio: float = 0,
                        masking_ratio: float = 0,
                        insertion_ratio: float = 0,
                        random_ratio: float = 0,
                        deletion_ratio: float = 0,
                        poisson_lambda: float = 1.,
                        use_rolling: bool = False,
                        p_perturbation: float = .5,
                        ):
    """
    This works as a factory of a perturbation strategy: a chain of atomic ops
    applied to a sequential stream, whose behaviour is fully parametrised and
    also controlled by the name of the strategy that is chosen. Note that, by
    default, all atomic perturbations are disabled due to initialisation.

    Parameters
    ----------
    strategy : str or fn
        The strategy to use to combine or choose among atomic perturbations.
    mask_idx : int
        The idx associated to the masking token in the encoding in use.
    low_idx : int
        The lowest token idx in the vocabulary, following special tokens.
    vocab_size : int
        The size of the vocabulary, which will be used for noise tokens.
    permutation_ratio : float, optional
        The ratio at which tokens will be permuted; 0 to deactivate.
    masking_ratio : float, optional
        The ratio at which tokens will be masked; 0 to deactivate.
    insertion_ratio : float, optional
       The ratio at which tokens will be inserted; 0 to deactivate.
    random_ratio : float, optional
        The ratio at which tokens will be randomised, when requested.
    deletion_ratio : float, optional
        The ratio at which tokens will be deleted; 0 to deactivate.
    poisson_lambda : float, optional
        Lambda value for the poisson distribution used for span-masking.
    p_perturbation : float, optional
        If ``strategy == 'rapply'`` this is the probability to perturb.

    Returns
    -------
    torch.Module
        A function defined on sequences that implements the required strategy.

    Note: FIXME At the moment the vocabular size may lead to inconsistent tokens
          and this happens because a lower bound (offsetting the special tokens)
          is not specified at the moment.
    """
    transformations = []
    span_distrib = du.span_poisson(poisson_lambda)

    if permutation_ratio > 0:
        transformations.append(TokenPermutation(
            permutation_ratio=permutation_ratio
        ))
    if insertion_ratio > 0:
        transformations.append(SequenceInsertionNoise(
            insertion_ratio=insertion_ratio,
            random_ratio=random_ratio,
            mask_idx=mask_idx,
            vocab_size=vocab_size,
            low_idx=low_idx,
        ))
    if masking_ratio > 0:
        transformations.append(SequenceMasking(
            masking_ratio=masking_ratio,
            mask_idx=mask_idx,
            random_ratio=random_ratio,
            vocab_size=vocab_size,
            low_idx=low_idx,
            span_distribution=span_distrib,
            delete_masked=False
        ))
    if deletion_ratio > 0:
        transformations.append(TokenDeletion(
            deletion_ratio=deletion_ratio
        ))
    if use_rolling:
        transformations.append(SequenceRolling())

    if strategy in ["sequential", "rapply"]:
        transformations = torch.nn.Sequential(*transformations) \
            if p_perturbation == 1 else RandomApply(
                transformations, p=p_perturbation)
    elif strategy == "rchoice":  # sthochastic choice
        transformations = RandomChoice(transformations, p=[.3, .1, .3, .1, .1])
    elif strategy == "rorder":
        transformations = RandomOrder(transformations)
    else:  # the strategy is unknown / not supported
        raise ValueError(f"Not a supported strategy: {strategy}")

    return transformations


def create_datasets(path_store: str, xtokens: list, strategy: str,
                    split_idxs=None, **kwargs):
    """
    Create the main dataset as well as the specific tr/va/te partitions, which
    will be either reconstructed from a split assignment or randomly created.

    Note: TODO should be parameterised according to the special tokens used
    Note: should remove repeated code with `load_datasets`
    Note: currently the split idxs need to be read from file first.
    """
    # Create the mother dataset wrapping all possible splits
    dataset = PathDataset(path_store=path_store, xtokens=xtokens)
    if split_idxs is None:  # no dataset split is provided, do a random split
        logger.info("Creating new random splits for path dataset")
        train_set, valid_set, test_set = du.split_datasets(dataset)
    else:  # subsets are created based on the given split
        logger.info("Using given split assignments to partition the dataset")
        train_set, valid_set, test_set = du.partition_dataset(dataset,
                                                              *split_idxs)
    # Update the data transform operations for the training set (currently all)
    transforms = get_data_transforms(
        strategy, mask_idx=dataset.tokens_to_idxs["MSK"],
        vocab_size=dataset.vocab_size, **kwargs)
    logger.info(f"Using {strategy} strategy with {len(transforms)} transforms")
    # Setting transformation into the main (mother) dataset; will apply to all
    # If this is not what you want, then you should create 3 different datasets
    dataset.transform = transforms
    # train_set.transform = transforms["train"]
    return train_set, valid_set, test_set


def load_datasets(train_pathstore, valid_pathstore, test_pathstore,
                  xtokens: list, strategy: str, **kwargs):
    """
    Read and load the datasets to use for training, validation, and test using
    the files provided at the given locations. Dataset creation is parametrised
    with special tokens, and perturbation strategy (type, and hparams).

    Note: TODO should be parameterised according to the special tokens used
    Note: epoch-wise seeding of datasets will override the main seed.
    Note: should remove repeated code with `create_datasets`
    """
    # Create the vocabulary once, reuse it for all the other sets
    train_set = PathDataset(path_store=train_pathstore, xtokens=xtokens)
    valid_set = PathDataset(path_store=valid_pathstore, xtokens=xtokens,
                            tokens_to_idxs=train_set.tokens_to_idxs)
    test_set = PathDataset(path_store=test_pathstore, xtokens=xtokens,
                           tokens_to_idxs=train_set.tokens_to_idxs)

    transforms = get_data_transforms(
        strategy, mask_idx=train_set.tokens_to_idxs["MSK"],
        vocab_size=train_set.vocab_size, **kwargs, )
    logger.info(f"Using {strategy} strategy with {len(transforms)} transforms")
    for pset in [train_set, valid_set, test_set]:  # XXX should exclude testset
        pset.transform = transforms

    return train_set, valid_set, test_set


def load_unrolled_datasets(train_pathstore, valid_pathstore, test_pathstore,
                           pstrategy_dict: dict, relcontext_size: int):
    """
    The analogous method to ``load_dataset`` for loading an unrolled dataset,
    using the file system path of each partition (this is now the pathstore).
    """
    # First step is to create the three dataset, reusing the token mapping
    data_partitions, tokens_to_idxs = [], None
    for data_pathstore in [train_pathstore, valid_pathstore, test_pathstore]:
        # Retrieving the filesystem path of: paths, rel context and neighbour
        paths, relcon, relnei = du.load_unrolled_setup(data_pathstore)
        partition = UnrolledPathDataset(
            path_store=paths, relneighb_store=relnei, relcontext_store=relcon,
            relcontext_size=relcontext_size, tokens_to_idxs=tokens_to_idxs
        )  # this is populated at first shot, for training
        tokens_to_idxs = partition.tokens_to_idxs \
            if tokens_to_idxs is None else tokens_to_idxs
        data_partitions.append(partition)  # order-sensitive

    train_set, valid_set, test_set = data_partitions
    # Second step is to create the localised transformation and register them
    # Arguments shared among all the localised transformation strategies
    base_params = dict(mask_idx=train_set.tokens_to_idxs["MSK"],
                       low_idx=train_set.vocab_size - train_set.no_relations,
                       vocab_size=train_set.vocab_size, )

    transforms = []  # holding all the tranformation, in order
    for target in ["base", "rp", "ep"]:  # transformation target (base == RC)
        transform = get_data_transforms(**pstrategy_dict[target], **base_params)
        logger.info(f"Perturb strategy for {target}: {transform}")
        transforms.append(transform)
    # Registering the transformations in the training and validation sets
    for partition in [train_set, valid_set]:
        partition.transform_rc = transforms[0]
        partition.transform_re = transforms[1]
        partition.transform_en = transforms[2]

    return train_set, valid_set, test_set
