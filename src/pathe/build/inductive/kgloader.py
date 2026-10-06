import os

import torch
from pykeen.datasets.inductive import InductiveWN18RR, InductiveFB15k237, InductiveNELL
from pykeen.triples import TriplesFactory

from .. import statics


class KgLoader:
    """
    A class that loads the graph datasets and converts them to a common format
    of torch.Tensor of shape [num_triples, 3 - (head id,relation id,tail id)].
    The class also provides train, test and val splits as well as the total number of nodes
    """
    dataset: str
    train_triples: torch.Tensor
    val_triples: torch.Tensor
    test_triples: torch.Tensor
    load_success: bool
    num_nodes_total: int
    add_inverse: bool

    def __init__(self, dataset: str = None, add_inverse: bool = None,
                 part: str = None):
        """
        :param dataset: the name of the inductive dataset. Currently supported:
        fb15k237, wn18rr
        :param add_inverse: If inverse edges should be added
        """
        # Unique nodes in train, val, test and inference graph
        # val and test are sampled from the inference graph which is disjoint
        # from the train one (no common entities)


        self.unique_total_inference = None
        self.unique_test = None
        self.unique_val = None
        self.unique_train = None
        self.unique_inference = None

        # FIXME maybe not needed
        self.unique_val_rel = None
        self.unique_test_rel = None
        self.unique_inference_rel = None
        self.unique_train_rel = None

        # the triples
        self.test_no_inv = None
        self.train_no_inv = None
        self.val_no_inv = None
        self.inference_no_inv = None
        self.inference_triples_total = None
        self.inference_triples_total_no_inv = None
        self.inference_triples = None

        # Total unique nodes (maybe useless)
        self.num_nodes_train = 0
        self.num_nodes_inference = 0
        self.num_nodes_inference_total = 0
        self.num_nodes_total = 0
        self.dataset = dataset.lower() if dataset is not None else 'fb15k237'
        self.add_inverse = add_inverse if add_inverse is not None else True
        self.part = part if part is not None else 'v1'
        self.triple_factory = None
        self.unique_relations = None
        self.unique_entities = None
        self.entity_stats = {}
        self.edge_stats = {}
        self.edge_stats_no_inv = {}
        self.load_success = self.load_data()
        if not self.load_success:
            raise Exception("Unable to load the selected dataset.")
        self.get_unique_relations_and_entities()
        # self.count_entity_stats()
        # self.count_edge_stats()
        # self.count_edge_stats_no_inv()
        # self.count_and_save_edge_stats()
        # self.count_and_save_entity_stats()

    def load_data(self):
        """
        Loads the data of the selected dataset.
        :return: True if triples were loaded successfully, False otherwise
        """
        if self.dataset == 'fb15k237':
            self.triple_factory = InductiveFB15k237(
                create_inverse_triples=self.add_inverse, version=self.part)
        elif self.dataset == 'wn18rr':
            self.triple_factory = InductiveWN18RR(
                create_inverse_triples=self.add_inverse, version=self.part)
        elif self.dataset == 'nell':
            self.triple_factory = InductiveNELL(
                create_inverse_triples=self.add_inverse, version=self.part)
        if self.triple_factory is not None:
            return self.prepare_triples()
        else:
            return False

    def prepare_triples(self):
        """
        A function that reads the triples from the datasets, splits them into train.val and test sets
        and converts them to Torch tensors of shape (num triples, 3). The first column is the Head entity
        the second is the relation and the third is the tail entity
        :return: True if tensors were populated successfully, false otherwise
        """
        if self.dataset in ('fb15k237', 'wn18rr', 'nell'):
            self.num_nodes_train = (
                self.triple_factory.transductive_training.num_entities)
            self.num_nodes_inference = (
                self.triple_factory.inductive_inference.num_entities)
            # to get the inverse triples as well
            if self.add_inverse:
                self.train_triples = (
                    self.triple_factory.transductive_training
                    ._add_inverse_triples_if_necessary(
                    self.triple_factory.transductive_training.mapped_triples))
                self.val_triples = (
                    self.triple_factory.inductive_validation
                    ._add_inverse_triples_if_necessary(
                    self.triple_factory.inductive_validation.mapped_triples))
                self.test_triples = (
                    self.triple_factory.inductive_testing
                    ._add_inverse_triples_if_necessary(
                    self.triple_factory.inductive_testing.mapped_triples))
                self.inference_triples = (
                    self.triple_factory.inductive_inference
                    ._add_inverse_triples_if_necessary(
                    self.triple_factory.inductive_inference.mapped_triples))
                self.inference_triples_total = torch.cat((self.val_triples,
                                                            self.test_triples, 
                                                            self.inference_triples), dim=0)
                self.train_no_inv = self.triple_factory.transductive_training.mapped_triples
                self.val_no_inv = self.triple_factory.inductive_validation.mapped_triples
                self.test_no_inv = self.triple_factory.inductive_testing.mapped_triples
                self.inference_no_inv = self.triple_factory.inductive_inference.mapped_triples
                self.inference_triples_total_no_inv = torch.cat((
                    self.triple_factory.inductive_validation.mapped_triples,
                    self.triple_factory.inductive_testing.mapped_triples, 
                    self.triple_factory.inductive_inference.mapped_triples), 
                    dim=0)

            else:
                self.train_triples = self.triple_factory.transductive_training.mapped_triples
                self.val_triples = self.triple_factory.inductive_validation.mapped_triples
                self.test_triples = self.triple_factory.inductive_testing.mapped_triples
                self.inference_triples = self.triple_factory.inductive_inference.mapped_triples
                self.inference_triples_total = torch.cat((
                    self.triple_factory.inductive_validation.mapped_triples,
                    self.triple_factory.inductive_testing.mapped_triples,
                    self.triple_factory.inductive_inference.mapped_triples),
                    dim=0)
                self.train_no_inv = self.triple_factory.transductive_training.mapped_triples
                self.val_no_inv = self.triple_factory.inductive_validation.mapped_triples
                self.test_no_inv = self.triple_factory.inductive_testing.mapped_triples
                self.inference_no_inv = self.triple_factory.inductive_inference.mapped_triples
                self.inference_triples_total_no_inv = torch.cat((
                    self.triple_factory.inductive_validation.mapped_triples,
                    self.triple_factory.inductive_testing.mapped_triples,
                    self.triple_factory.inductive_inference.mapped_triples),
                    dim=0)
            self.offset_inductive_entities()
            return True
        else:
            return False

    def offset_inductive_entities(self):
        max_train_index = torch.max(self.train_triples)
        self.val_triples[:,0] += max_train_index
        self.val_triples[:,2] += max_train_index
        self.test_triples[:,0] += max_train_index
        self.test_triples[:, 2] += max_train_index
        self.inference_triples[:,0] += max_train_index
        self.inference_triples[:,2] += max_train_index
        self.inference_triples_total[:,0] += max_train_index
        self.inference_triples_total[:,2] += max_train_index
        if self.add_inverse:
            self.val_no_inv[:,0] += max_train_index
            self.test_no_inv[:,0] += max_train_index
            self.inference_no_inv[:,0] += max_train_index
            self.inference_triples_total_no_inv[:,0] += max_train_index
            self.val_no_inv[:,2] += max_train_index
            self.test_no_inv[:,2] += max_train_index
            self.inference_no_inv[:,2] += max_train_index
            self.inference_triples_total_no_inv[:,2] += max_train_index


    def count_and_save_edge_stats(self):
        """
        A function that saves unique edges and their frequencies for the training set
        :return: None
        :rtype: None
        """
        save_dir = os.path.join(os.path.join(
            statics.data_dir(), "Dataset_Stats"), self.dataset)
        statics.check_and_make_dir(save_dir)
        save_path = os.path.join(save_dir,
                                 "edge_stats_" + self.dataset + ".csv")
        if not os.path.exists(save_path):
            unique, counts = torch.unique(self.train_triples[:, 1],
                                          return_counts=True)
            with open(save_path, 'w') as f:
                f.write("Edge;Count\n")
                unique = unique.tolist()
                counts = counts.tolist()
                for idx, x in enumerate(unique):
                    f.write(str(unique[idx]) + ";" + str(counts[idx]) + "\n")

    def count_and_save_entity_stats(self):
        """
        A function that saves the unique entities and their frequencies for the heads, the tails,
        the head-only and the tail-only entities. Moreover the function saves for each head node
        the number of outgoing edges, the number of unique outgoing edges and all the outgoing edges
        as well as their frequencies. Finally, the same is saved for all incoming edges of the tail nodes
        :return:
        :rtype:
        """
        save_dir = os.path.join(os.path.join(
            statics.data_dir(), "Dataset_Stats"), self.dataset)
        statics.check_and_make_dir(save_dir)
        for part in ['train', 'validation', 'test']:
            if part == 'train':
                triples = self.train_no_inv
            elif part == 'validation':
                triples = self.val_no_inv
            else:
                triples = self.test_no_inv
            save_path_heads = os.path.join(save_dir,
                                           "node_stats_heads_" + self.dataset + "_" + part + ".csv")
            save_path_out_edges = os.path.join(save_dir,
                                               "node_stats_heads_edges_" + self.dataset + "_" + part + ".csv")
            save_path_tails = os.path.join(save_dir,
                                           "node_stats_tails_" + self.dataset + "_" + part + ".csv")
            save_path_in_edges = os.path.join(save_dir,
                                              "node_stats_tails_edges_" + self.dataset + "_" + part + ".csv")
            save_path_head_only = os.path.join(save_dir,
                                               "node_only_heads_" + self.dataset + "_" + part + ".csv")
            save_path_tail_only = os.path.join(save_dir,
                                               "node_only_tails_" + self.dataset + "_" + part + ".csv")
            unique_heads = []
            unique_tails = []
            head_counts = {}
            tail_counts = {}
            tail_only = []
            head_only = []
            num_outgoing = {}
            outgoing_edges = {}
            num_incoming = {}
            incoming_edges = {}
            if not os.path.exists(save_path_heads) or not os.path.exists(
                    save_path_out_edges):
                unique, counts = torch.unique(triples[:, 0], return_counts=True)
                unique = unique.tolist()
                for head in unique:
                    head_outgoing = triples[triples[:, 0] == head]
                    edges, numbers = torch.unique(head_outgoing[:, 1],
                                                  return_counts=True)
                    outgoing_edges[head] = {'Total': head_outgoing.size()[0],
                                            'unique': edges.size()[0],
                                            'edges': edges.tolist(),
                                            'counts': numbers.tolist()}
                with open(save_path_heads, 'w') as f:
                    f.write("Node;Count\n")
                    counts = counts.tolist()
                    unique_heads = set(unique)
                    for idx, x in enumerate(unique):
                        head_counts[x] = counts[idx]
                        f.write(
                            str(unique[idx]) + ";" + str(counts[idx]) + "\n")
                with open(save_path_out_edges, 'w') as f1:
                    f1.write("Node;Total;Unique;edge:count\n")
                    for key in outgoing_edges.keys():
                        f1.write(str(key) + ";" + str(
                            outgoing_edges[key]["Total"]) + ";" + str(
                            outgoing_edges[key]["unique"]) + ";")
                        edge_list = outgoing_edges[key]['edges']
                        count_list = outgoing_edges[key]['counts']
                        for index, edge in enumerate(edge_list):
                            f1.write(
                                str(edge) + ";" + str(count_list[index]) + ";")
                        f1.write("\n")
            if not os.path.exists(save_path_tails) or not os.path.exists(
                    save_path_in_edges):
                unique, counts = torch.unique(triples[:, 2], return_counts=True)
                unique = unique.tolist()
                for tail in unique:
                    tail_incoming = triples[(triples[:, 2] == tail)]
                    edges, numbers = torch.unique(tail_incoming[:, 1],
                                                  return_counts=True)
                    incoming_edges[tail] = {'Total': tail_incoming.size()[0],
                                            'unique': edges.size()[0],
                                            'edges': edges.tolist(),
                                            'counts': numbers.tolist()}
                with open(save_path_tails, 'w') as f:
                    f.write("Node;Count\n")
                    counts = counts.tolist()
                    unique_tails = set(unique)
                    for idx, x in enumerate(unique):
                        tail_counts[x] = counts[idx]
                        f.write(
                            str(unique[idx]) + ";" + str(counts[idx]) + "\n")
                with open(save_path_in_edges, 'w') as f1:
                    f1.write("Node;Total;Unique;edge:count\n")
                    for key in incoming_edges.keys():
                        f1.write(str(key) + ";" + str(
                            incoming_edges[key]["Total"]) + ";" + str(
                            incoming_edges[key]["unique"]) + ";")
                        edge_list = incoming_edges[key]['edges']
                        count_list = incoming_edges[key]['counts']
                        for index, edge in enumerate(edge_list):
                            f1.write(
                                str(edge) + ";" + str(count_list[index]) + ";")
                        f1.write("\n")
            if not os.path.exists(save_path_head_only):
                head_only = unique_heads.difference(unique_tails)
                with open(save_path_head_only, 'w') as f:
                    f.write("Node;Count\n")
                    for x in head_only:
                        f.write(str(x) + ";" + str(head_counts[x]) + "\n")

            if not os.path.exists(save_path_tail_only):
                tail_only = unique_tails.difference(unique_heads)
                with open(save_path_tail_only, 'w') as f:
                    f.write("Node;Count\n")
                    for x in tail_only:
                        f.write(str(x) + ";" + str(tail_counts[x]) + "\n")
            self.entity_stats[part] = {
                'unique_heads': unique_heads,
                'unique_tails': unique_tails,
                'count_heads': head_counts,
                'count_tails': tail_counts,
                'head_only': head_only,
                'tail_only': tail_only
            }

    def count_entity_stats(self):
        """
        A function that records for each of the train, val and test sets
        the unique heads, unique tails, their counts and the head only and tail only nodes.
        These are all stored in a dictionary
        :return: None
        :rtype: Node
        """
        for part in ['train', 'validation', 'test']:
            if part == 'train':
                triples = self.train_no_inv
            elif part == 'validation':
                triples = self.val_no_inv
            else:
                triples = self.test_no_inv
            head_counts = {}
            tail_counts = {}
            unique, counts = torch.unique(triples[:, 0], return_counts=True)
            unique = unique.tolist()
            counts = counts.tolist()
            unique_heads = set(unique)
            for idx, x in enumerate(unique):
                head_counts[x] = counts[idx]
            unique, counts = torch.unique(triples[:, 2], return_counts=True)
            unique = unique.tolist()
            counts = counts.tolist()
            unique_tails = set(unique)
            for idx, x in enumerate(unique):
                tail_counts[x] = counts[idx]
            head_only = unique_heads.difference(unique_tails)
            tail_only = unique_tails.difference(unique_heads)
            self.entity_stats[part] = {
                'unique_heads': unique_heads,
                'unique_tails': unique_tails,
                'count_heads': head_counts,
                'count_tails': tail_counts,
                'head_only': head_only,
                'tail_only': tail_only
            }

    def count_edge_stats(self):
        """
        A function that records the unique edges and their frequency
        :return:
        :rtype:
        """
        for part in ['train', 'validation', 'test']:
            if part == 'train':
                triples = self.train_triples
            elif part == 'validation':
                triples = self.val_triples
            else:
                triples = self.test_triples
            edge_counts = {}
            unique, counts = torch.unique(triples[:, 1], return_counts=True)
            unique_edges = unique.tolist()
            counts = counts.tolist()
            for idx, x in enumerate(unique_edges):
                edge_counts[x] = counts[idx]
            self.edge_stats[part] = {
                'unique_edges': set(unique_edges),
                'edge_counts': edge_counts,
            }

    def count_edge_stats_no_inv(self):
        """
        A function that records the unique edges and their frequency for each part of the dataset.
        These are measured before adding inverse edges.
        :return: None
        :rtype: None
        """
        for part in ['train', 'validation', 'test']:
            if part == 'train':
                triples = self.train_no_inv
            elif part == 'validation':
                triples = self.val_no_inv
            else:
                triples = self.test_no_inv
            edge_counts = {}
            unique, counts = torch.unique(triples[:, 1], return_counts=True)
            unique_edges = unique.tolist()
            counts = counts.tolist()
            for idx, x in enumerate(unique_edges):
                edge_counts[x] = counts[idx]
            self.edge_stats_no_inv[part] = {
                'unique_edges': set(unique_edges),
                'edge_counts': edge_counts,
            }

    def get_unique_relations_and_entities(self):
        """
        A function that stores the unique entities and relations in the entire dataset
        :return: None
        :rtype: None
        """
        triples = torch.cat((self.train, self.validation, self.test, self.inference),
                            dim=0)
        self.unique_entities = torch.stack((triples[:, 0], triples[:, 2]),
                                           dim=0).unique(sorted=True)
        self.num_nodes_total = self.unique_entities.size()[0]
        self.unique_relations = triples[:, 1].unique(sorted=True)
        self.unique_train = torch.stack((self.train_triples[:, 0],
                                        self.train_triples[:, 2]), dim=0).unique(
            sorted=True)
        self.unique_test = torch.stack((self.test_triples[:, 0],
                                        self.test_triples[:, 2]), dim=0).unique(
            sorted=True)
        self.unique_val = torch.stack((self.val_triples[:, 0],
                                       self.val_triples[:, 2]), dim=0).unique(
            sorted=True)
        self.unique_inference = torch.stack((self.inference_triples[:, 0],
                                       self.inference_triples[:, 2]), dim=0).unique(
            sorted=True)
        self.unique_total_inference = torch.stack((
            self.inference_triples_total[:, 0],
                                       self.inference_triples_total[:, 2]), dim=0).unique(
            sorted=True)
        self.unique_test_rel = self.test_triples[:, 1].unique(
            sorted=True)
        self.unique_val_rel = self.val_triples[:, 1].unique(
            sorted=True)
        self.unique_inference_rel = self.inference_triples_total[:,1].unique(
            sorted=True)
        self.unique_train_rel = self.train_triples[:, 1].unique(
            sorted=True)

    @property
    def train(self):
        return self.train_triples

    @property
    def validation(self):
        return self.val_triples

    @property
    def test(self):
        return self.test_triples

    @property
    def inference(self):
        return self.inference_triples

    @property
    def inference_all(self):
        return self.inference_triples_total

    @property
    def data_object(self):
        return self.triple_factory

    @property
    def num_total_nodes(self):
        return self.num_nodes_total


# dataset = 'fb15k237'
# g = KgLoader('fb15k237', add_inverse=False)
# print()