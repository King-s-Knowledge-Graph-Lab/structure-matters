"""
Scalable routines for training a PathE model.

"""
import os
import logging
import datetime
from functools import partial
from math import floor

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer

from . import triple_lib
from .data_utils import load_triple_tensors_inductive
# from pathe.debug_parallel import unique_entities
# from pathe.pather_models import num_nodes, num_dim
from .pather_models import PathEModel
from .pathdata import NegativeTripleSampler, TripleEntityMultiPathDataset
from .data_utils import collate_multipaths, load_triple_tensors, \
    load_corrupted_triples_from_dir, memmap_corrupted_triples_from_dir
from .callbacks import DatasetUpdater
from .corruption import CorruptHeadGenerator, CorruptTailGenerator, \
    CorruptLinkGenerator, SeededLinkCorruptor
from .utils import stageprint, bundle_arguments, namespace_to_dict
from .wrappers import PathEModelWrapper
from .path_lib import encode_relcontext_freqs
from . import data_utils as du

logger = logging.getLogger(__name__)


def create_and_run_training_exp(args):
    """
    Main entry point for the experimental setup, execution, and evaluation.
    """
    model_name = args.model
    start_time = datetime.datetime.now()
    stageprint(f"Starting {args.expname} at: {start_time.strftime('%H:%M:%S')}")

    # Inductive setting (GraIL splits) if an inference graph is given: the
    # validation and test triples are scored in the context of the inference
    # graph alone (its walks and relational context), which shares no entity
    # with the training graph. Otherwise the transductive setting: one graph,
    # the training triples are the context.
    inductive = args.inference_paths is not None

    stageprint("Loading data and creating corruptions...")
    if inductive:
        if args.num_negatives > 0 and args.corruption != "all":
            raise ValueError("The inductive setting corrupts heads and tails "
                             "(--corruption all)")
        train_triples, val_triples, test_triples, inference_triples = (
            load_triple_tensors_inductive(
            args.train_paths, args.valid_paths, args.test_paths, args.inference_paths))
        train_paths, relcon, _ = du.load_unrolled_setup(args.train_paths,
                                                   args.path_setup)
        valid_paths, valid_relcon = du.load_inference_setup(
            args.inference_paths, args.path_setup)
        du.check_inference_context(valid_paths, valid_relcon, inference_triples,
                                   torch.cat((val_triples, test_triples)))

        # Creating filtration dictionaries and utilities for link prediction
        filtration_dict_train = triple_lib.make_relation_filter_dict_no_sp_tokens_train(
            train_triples)
        filtration_dict_inference = (
            triple_lib.make_relation_filter_dict_no_sp_tokens_inference(
            val_triples, test_triples, inference_triples))
        # Creating the head and tail filtered dict for the triple corruptors
        head_filter_dict_train, tail_filter_dict_train = (
            triple_lib.make_head_tail_dicts_train(train_triples))
        head_filter_dict_inference, tail_filter_dict_inference = (
            triple_lib.make_head_tail_dicts_inference(
            val_triples, test_triples, inference_triples))
        unique_entities_train = triple_lib.get_unique_entities_train(train_triples)
        unique_entities_inference = inference_triples[:, [0, 2]].unique()
        unique_entities = triple_lib.get_unique_entities_all(train_triples,
                                                             val_triples,
                                                             test_triples,
                                                             inference_triples)
    else:
        train_triples, val_triples, test_triples = load_triple_tensors(
            args.train_paths, args.valid_paths, args.test_paths)
        inference_triples = None
        train_paths, relcon, _ = du.load_unrolled_setup(args.train_paths,
                                                   args.path_setup)
        # One filter over train, validation and test, shared by all splits
        filtration_dict_inference = \
            triple_lib.make_relation_filter_dict_no_sp_tokens(
                train_triples, val_triples, test_triples)
        head_filter_dict_train, tail_filter_dict_train = \
            triple_lib.make_head_tail_dicts(
                train_triples, val_triples, test_triples)
        unique_entities = triple_lib.get_unique_entities(
            train_triples, val_triples, test_triples)
        unique_entities_train = unique_entities
    class_weights = triple_lib.get_class_weights_without_special_tokens(
        train_triples) if args.class_weigths else None

    # Creating the triple corruptors for head, tail, or both (merged) and
    # preserving the number of positive triples in case of H/T corruption
    tr_positives, va_positives, te_positives = \
        len(train_triples), len(val_triples), len(test_triples)
    # instantiated only if num_negatives > 0; in the transductive setting the
    # same corruptor serves training, validation and test
    triple_corruptor_train = triple_corruptor_valid = triple_corruptor_test = None
    if args.num_negatives > 0 and args.corruption in ["h", "t"]:
        if args.corruption == "h":
            triple_corruptor_train = CorruptHeadGenerator(
                filter_dict=head_filter_dict_train, entities=unique_entities_train)
        else:
            triple_corruptor_train = CorruptTailGenerator(
                filter_dict=tail_filter_dict_train,
                entities=unique_entities_train)
        triple_corruptor_valid = triple_corruptor_test = triple_corruptor_train
    elif args.num_negatives > 0:  # "all" applies and merge both corruptions
        triple_corruptor_train = CorruptLinkGenerator(
            head_filter_dict=head_filter_dict_train,
            tail_filter_dict=tail_filter_dict_train,
            entities=unique_entities_train)
        if inductive:  # negatives from the inference graph, seeded per split
            triple_corruptor_valid, triple_corruptor_test = [
                SeededLinkCorruptor(
                    head_filter_dict=head_filter_dict_inference,
                    tail_filter_dict=tail_filter_dict_inference,
                    entities=unique_entities_inference,
                    seed=(args.eval_seed, split))
                for split in (0, 1)]
        else:
            triple_corruptor_valid = triple_corruptor_test = \
                triple_corruptor_train
        # The number of positive is simply duplicated to measure LP results
        tr_positives, va_positives, te_positives = \
            [i * 2 for i in [tr_positives, va_positives, te_positives]]

    parallel = True
    negatives = [None, None, None]  # assume no negative dumpset is available
    if args.dump_dir is not None:
        print(f"Loading checkpointed negative triples from {args.dump_dir}")
        # The training pos-neg triples are loaded as a numpy memory map 
        negatives[0] = memmap_corrupted_triples_from_dir(
            os.path.join(args.dump_dir, "train"))
        # Validation and test triples are simply loaded as torch tensors
        negatives[1] = load_corrupted_triples_from_dir(
                        os.path.join(args.dump_dir, "valid"))
        negatives[2] = load_corrupted_triples_from_dir(
                        os.path.join(args.dump_dir, "test"))

    stageprint("Creating datasets and dataloaders...")
    train_set = TripleEntityMultiPathDataset(
        path_store=train_paths,
        relcontext_store=relcon,
        triple_store=train_triples,
        context_triple_store=train_triples,
        maximum_triple_paths=args.max_ppt,
        num_negatives=args.num_negatives,
        triple_corruptor=triple_corruptor_train,
        parallel=parallel,
        neg_triple_store=negatives[0],
    )
    # Using shared data structures for valid and test
    tokens_to_idxs = train_set.tokens_to_idxs
    path_store = train_set.relation_paths, \
                 train_set.entity_paths, \
                 train_set.path_index
    if inductive:
        eval_paths, eval_relcon, eval_context = \
            valid_paths, valid_relcon, inference_triples
    else:
        eval_paths, eval_relcon, eval_context = \
            path_store, relcon, train_triples
    valid_set = TripleEntityMultiPathDataset(
        path_store=eval_paths,
        relcontext_store=eval_relcon,
        triple_store=val_triples,
        context_triple_store=eval_context,
        maximum_triple_paths=args.max_ppt,
        tokens_to_idxs=tokens_to_idxs,
        num_negatives=args.val_num_negatives,
        triple_corruptor=triple_corruptor_valid,
        parallel=parallel,
        neg_triple_store=negatives[1],
    )
    test_set = TripleEntityMultiPathDataset(
        path_store=eval_paths,
        relcontext_store=eval_relcon,
        triple_store=test_triples,
        context_triple_store=eval_context,
        maximum_triple_paths=args.max_ppt,
        tokens_to_idxs=tokens_to_idxs,
        num_negatives=args.val_num_negatives,
        triple_corruptor=triple_corruptor_test,
        parallel=parallel,
        neg_triple_store=negatives[2],
    )
    if inductive:
        # the path sample is seeded by (seed, epoch, index); with epoch None
        # the seed is hash(None), which changes from one process to the next
        test_set.set_epoch(0)

    if args.num_negatives > 0 and args.dump_negatives:
        print(f"Dumping negative triples in {args.dump_dir}")
        for n, d in zip(["tr", "va", "te"], [train_set, valid_set, test_set]):
            negt = d.triplestore  # FIXME via the dump_negatives()
            torch.save(negt, os.path.join(args.dump_dir, f'{n}_negatives.pt'))

    print("Found {} samples in the dataset: Tr {}, Va {}, Te {}"
          .format(len(train_set) + len(valid_set) + len(test_set),
                  len(train_set), len(valid_set), len(test_set)))
    print(f"Vocabulary size (with special tokens): {len(tokens_to_idxs)}")

    if args.val_batch_size % (args.val_num_negatives + 1) != 0:
        positive_batches = args.val_batch_size // (args.val_num_negatives + 1)
        args.val_batch_size = positive_batches * (args.val_num_negatives + 1)
        # discarded_size = args.batch_size - fixed_bsize
        logger.warning(f"Clamping val/test batch size to:"
                       f" {args.val_batch_size}")
    if args.batch_size % (args.num_negatives + 1) != 0:
        positive_batches = args.batch_size // (args.num_negatives + 1)
        args.batch_size = positive_batches * (args.num_negatives + 1)
        # discarded_size = args.batch_size - fixed_bsize
        logger.warning(f"Clamping train batch size to: {args.batch_size}")

    # Creating the data loaders for each partition
    collate_fn = partial(collate_multipaths, padding_idx=tokens_to_idxs["PAD"])
    tr_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        collate_fn=collate_fn, shuffle=False,
        pin_memory=False, num_workers=args.num_workers,
        sampler=NegativeTripleSampler(tr_positives, args.num_negatives))
    va_dataloader = torch.utils.data.DataLoader(
        valid_set, batch_size=args.val_batch_size,
        collate_fn=collate_fn, shuffle=False,
        pin_memory=False, num_workers=args.num_workers,
        sampler=NegativeTripleSampler(va_positives, args.val_num_negatives))
    te_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=args.val_batch_size,
        collate_fn=collate_fn, shuffle=False,
        pin_memory=False, num_workers=args.num_workers,
        sampler=NegativeTripleSampler(te_positives, args.val_num_negatives))

    stageprint("Creating model and loading checkpoints")
    # getting number of unique entities
    num_entities = unique_entities.size()[0]

    injector_params = None
    #### ADDING EMBEDDING INJECTOR #################################
    if args.pretrained_path is not None:
        # loading pretraind embeddings and selecting nodes to use
        stageprint("Loading pretrained embeddings and selecting anchor nodes")
        embeddings = torch.load(args.pretrained_path)
        pretrained_number = floor(args.pretrained_ratio *
                                       embeddings.size()[0])
        permuted = torch.randperm(embeddings.size(0))
        node_indices = permuted[:pretrained_number]
        selected_embeddings = embeddings[node_indices]
        num_nodes = num_entities
        fusion = args.fusion_strategy
        injector_params = {
            "embeddings" : selected_embeddings,
            "node_indices" : node_indices,
            "num_nodes" : num_nodes,
            "projector" : True,
            "fusion" : fusion
        }
        pathe_dim = args.d_model
        pretrained_dim = selected_embeddings.size()[1]
        if pathe_dim < pretrained_dim:
            stageprint(f'Pathe dimension {str(pathe_dim)} increased to '
                       f'pre-trained dim {str(pretrained_dim)}.')
            args.d_model = pretrained_dim
    # This should assume model and basic parameters
    relcontext_graph = encode_relcontext_freqs(
        relcontext=relcon,
        num_entities=num_entities,
        num_relations=train_set.vocab_size - 2,
        offset=2,  # applied to both ents and rels
    )
    relcontext_graphs = None
    if inductive:  # the ids of the two graphs overlap (one id is shared)
        relcontext_graphs = (relcontext_graph, encode_relcontext_freqs(
            relcontext=valid_relcon,
            num_entities=inference_triples[:, [0, 2]].max().item() + 1,
            num_relations=train_set.vocab_size - 2,
            offset=2,
        ))
    bundle = partial(bundle_arguments, exclude=["vocab_size"],
                     args=namespace_to_dict(args))
    model = PathEModel(
        vocab_size=train_set.vocab_size,
        padding_idx=tokens_to_idxs["PAD"],
        relcontext_graph=relcontext_graph,
        injector_params=injector_params,
        **bundle(target_class=PathEModel),
    )
    # class_weights = triple_lib.get_class_weights(train_triples, tokens_to_idxs)
    pl_model = PathEModelWrapper(
        pathe_model=model,
        filtration_dict=filtration_dict_inference,  # FIXME
        class_weights=class_weights,  # for rel imbalance
        relcontext_graphs=relcontext_graphs,
        **namespace_to_dict(args),  # model hparameters
    )
    print(pl_model.model)  # keras-style model overview
    pl_model.model.to(torch.device(args.device))

    stageprint("Creating loggers, callbacks and setting up trainer")
    # Registering loggers based on the experiment name and version
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.expname,
        version=args.version)
    exp_dir = tb_logger.log_dir  # equivalent to args.exp_dir
    loggers = [tb_logger]
    if args.wandb_project is not None:  # use synchronised wandb logger
        wb_logger = WandbLogger(
            id=args.wandb_id,
            save_dir=exp_dir,
            name=args.expname,
            project=args.wandb_project,
            log_model="all",
            sync_tensorboard=True)
        # Logging gradients, topology, histogram
        wb_logger.watch(pl_model, log="all")
        wb_logger.log_hyperparams(args)
        loggers.append(wb_logger)

    # Creating callbacks for checkpointing and early stopping
    mode = "min" if args.monitor.endswith("loss") else "max"
    checkpoint_callbk = ModelCheckpoint(
        monitor=args.monitor, dirpath=args.checkpoint_dir, mode=mode,
        filename=model_name + f"-{{epoch}}-{{{args.monitor}:.2f}}",
        every_n_train_steps=args.chekpoint_ksteps)
    estopping_callbk = EarlyStopping(
        monitor=args.monitor, patience=args.patience, mode=mode)
    dataset_callbk = DatasetUpdater(  # datasets to seed per-epoch
        [train_set, valid_set] if args.train_paths else [train_set.dataset])

    # accumulated_batches = 5 # {2: 3, 4: 6}
    tr_limit, va_limit = (.1, .2) if args.debug else (1., 1.)
    accelerator = "gpu" if args.device == "cuda" else "cpu"

    trainer = Trainer(
        max_epochs=args.max_epochs, gradient_clip_val=1.0,
        accelerator=accelerator, devices=args.num_devices, num_nodes=1,
        limit_train_batches=tr_limit, limit_val_batches=va_limit,
        logger=loggers, log_every_n_steps=5,
        val_check_interval=args.val_check_interval,
        gradient_clip_algorithm='norm',  # CHANGED THIS
        accumulate_grad_batches=args.accumulate_gradient,
        callbacks=[estopping_callbk, checkpoint_callbk, dataset_callbk],
    )

    if args.cmd in ["train", "resume"]:
        # Train and resume are the same assuming their setup is consistent
        stageprint("Training-validating the model, be patient!")
        trainer.fit(pl_model, tr_dataloader, va_dataloader,
                    ckpt_path=args.checkpoint)  # load or None
        args.checkpoint = checkpoint_callbk.best_model_path
        print(f"Done. Best model saved in {args.checkpoint}")
        ttime = (datetime.datetime.now() - start_time).total_seconds() / 3600
        print(f"Training time: {round(ttime, 2)} hours")

    # stageprint("Evaluating the model on the validation set")
    # valid_dict = trainer.validate(
    #     dataloaders=va_dataloader,
    #     ckpt_path=checkpoint_callbk.best_model_path)[0]
    # del va_dataloader, valid_set  # free some memory
    # print("\nValidation results: {}".format(valid_dict))

    stageprint("Evaluating the model on the test set")
    test_dict = trainer.test(model=pl_model if args.cmd == "test" else None,
                             dataloaders=te_dataloader,
                             ckpt_path=args.checkpoint)[0]
    print("\nTesting results: {}".format(test_dict))

    # results_dict = {**valid_dict, **test_dict}
    # results_dict.update(namespace_to_dict(args))  # +hparams
    # fname = os.path.join(args.log_dir, "results_summary.csv")
    # write_csv(results_dict, fname)
    # results_raw = pd.read_csv(fname)

    # return results_dict, results_raw
