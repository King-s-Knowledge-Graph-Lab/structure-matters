"""
A script to generate SLURM scripts for random/grid search.

Notes
-----
- Sub-batching is currently not supported here
- The grid can only be changed by editing this file
- To provide varying level of experimentations with light grids
"""
import os
import re
import random
import argparse
import itertools
import subprocess
from tqdm import tqdm
from glob import glob

from pathe.utils import create_dir, namespace_to_dict, stageprint

template_header = \
"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=pathe_{dataset}
#SBATCH --output=/scratch/users/{user}/pathe_{dataset}_{trial}_%A.out
#SBATCH --time {hours}:{minutes}:00
#SBATCH --ntasks-per-node={threads}
#SBATCH --gres=gpu
#SBATCH --partition=gpu
#SBATCH --mem={memory}
"""

template_exports = \
"""
# export CUDA_LAUNCH_BLOCKING=1
"""

template_command = \
"""
ulimit -n 4096

/users/{user}/miniconda3/envs/pathe/bin/python \\
  -u /users/{user}/projects/pathe/pathe/runner.py train pathe \\
  --expname {expname} \\
  --wandb_project {wandb_project} \\
  --log_dir /scratch/users/{user}/pathe-exps/{dataset} \\
  --train_paths /users/{user}/data/path-datasets/{dataset}/train/ \\
  --valid_paths /users/{user}/data/path-datasets/{dataset}/val/ \\
  --test_paths /users/{user}/data/path-datasets/{dataset}/test/ \\
  --path_setup {path_setup} --max_ppt {max_ppt} --batch_size {batch_size} \\
  --num_negatives {num_negatives} --val_num_negatives {val_num_negatives} \\
  --nhead {nhead} --num_encoder_layers {num_encoder_layers} \\
  --ent_aggregation {ent_aggregation} --num_agg_layers {num_agg_layers} \\
  --num_agg_heads {num_agg_heads} --context_heads {context_heads} \\
  --dim_feedforward {dim_feedforward} --embedding_dim {embedding_dim} \\
  --lp_loss_fn {lp_loss_fn} --accumulate_gradient {accumulate_gradient} \\
  --monitor valid_link_mrr --patience {patience} --dropout {dropout} \\
  --lrate {lrate} --weight_decay {weight_decay} --label_smoothing {label_smoothing} \\
  --device cuda --num_devices 1 --num_workers {num_workers} --seed {seed}
"""

GRIDS = {
    "large": {  # FIXME update with agg hps
        "lp_loss_fn": ['bce', 'ce', 'nssa'],
        "ent_aggregation": ["avg", "transformer", "recurrent"],
        "batch_size": [64, 128, 256, 512, 1024, 2048],
        "dropout": [.05, .1, .2, .3, .4],
        "lrate": [.01, .005, .001],
        "class_weigths": [True, False],  # REL prediction
        "max_ppt": [4, 8, 16, 32, 64, 128],
        "num_negatives": [16, 32, 64, 128, 256],
        "nhead": [2, 3, 4, 5, 6],
        "num_encoder_layers": [2, 3, 4],
        "dim_feedforward": [64, 128, 256],
        "embedding_dim": [64, 128, 256, 512],
        "accumulate_gradient": [1, 32, 64, 128],
        "weight_decay": [0.1, 0.01, 0.001, 0],  # CE
        "label_smoothing": [0.1, 0.01, 0.001, 0.],  # CE
    },
    "medium": {  # FIXME update with agg hps
        "lp_loss_fn": ['bce', 'ce', 'nssa'],
        "ent_aggregation": ["avg", "transformer", "recurrent"],
        "batch_size": [64, 128, 256, 512, 1024, 2048],
        "dropout": [.05, .1, .2],
        "lrate": [.01, .005, .001],
        "class_weigths": [True, False],  # REL prediction
        "max_ppt": [4, 8, 16, 32, 64, 128],
        "num_negatives": [16, 32, 64, 128, 256],
        "nhead": [2, 4, 6],
        "num_encoder_layers": [2, 3, 4],
        "dim_feedforward": [64, 128, 256],
        "embedding_dim": [64, 128, 256],
        "accumulate_gradient": [1, 32, 64, 128],
        "weight_decay": [0.01, 0.001, 0],  # CE
        "label_smoothing": [0.1, 0.01, 0.001, 0.],  # CE
    },
    "small": {
        "lp_loss_fn": ['ce'],
        "ent_aggregation": ["avg", "transformer", "recurrent"],
        "num_agg_layers": [1, 2],
        "num_agg_heads": [1, 2],
        "context_heads": [0, 1, 2],
        "batch_size": [1024],
        "dropout": [.1, .15],
        "lrate": [.001],
        "class_weigths": [True, False],  # REL prediction
        "max_ppt": [2, 4, 10],
        "num_negatives": [99],
        "nhead": [2, 4],
        "num_encoder_layers": [1, 2],
        "dim_feedforward": [256],
        "embedding_dim": [64, 128],
        "accumulate_gradient": [32],
        "weight_decay": [0],  # CE
        "label_smoothing": [0.1, .15],  # CE
    }
}

DEFAULTS = {
    "dropout": .1,
    "lrate": .001,
    "nhead": 2,
    "weight_decay": 0,
    "label_smoothing": 0,
    "class_weigths": False,
}


def create_expname_from_hpdict(hpdict: dict, defaults: dict = DEFAULTS):
    """
    Generates a single-string unique name for the given hparameter configuration
    in the format hparam_1:value_hparam2:value etc. using acronyms.

    Parameters
    ----------
    hpdict : dict
        The hyper-parameter dictionary, mapping names to their value.
    defaults : dict, optional
        A dictionary of default hparam values that will not be included in the
        generation of the trail name.

    Returns
    -------
    expname : str
        A name describing the hparam configuration as a single string.
    """
    expname = "_".join(["".join([a[:1] for a in k.split("_")]) + f":{v}"
        for k, v in hpdict.items() if k!="class_weigths"
        and v != defaults.get(k, None)])
    # Append store true arguments if present
    if hpdict["class_weigths"]:
        expname += "_cw"

    return expname


def check_hparam_config(hpdict: dict):
    """
    A simple check on the validity of a given configuration; used to limit the
    number of configuration for a search.
    """
    if hpdict["lp_loss_fn"] != "ce":
        if hpdict["weight_decay"] != 0 or hpdict["label_smoothing"] != 0:
            return False  # parametrisations are only valid for CE loss
        if hpdict["num_negatives"] >= hpdict["batch_size"]:
            return False
        if hpdict["batch_size"] < 256 and hpdict["accumulate_gradient"] < 32:
            return False
        if (hpdict["lp_loss_fn"] == 'nssa' and not (hpdict["batch_size"] // 2) >
                hpdict["num_negatives"]):
            return False

    # Context heads only valid with recurrent and single_query
    if hpdict["ent_aggregation"] in ["avg", "transformer"]:
        if hpdict["context_heads"] > 0:  # context should be 0
            return False
    # Aggregation layers are not neeeded for avg and laf
    if hpdict["ent_aggregation"] in ["avg", "laf"]:
        if hpdict["num_agg_layers"] > 1:
            return False
    # Aggregation heads are only needed for transformer and single_query
    if hpdict["ent_aggregation"] not in ["transformer", "single_query"]:
        if hpdict["num_agg_heads"] > 1:
            return False

    return True


def generate_full_grid(search_space: dict, fix_hparam: dict = {}):
    """
    Generate a full grid using the given search space of parameters, each of
    which is expected to be associated to a list of choices. This will simply
    compute all the combinations after passing a sanity check.

    Parameters
    ----------
    search_space : dict, optional
        A dictionary representation of the search space, by default GRID
    fix_hparam : dict, optional
        A mapping of hyper-parameters that will be fixed, by default {}

    Returns
    -------
    permutations_dicts : list
        A list of dictionaries, each providing a hparam configuration (trial).
    """
    # First step is to remove fixed hparams from the search space
    grid_pan = {k: [fix_hparam[k]] if fix_hparam.get(k, None) is not None
                else v for k, v in search_space.items()}
    # Creating the grid by combinations and sanity checks
    keys, values = zip(*grid_pan.items())
    permutations_dicts = [dict(zip(keys, v)) for v in tqdm(
                            itertools.product(*values),
                            desc="Generating trials")
                        if check_hparam_config(dict(zip(keys, v)))]

    return permutations_dicts


def generate_slurm_grid(args_dict):

    full_grid = generate_full_grid(
        GRIDS[args_dict["grid"]], fix_hparam=args_dict)
    print(f"Generated {len(full_grid)} valid trials")

    sel_trial_idxs = random.sample(
        range(0, len(full_grid)), args_dict["trials"])
    for sel_trial_idx in tqdm(sel_trial_idxs, desc="Writing trials"):
        trial_dict = full_grid[sel_trial_idx]
        trial_name = create_expname_from_hpdict(trial_dict)
        trial_name = args_dict["expname"] + "_" + trial_name \
            if args_dict["expname"] is not None else trial_name
        # Instantiating the template to create header and command
        slurm_header = template_header.format(**args_dict, trial=sel_trial_idx)
        slurm_exclude = "" if args_dict["exclude"] is None else \
            "#SBATCH --exclude {exclude}".format(args_dict["exclude"])

        trial_dict["val_num_negatives"] = trial_dict["num_negatives"] \
            if args_dict["val_num_negatives"] is None \
                else args_dict["val_num_negatives"]

        slurm_command = template_command.format(
            user=args_dict["user"], dataset=args_dict["dataset"],
            patience=args_dict["patience"], seed=args_dict["seed"],
            num_workers=args_dict["num_workers"],
            path_setup=args_dict["path_setup"],
            wandb_project=args_dict["wandb_project"],
            expname=trial_name+f"_{sel_trial_idx}", **trial_dict)

        slurm_bash = slurm_header + slurm_exclude \
                     + template_exports + slurm_command
        trial_fname = os.path.join(
            args_dict["script_dir"],
            f"{args_dict['dataset']}_{sel_trial_idx}.sh")

        with open(trial_fname, "w") as slurmsh:
            slurmsh.write(slurm_bash)


def schedule_slurm_runs(script_dir):

    all_scripts = glob(os.path.join(script_dir, "*.sh"))
    print(f"Found {len(all_scripts)} trials in {script_dir}")

    for script_file in all_scripts:
        # script_path = os.path.join(script_dir, script_file)
        print(f"Attempting scheduling of {script_file}")
        slurm_out = subprocess.run(["sbatch", script_file])
        print("\tscheduler response: %d" % slurm_out.returncode)


def find_cmd_param(param_name: str, script: str):

    search = re.search(param_name, script)
    if search is None:
        return None

    after_script = script[search.end() + 1:].lstrip()
    param_end = re.search(r" |\n", after_script).start()
    
    return after_script[:param_end].rstrip()


def find_outfname(target_expname: str, script_dir: str):

    script_paths = glob(os.path.join(script_dir, "*.sh"))
    print(f"Found {len(script_paths)} in {script_dir}")
    
    for script_path in script_paths:

        with open(script_path, "r") as slurmsh:
            cmd_str = slurmsh.read()
        expname = find_cmd_param("expname", cmd_str)
        if expname == target_expname:
            outfile = find_cmd_param("SBATCH --output=", cmd_str)
            outfile = outfile.split("/")[-1]
            print(outfile)
            return outfile

    print(f"No run found for expname: {target_expname}")


def main():
    """
    Create your own SLURM scripts and allocate them all.
    """
    parser = argparse.ArgumentParser(
        description='Main entry point for PathE SLURM Grid.')

    parser.add_argument('cmd', choices=['generate', 'run', 'search'],
                        help='A supported command to execute.')
    parser.add_argument('user', action='store', type=str,
                        help='A valid account number to use.')
    parser.add_argument('dataset', action='store', type=str,
                        help='The name of the dataset for the experiment')
    parser.add_argument('script_dir', action='store', type=str,
                        help='Path to the directionary to save/load scripts.')
    parser.add_argument('--trials', action='store', type=int, default=50,
                        help='The number of trials for random search.')
    parser.add_argument('--expname', type=str, action='store',
                        help='The base name to use for the experiment.')
    parser.add_argument('--exclude', type=str, action='store',
                        help='List of computational nodes to exclude.')
    parser.add_argument('--hours', type=int, action='store', default=12,
                        help='Number of hours for SLURM time-out.')
    parser.add_argument('--minutes', type=int, action='store', default=00,
                        help='Number of minutes for SLURM time-out.')
    parser.add_argument('--threads', type=int, action='store', default=64,
                        help='Number of threads to request.')
    parser.add_argument('--memory', type=int, action='store', default=64000,
                        help='Number of MBs to request for system memory.')
    parser.add_argument('--grid', choices=GRIDS.keys(), type=str, default="small",
                        help='The name of a supported GRID to use to search.')
    parser.add_argument('--wandb_project', action='store', type=str,
                        help='The name of the wandb project for logging.')
    parser.add_argument('--path_setup', type=str, default='20_20',
                        help='The path configuration to load if more present')

    # Parameters assumed fixed (no grid for them)
    parser.add_argument('--patience', action='store', type=int, default=10,
                        help='Number of validation epochs with no improvement.')
    parser.add_argument('--num_workers', action='store', type=int, default=0,
                        help='No. of threads for parallel data loading.')
    parser.add_argument('--val_num_negatives', type=int, default=None,
                        help='No. of negative/corrupt triples per positive '
                             'for validation and test sets.')

    # If provided, these parameters will override the grid
    parser.add_argument('--max_ppt', action='store', type=int, default=None,
                        help='Maximum number of paths per triple to sample.')
    parser.add_argument('--batch_size', action='store', type=int, default=None,
                        help='Batch size for training and validation loaders.')
    parser.add_argument('--accumulate_gradient', type=int, default=None,
                        help='No. of batches for gradient acc (1==off).')

    parser.add_argument('--ent_aggregation', action='store', default=None,
                        help='The name of the entity aggregation strategy.')
    parser.add_argument('--num_agg_heads', action='store', type=int, default=None,
                        help='Number of attention heads in the aggregator. '
                             'Only used when ent_aggregation is based on SAN.')
    parser.add_argument('--num_agg_layers', action='store', type=int, default=None,
                        help='Number of layers in the aggregator.')
    parser.add_argument('--context_heads', action='store', type=int, default=None,
                        help='Number of heads for entity contextualisation.')
    parser.add_argument('--num_encoder_layers', type=int, default=None,
                        help='Number of encoder layers in the arhictecture.')
    parser.add_argument('--embedding_dim', dest="d_model", type=int, default=None,
                        help='Size of the embedding vectors to learn.')
    parser.add_argument('--nhead', action='store', type=int, default=None,
                        help='Number of attention heads in the encoder.')
    parser.add_argument('--dropout', type=float, default=None,
                        help='The dropout percentage at training time.')
    parser.add_argument('--dim_feedforward', type=int, default=None,
                        help='Dimension of the feedforward network model.')
    parser.add_argument('--lp_loss_fn', action='store',
                        help='The name of the loss to use for link prediction.')
    parser.add_argument('--class_weigths', choices=[0, 1], type=int, default=None,
                        help='Whether to weight the loss with class frequencies.')
    parser.add_argument('--num_negatives', type=int, default=None,
                        help='Number of negative/corrupt triples per positive.')
    # The following only apply when the using a CE loss
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay for L2 regularisation.')
    parser.add_argument('--label_smoothing', type=float, default=None,
                        help='The amount of smoothing when computing the loss.')


    parser.add_argument('--seed', action='store', type=int, default=46,
                        help='Random seed for reproducibility of the exp.')    


    args = parser.parse_args()
    assert args.grid == "small", "Only small grid is supported for now."

    if args.class_weigths is not None:
        args.class_weigths = True if args.class_weigths == 1 else False
    # Tie related parameters together before the grid gen
    args.val_num_negatives = args.num_negatives \
        if args.val_num_negatives is None else args.val_num_negatives
    if args.wandb_project is None:  # default to datasetß
        args.wandb_project = args.dataset

    if args.cmd == "generate":
        stageprint(f"Generation of {args.trials} trials from GRID {args.grid}")
        create_dir(args.script_dir)
        args_dict = namespace_to_dict(args)
        generate_slurm_grid(args_dict)
    elif args.cmd == "run":
        stageprint(f"Scheduling {args.trials} trials via SLURM")
        schedule_slurm_runs(args.script_dir)
    else:  # assume search
        find_outfname(args.expname, args.script_dir)


if __name__ == "__main__":
    main()