"""
Mine a path dataset: <data_root>/path_datasets/<dataset>/{train,val,test}/ for
the transductive builder, <data_root>/path_datasets/<dataset>/<version>/
{train,val,test,inference}/ for the inductive builder (GraIL splits).

The released path files were mined without a seed, so a rebuild does not
reproduce them exactly; the released files reproduce the reported numbers.
"""
import argparse

from . import statics
from ..paths import load_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('builder', choices=['transductive', 'inductive'])
    parser.add_argument('dataset', help='e.g. fb15k237, wn18rr, codex-large, '
                        'yago (transductive); fb15k237, wn18rr, nell (inductive)')
    parser.add_argument('--version', choices=['v1', 'v2', 'v3', 'v4'],
                        help='GraIL split (inductive builder only).')
    parser.add_argument('--num_paths', type=int, default=20,
                        help='Random walks per entity (20 for the released files).')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Steps per walk (10 for the released files).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed of the random walks.')
    parser.add_argument('--serial', action='store_true',
                        help='Use the single-process walk function. It starts '
                        'walks from other entities than the parallel one, with '
                        'which the released files were mined.')
    parser.add_argument('--data_root', default=None,
                        help='Default: data_root of configs/paths.yaml.')
    args = parser.parse_args()

    statics.DATA_DIR = args.data_root or str(load_paths()["data_root"])
    if args.builder == 'inductive':
        if args.version is None:
            parser.error('the inductive builder needs --version')
        from .inductive.kgloader import KgLoader
        from .inductive.pathdataset import PathDataset
        dataset = KgLoader(args.dataset, add_inverse=False, part=args.version)
    else:
        from .transductive.kgloader import KgLoader
        from .transductive.pathdataset import PathDataset
        dataset = KgLoader(args.dataset, add_inverse=False)
    builder = PathDataset(dataset=dataset, num_paths_per_entity=args.num_paths,
                          num_steps=args.num_steps, parallel=not args.serial,
                          seed=args.seed)
    print(f'Path dataset written to {builder.dataset_dir}')


if __name__ == '__main__':
    main()
