import os
import argparse
import pickle
import resource
from datetime import datetime

from supporter.python.fsct_initial_param import initial_parameters
from supporter.python.separation_tools import DictToClass
from supporter.python.preprocessing import preprocessing
from supporter.python.semantic import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic settings
    parser.add_argument('--point-cloud', '-pc',default='', type=str, required=True,
                        help='Path to point cloud')
    parser.add_argument('--params', default='', type=str, required=True,
                        help='Path to pickled parameter file')
    parser.add_argument('--out_dir', '-odir', default='', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--out_fmt', default='las', type=str,
                        help='File type of output.')
    # Run or redo steps
    parser.add_argument('--step', default=3, type=int,
                        help='Process steps to run (i.e., Which process to run to)')
    parser.add_argument('--redo', default=None, type=int,
                        help='Which process to run to')
    # Hardware settings
    parser.add_argument('--batch-size', default=10, type=int,
                        help='Batch size for CUDA. Can try lowering the number when CUDA error raised.')
    parser.add_argument('--num-procs', default=10, type=int,
                        help='Number of CPU cores to use. Can try lowering the number when run out of RAM.')
    parser.add_argument('--keep-npy', action='store_true', type=bool,
                        help='Keeps .npy files used for segmentation after inference finished.')
    parser.add_argument('--wood-threshold', default=1, type=float,
                        help='A probability above which points are classified as wood.')
    parser.add_argument('--model', default=None, type=str,
                        help='Path to candidate model.')
    parser.add_argument('--verbose', action='store_true', help='Whether to print detailed info.')

    params = parser.parse_args()

    # Sanity checks
    if not os.path.isfile(params.point_cloud):
        if not os.path.isfile(params.params):
            raise Exception(f"! No point cloud at {params.point_cloud}.")

    start = datetime.now()

    if os.path.isfile(params.params):
        p_space = pickle.load(open(params.params, 'rb'))
        for k, v in p_space.__dict__.items():
            # override initial parameters
            if k == 'params': continue
            if k == 'step': continue
            if k == 'redo': continue
            if k == 'batch_size': continue
            setattr(params, k, v)
    else:
        for k, v in initial_parameters.items():
            if k == 'model' and params.model is not None:
                setattr(params, k, params.model)
            elif k == 'wood_threshold' and params.wood_threshold < 1:
                setattr(params, k, params.wood_threshold)
            else:
                setattr(params, k, v)
        params.steps_completed = {0:False, 1:False}

    if params.redo is not None:
        for k in params.steps_completed.keys():
            if k >= params.redo:
                params.steps_completed[k] = False

    if params.verbose:
        print('\n>> Parameter used:')
        for k, v in params.__dict__.items():
            if k == 'pc': v = '{}points'.format(len(v))
            if k == 'global_shift': v = v.values
            print('{:<35}{}'.format(k, v))

    if params.step >= 0 and not params.steps_completed[0]:
        params = preprocessing(params)
        params.steps_completed[0] = True
        pickle.dump(params, open(os.path.join(params.out_dir, f'{params.basename}.params.pkl'), 'wb'))

    if params.step >= 1 and not params.steps_completed[1]:
        params = semantic_segment(params)
        params.steps_completed[1] = True
        pickle.dump(params, open(os.path.join(params.out_dir, f'{params.basename}.params.pkl'), 'wb'))

    if params.verbose: print(f'runtime: {(datetime.now() - start).seconds}')
    if params.verbose: print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
