import os
from utils.helpers import save_json
import argparse

from Benchmarks.RAILBENCH_Tracking import trackeval

# The tracking evaluation backend is adapted from TrackEval:
# https://github.com/JonathonLuiten/TrackEval

if __name__ == "__main__":

    # process command line arguments and set default configs for evaluation
    parser = argparse.ArgumentParser(description="Evaluate multi-object tracking")
    parser.add_argument("--project", default='railbench', help="Name of the project")
    args = parser.parse_args()

    # set configs for evaluation 
    default_eval_config = trackeval.Evaluator.get_default_eval_config()

    default_dataset_config = trackeval.datasets.RailBench.get_default_dataset_config()
    default_dataset_config['GT_FOLDER'] = os.path.join('data', args.project, 'gt')
    default_dataset_config['TRACKERS_FOLDER'] = os.path.join('data', args.project, 'trackers')
    default_dataset_config['CLASSES_TO_EVAL'] = ['person']
    default_dataset_config['SHOULD_CLASSES_COMBINE'] = len(default_dataset_config['CLASSES_TO_EVAL']) > 1  # Combine classes if more than 1 class to eval

    default_metrics_config = {'METRICS': ['HOTA']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if setting == 'project':  # Already processed
            continue
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    #------------------------------------------------------------------------------------------------

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.RailBench(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA()]
    _, _, output_summary = evaluator.evaluate(dataset_list, metrics_list)

    save_json(output_summary, os.path.join('data', args['project'], 'summary_results.json'))