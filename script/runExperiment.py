def main():
    import os
    import sys
    
    dev_path = os.path.dirname(__file__)
    src_path = os.path.join(dev_path, "..", 'src')
    sys.path.append(src_path)
    
    from nn_loader import Experiment
    import argparse
    
    parser = argparse.ArgumentParser(description='Run an experiment created during an interactive session.')
    parser.add_argument('--experiment-path', type=str, help='Path to experiment pickle file.')
    parser.add_argument('--pre-computed', action='store_true', help='Whether the experiment has already been computed.')
    parser.add_argument('--fold', type=int, help='Fold number to run experiment on.')
    
    
    args = parser.parse_args()
    experimentFile = args.experiment_path
    
    exp = Experiment(experimentFile)
    # Print experiment to its plot dir
    exp.run(args.pre_computed, fold = args.fold)

if __name__ == "__main__":
    main()