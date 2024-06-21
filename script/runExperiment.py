def main():
    import os
    import sys
    
    dev_path = os.path.dirname(__file__)
    src_path = os.path.join(dev_path, "..", 'src')
    sys.path.append(src_path)
    
    from cnn_loader import Experiment
    import argparse
    
    parser = argparse.ArgumentParser(description='Run an experiment created during an interactive session.')
    parser.add_argument('--experiment-path', type=str, help='Path to experiment pickle file.')
    
    args = parser.parse_args()
    experimentFile = args.experiment_path
    
    exp = Experiment(experimentFile)
    # Print experiment to its plot dir
    exp.run()

if __name__ == "__main__":
    main()