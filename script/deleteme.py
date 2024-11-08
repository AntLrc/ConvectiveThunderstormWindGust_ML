def main():
    import os
    import sys
    import jax
    
    dev_path = os.path.dirname(__file__)
    src_path = os.path.join(dev_path, "..", 'src')
    sys.path.append(src_path)
    
    from cnn_loader import Experiment
    import argparse
    
    parser = argparse.ArgumentParser(description='Updates test set.')
    parser.add_argument('--experiment-number', type=int, help='Experiment number')
    
    expdict = {1:8, 2:9, 3:10, 4:12, 5:13, 6:14, 7:15, 8:21, 9:23}
    
    args = parser.parse_args()
    experimentFile = f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/test/experiments/experiment_{expdict[args.experiment_number]}.pkl"
    
    exp = Experiment(experimentFile)
    # Print experiment to its plot dir
    if exp.nn_kwargs['spatial']['name'] == 'Conv_NN':
        if not isinstance(exp.nn_kwargs['spatial']['kwargs']['kernel_size'], tuple):
            exp.nn_kwargs['spatial']['kwargs']['kernel_size'] = (exp.nn_kwargs['spatial']['kwargs']['kernel_size'], exp.nn_kwargs['spatial']['kwargs']['kernel_size'])
            exp.NN = exp.NN_()
    exp.filter['storm_part']['test'] = (jax.random.key(5), 1.0)
    exp.save.experimentfile()
    exp.load_mean_std()
    exp.create_inputs(which_set = 'test')
    exp.run(preComputed = True)

if __name__ == "__main__":
    main()