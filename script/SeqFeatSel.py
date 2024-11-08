# Based on VGLM, computes a sequential feature selction

def main():
    import os
    import sys
    
    dev_path = os.path.dirname(__file__)
    src_path = os.path.join(dev_path, "..", 'src')
    sys.path.append(src_path)
    
    from vgam_loader import RExperiment
    import argparse
    
    parser = argparse.ArgumentParser(description='Computes a sequential feature selection')
    parser.add_argument('--vars', type=str, nargs = '+', help='Variables which will be compared')
    parser.add_argument('--pre-selected', type=str, nargs = '*', help='Pre-selected variables')
    parser.add_argument('--baseline-experiment', type=str, help='Baseline experiment')
    
    baseexp = RExperiment(args.baseline_experiment)
    for var in args.vars:
        exp = baseexp.copy(features = args.pre_selected + [var])
        expfile = exp.files['experiment']
        exp.save.experimentfile()
        
    
    args = parser.parse_args()

if __name__ == "__main__":
    main()
