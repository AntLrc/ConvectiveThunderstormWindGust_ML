def main():
    import os
    import sys
    
    dev_path = os.path.dirname(__file__)
    src_path = os.path.join(dev_path, "..", 'src')
    sys.path.append(src_path)
    
    import preprocessing as pp
    import argparse
    
    parser = argparse.ArgumentParser(description='Intersect dates of Pangu, labels, and Baseline datasets')
    parser.add_argument('--dir', type=str, help='Directory containing the datasets')
    parser.add_argument('--year', type=str, help='Year to intersect')
    
    args = parser.parse_args()
    nn_input_dir = args.dir
    year = args.year
    
    pp.intersect_dates(nn_input_dir, year)

if __name__ == "__main__":
    main()