def main():
    import os
    import sys
    
    dev_path = os.path.dirname(__file__)
    src_path = os.path.join(dev_path, "..", 'src')
    sys.path.append(src_path)
    
    import preprocessing as pp
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Baseline datasets')
    parser.add_argument('--dir', type=str, help='Input directory')
    parser.add_argument('--year', type=str, help='Year with which to work')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    inputDir = args.dir
    year = args.year
    outputDir = args.output
    
    pp.create_baseline_input(inputDir, year, outputDir)

if __name__ == "__main__":
    main()