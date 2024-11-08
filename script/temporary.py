def main():
    import os
    import sys
    
    dev_path = os.path.dirname(__file__)
    src_path = os.path.join(dev_path, "..", 'src')
    sys.path.append(src_path)
    
    from baselines import CRPSarrClimatology
    import pickle
    
    with open("/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/Baselines/EmpiricalCDF.pkl", "rb") as f:
        res = pickle.load(f)
    with open("/scratch/alecler1/downscaling/JAX_NN/Experiment_72/test_set.pkl", "rb") as f:
        _,_,obs = pickle.load(f)
    CRPSClimatology = CRPSarrClimatology(obs, res, 5)
    with open("/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/Baselines/CRPSEmpiricalCDF.pkl", "wb") as f:
        pickle.dump(CRPSClimatology, f)

if __name__ == "__main__":
    main()