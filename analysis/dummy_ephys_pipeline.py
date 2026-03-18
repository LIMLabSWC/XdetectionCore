import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import MagicMock

# Import your core ephys analysis classes and functions
from xdetectioncore.ephys.population_analysis_funcs import PopPCA
from xdetectioncore.ephys.aggregate_ephys_funcs import run_decoding

def generate_dummy_ephys_tensors():
    """
    Generates a dummy dictionary of aligned neural tensors.
    Simulates Rule 0 (ABCD) and Rule 1 (ABBA) with a latent 'rule' signal 
    emerging late in the trial.
    """
    n_neurons = 50
    n_times = 100 # e.g., 1 second at 10ms bins
    n_trials = 30
    times = np.linspace(0, 1.0, n_times)
    
    # Create conditions
    conds = ['ABCD_rule0', 'ABBA_rule1']
    resps_by_cond = {}
    
    for ci, cond in enumerate(conds):
        # Base activity (random Poisson-like noise)
        tensor = np.random.poisson(5, (n_neurons, n_times, n_trials)).astype(float)
        
        # Inject a 'Rule' signal into the last 20 neurons after t=0.5s
        if ci == 1: # Rule 1
            tensor[30:, 50:, :] += 15.0 
        else: # Rule 0
            tensor[30:, 50:, :] += 2.0
            
        resps_by_cond[cond] = tensor
        
    return resps_by_cond, times

def test_ephys_analysis():
    print("Generating dummy neural tensors [Neurons x Time x Trials]...")
    resps_by_cond, times = generate_dummy_ephys_tensors()
    
    # --- 1. Test Population PCA (State-Space) ---
    print("Testing PopPCA (Figure 4/5 logic)...")
    try:
        # Initialize your PopPCA class
        pca_obj = PopPCA(resps_by_cond)
        
        # Project onto first 3 PCs
        # This tests the internal trial-averaging and concatenation logic
        pca_obj.project_on_pc(resps_by_cond, n_components=3)
        
        print(f"SUCCESS: PCA completed. Explained variance: {pca_obj.pca.explained_variance_ratio_[:3]}")
    except Exception as e:
        print(f"FAILED: PopPCA failed with error: {e}")

    # --- 2. Test Neural Decoding (Figure 3/5 logic) ---
    print("\nTesting Decoding Pipeline...")
    try:
        # Mocking the AggregateSession structure your run_decoding expects
        # We test if the decoder can distinguish Rule 0 from Rule 1
        decode_results, cms = run_decoding(
            resps_by_cond, 
            cond_pairs=[('ABCD_rule0', 'ABBA_rule1')],
            n_shuffles=20,
            # In your code, decoding is often done on mean activity in a window
            # We'll simulate a single window decoding here
        )
        
        acc = decode_results['ABCD_rule0_vs_ABBA_rule1']['data_accuracy'].mean()
        print(f"SUCCESS: Decoding accuracy: {acc:.2f}")
    except Exception as e:
        print(f"FAILED: Decoding failed with error: {e}")

    print("\nEphys pipeline test complete.")

if __name__ == "__main__":
    test_ephys_analysis()