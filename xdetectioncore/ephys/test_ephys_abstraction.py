import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Importing your provided classes/functions
from xdetectioncore.ephys.population_analysis_funcs import PopPCA
from xdetectioncore.ephys.aggregate_ephys_funcs import run_decoding

def test_abstraction_pipeline(data_path="./data/test_ephys_data.joblib"):
    print("Loading aligned ephys tensors...")
    resps_by_cond = joblib.load(data_path)
    
    # --- TEST 1: State-Space Projection (PopPCA) ---
    print("\nRunning PopPCA Projection (Figure 4/5 logic)...")
    try:
        pca_mgr = PopPCA(resps_by_cond)
        # Project data into top 3 PCs
        pca_mgr.project_on_pc(resps_by_cond, n_components=3)
        
        # Verify the shape of projected data: [Condition] -> [PC x Time]
        for cond, proj in pca_mgr.projected_pca_ts_by_cond.items():
            print(f"  {cond} projected shape: {proj.shape}")
            
        print("SUCCESS: State-space trajectories generated.")
    except Exception as e:
        print(f"FAILED: PopPCA error: {e}")

    # --- TEST 2: Rule Decoding (Figure 5 logic) ---
    print("\nRunning Decoding Analysis...")
    try:
        # We test if the population can distinguish Rule 0 from Rule 1
        # Using a late window (bins 80 to 120) where rule info should exist
        late_resps = {k: v[:, 80:120, :].mean(axis=1) for k, v in resps_by_cond.items()}
        
        decode_results, _ = run_decoding(
            late_resps, 
            cond_pairs=[('ABCD_rule0', 'ABBA_rule1')],
            n_shuffles=50
        )
        
        res = decode_results['ABCD_rule0_vs_ABBA_rule1']
        acc = res['data_accuracy'].mean()
        shuff_acc = res['shuff_accuracy'].mean()
        
        print(f"  Rule Decoding Accuracy: {acc:.2f} (Shuffled: {shuff_acc:.2f})")
        
        if acc > 0.7:
            print("SUCCESS: Rule abstraction detected in late-window neural activity.")
        else:
            print("WARNING: Decoding accuracy lower than expected for synthetic effect.")
            
    except Exception as e:
        print(f"FAILED: Decoding error: {e}")

if __name__ == "__main__":
    test_abstraction_pipeline()