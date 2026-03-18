import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import MagicMock

# Import the core analysis function and class from your provided files
from xdetectioncore.pupil.pupil_analysis_funcs import PupilCondAnalysis, run_pupil_cond_analysis

def generate_dummy_aligned_matrices():
    """
    Creates a dummy A_by_cond dictionary containing aligned DataFrames
    simulating the structure expected by the manuscript analysis.
    """
    n_sess = 3
    n_trials_per_sess = 20
    times = np.linspace(-1, 5, 180) # 1s baseline to 5s post-onset
    
    # Create MultiIndex: (Session ID, Mouse Name, Trial Number)
    sessions = [f'Sess_{i}' for i in range(n_sess)]
    mice = ['Mouse_A', 'Mouse_B', 'Mouse_C']
    
    iterables = [sessions, mice, range(n_trials_per_sess)]
    index = pd.MultiIndex.from_product(iterables, names=['sess', 'name', 'trial'])
    
    A_by_cond = {}
    
    # Generate Normal Condition (Base response)
    # Shape: [Total Trials x Timepoints]
    base_signal = 0.3 * np.exp(-(times - 1.2)**2 / 0.8) # Standard arousal peak
    noise = np.random.normal(0, 0.05, (len(index), len(times)))
    A_by_cond['normal'] = pd.DataFrame(base_signal + noise, index=index, columns=times)
    
    # Generate Deviant Condition (Standard peak + Surprise peak at 2.5s)
    surprise_peak = 0.6 * np.exp(-(times - 2.8)**2 / 1.0)
    noise_dev = np.random.normal(0, 0.05, (len(index), len(times)))
    A_by_cond['deviant'] = pd.DataFrame(base_signal + surprise_peak + noise_dev, index=index, columns=times)
    
    return A_by_cond, times

def run_test_with_matrices():
    print("Generating aligned dummy matrices...")
    A_by_cond, times = generate_dummy_aligned_matrices()
    
    # Initialize the analysis object
    # We use the class from your pupil_analysis_funcs.py
    analysis = PupilCondAnalysis()
    analysis.by_cond = A_by_cond
    analysis.times = times
    analysis.fs = 30 # Assuming 30Hz for this test
    
    # Setup output directory
    output_dir = Path("./test_results")
    output_dir.mkdir(exist_ok=True)
    
    print("Running PDR Cluster Analysis (Permutation Test)...")
    
    # This calls the exact function from your ch_2_pupil... script
    # It will perform the t-test at every time point and find significant clusters
    try:
        run_pupil_cond_analysis(
            analysis, 
            cond_comps=[('normal', 'deviant')], 
            run_shuff=True, 
            n_shuffles=50, # Low count for fast verification
            save_dir=output_dir,
            # line_kwargs are expected by your plotting code
            line_kwargs={'normal': {'c': 'gray'}, 'deviant': {'c': 'red'}}
        )
        print(f"\nSuccess! Check '{output_dir}' for:")
        print(" - ts_by_cond_normal_deviant.pdf (The PDR Plot)")
        print(" - Stats .tex files and cluster logs.")
        
    except Exception as e:
        print(f"\nAnalysis failed. Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test_with_matrices()