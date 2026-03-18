import numpy as np
import joblib
from pathlib import Path

def generate_aligned_ephys_tensors(save_path="./data/test_ephys_data.joblib"):
    """
    Generates a dictionary of aligned PSTH tensors for reviewers.
    Structure: { 'cond_name': [Neurons x Time x Trials] }
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    n_neurons = 100
    n_times = 150  # 1.5 seconds at 10ms bins
    n_trials = 40
    
    conditions = ['ABCD_rule0', 'ABBA_rule1']
    resps_by_cond = {}
    
    for ci, cond in enumerate(conditions):
        # 1. Background activity (Poisson noise)
        tensor = np.random.poisson(2, (n_neurons, n_times, n_trials)).astype(float)
        
        # 2. Sensory identity (Pips 1-4)
        # Neurons 0-40 respond to pips at specific time offsets
        for pip_idx in range(4):
            onset = pip_idx * 25  # 250ms spacing
            neurons = slice(pip_idx*10, (pip_idx+1)*10)
            tensor[neurons, onset:onset+15, :] += 10.0
            
        # 3. Abstract Rule Signal (The "Nature" finding)
        # Neurons 60-80 encode Rule 1 vs Rule 0, starting after Pip 3 (idx 75)
        if 'rule1' in cond:
            tensor[60:80, 75:, :] += 8.0  # Stronger late response for Rule 1
        else:
            tensor[60:80, 75:, :] += 1.0  # Baseline for Rule 0
            
        resps_by_cond[cond] = tensor

    joblib.dump(resps_by_cond, save_path)
    print(f"Ephys tensor dictionary saved to {save_path}")

if __name__ == "__main__":
    generate_aligned_ephys_tensors()