import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import the class you are testing
from xdetectioncore.ephys.spike_time_utils import SessionSpikes

@pytest.fixture
def sess_spike_obj_mock():
    # 1. Define the synthetic data we want returned
    # Let's create 2 units with 5 spikes total
    fake_spikes = np.array([1.0, 1.1, 2.0, 5.0, 10.0]) 
    fake_clusters = np.array([101, 101, 102, 102, 101]) # Unit 101 and 102
    
    # 2. Patch load_spikes WHERE IT IS USED (in spike_time_utils)
    # This prevents the code from ever entering io_utils.py
    patch_path = 'xdetectioncore.ephys.spike_time_utils.load_spikes'
    
    with patch(patch_path) as mocked_load:
        mocked_load.return_value = (fake_spikes, fake_clusters)
        
        # We also need to patch the Path checks inside SessionSpikes if they exist
        # or just provide valid-looking Path objects
        obj = SessionSpikes(
            spike_times_path=Path("dummy_times.npy"),
            spike_clusters_path=Path("dummy_clusters.npy"),
            sess_start_time=0.0,
            parent_dir=Path("."),
            fs=1.0,           # Set to 1.0 so spike_times / fs doesn't change values
            resample_fs=1.0
        )
        
        return obj