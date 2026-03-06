import pytest
import numpy as np
from xdetectioncore.ephys.spike_time_utils import fast_instantaneous_rate

def test_integral():
    spikes = np.random.uniform(0.1, 0.9, 100)
    fs = 0.001
    rate = fast_instantaneous_rate(spikes, 0, 1.0, fs)
    
    total_spikes_calc = np.sum(rate) * fs
    # Should be close to 100
    assert np.isclose(total_spikes_calc, 100, rtol=0.05)
    print("Integral Test: Passed")

def test_causal_gaussian_logic():
    """
    Ensures that for a spike at time T, the firing rate 
    is exactly 0 at all times t < T.
    """
    
    t_start, t_stop = 0.0, 2.0
    sampling_period = 0.01  # 10ms bins
    spike_time = 1.0
    
    # Calculate rate
    rate = fast_instantaneous_rate(
        np.array([spike_time]), t_start, t_stop, sampling_period, 
        kernel_type='gaussian', kernel_width=0.04
    )
    
    spike_idx = int(spike_time / sampling_period)
    
    # Assert causality: No signal before the spike
    assert np.all(rate[:spike_idx] == 0), "Causality leak: Rate detected before spike!"
    # Assert response: Signal exists after the spike
    assert np.max(rate[spike_idx:]) > 0, "Filter failure: No rate detected after spike."

def test_rate_conservation():
    """
    Ensures the integral of the firing rate matches the spike count.
    """
    
    num_spikes = 100
    spikes = np.sort(np.random.uniform(0.2, 0.8, num_spikes))
    sampling_period = 0.001
    
    rate = fast_instantaneous_rate(spikes, 0, 1.0, sampling_period)
    
    # Area under the curve = sum(rate * dt)
    calculated_spikes = np.sum(rate) * sampling_period
    
    # Allow small margin for edge effects
    assert np.isclose(calculated_spikes, num_spikes, rtol=0.05)

def test_multiprocessing_caching(sess_spike_obj_mock):
    """
    Verifies that get_event_spikes populates the internal cache 
    and doesn't lose units during the parallel-to-serial merge.
    """
    event_times = [10.0, 20.0]
    event_name = "test_stim"
    window = [-0.5, 0.5]
    
    # Run first time
    sess_spike_obj_mock.get_event_spikes(event_times, event_name, window)
    
    # Check if dicts are populated
    key = f"{event_name}_{event_times[0]}"
    assert key in sess_spike_obj_mock.event_cluster_spike_times
    assert key in sess_spike_obj_mock.event_spike_matrices
    
    # Verify unit count matches
    matrix = sess_spike_obj_mock.event_spike_matrices[key]
    assert len(matrix) == len(sess_spike_obj_mock.units)
