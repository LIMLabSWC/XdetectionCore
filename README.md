# XdetectionCore

[![PyPI version](https://img.shields.io/pypi/v/XdetectionCore.svg)](https://pypi.org/project/XdetectionCore/)
[![Python versions](https://img.shields.io/pypi/pyversions/XdetectionCore.svg)](https://pypi.org/project/XdetectionCore/)

**XdetectionCore** is the foundational data processing engine for the Akrami Lab (LIM Lab). It provides standardized utilities for electrophysiology (ephys) and behavioral analysis, specifically designed to bridge the gap between Windows workstations and Linux-based HPC clusters.

## � Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Modules](#core-modules)
- [Components](#components)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Unified Session Management**: Centralized `Session` class for managing ephys and behavior data
- **Spike Analysis**: Tools for spike time processing, PSTH computation, and neural decoding
- **Behavioral Analysis**: Utilities for sound events, lick detection, and pupil tracking
- **Cross-Platform Support**: Seamless path handling between Windows and Linux systems
- **Scalable Processing**: Integration with parallel processing via `joblib` for large datasets
- **Statistical Analysis**: Built-in filtering, z-scoring, and neural population analysis
- **Visualization**: Matplotlib-based plotting with custom styling and publication-ready figures

## �🚀 Installation

### From PyPI
The easiest way to install the stable version is via pip:
```bash
pip install XdetectionCore```

### From Source
For development or to access the latest version:
```bash
git clone https://github.com/Akrami-Lab/XdetectionCore.git
cd XdetectionCore
pip install -e .
```

### Requirements
- Python >= 3.8
- numpy < 2.0
- pandas >= 1.3
- matplotlib
- scipy
- tqdm
- joblib


## 🎯 Quick Start

### Basic Session Setup
```python
from xdetectioncore.session import Session

# Initialize a session
session = Session(
    sessname='my_experiment',
    ceph_dir='path/to/ceph/data',
    pkl_dir='path/to/pkl/data'
)

# Initialize spike data
session.init_spike_obj(
    spike_times_path='spike_times.npy',
    spike_cluster_path='spike_clusters.npy',
    start_time=0,
    parent_dir='path/to/data'
)

# Initialize sound events
session.init_sound_event_dict(
    sound_write_path='sound_writes.bin',
    format_kwargs={'sampling_rate': 30000}
)
```

### Loading and Processing Data
```python
from xdetectioncore.io_utils import load_sess_pkl, load_spikes
from xdetectioncore.paths import posix_from_win

# Load spike data
spike_times, spike_clusters = load_spikes(
    spike_times_path='spike_times.npy',
    spike_clusters_path='spike_clusters.npy'
)

# Cross-platform path handling
linux_path = posix_from_win('C:\\data\\recording', '/nfs/nhome/live')
```

## 📦 Core Modules

### `session.py`
Central session management class that coordinates ephys and behavioral data:
- Manages spike objects, events, and behavioral measurements
- Handles trial data (td_df) and inter-trial-interval (ITI) statistics
- Integrates decoders for neural population analysis
- Aggregates multi-session data via `load_aggregate_td_df()`

### `ephys/`
Electrophysiology analysis tools:
- **`spike_time_utils.py`**: `SessionSpikes` class for spike handling, raster generation, PSTH computation
- **`generate_synthetic_spikes.py`**: Synthetic neural data generation for validation and testing

### `components/`
Modular components for specific data types:
- **`events.py`**: `SoundEvent` class for sound stimulus representation and PSTH analysis
- **`licks.py`**: `SessionLicks` for lick behavior tracking and analysis
- **`pupil.py`**: `SessionPupil` for pupil tracking and statistics
- **`utils.py`**: Utility functions including z-scoring and normalization

### `decoding/`
Neural population decoding and classification:
- **`decoding_funcs.py`**: `Decoder` class for various decoding algorithms

### Utility Modules
- **`io_utils.py`**: File I/O operations (spike loading, sound binary reading, pickle handling)
- **`paths.py`**: Cross-platform path utilities and date extraction
- **`plotting.py`**: Publication-ready visualization and styling (`format_axis()`, `unique_legend()`, `apply_style()`)
- **`behaviour.py`**: Behavioral data processing and formatting
- **`stats.py`**: Statistical analysis functions

## 🔧 Components

### SoundEvent
Represents a sound stimulus event with associated spike responses:
```python
from xdetectioncore.components.events import SoundEvent

event = SoundEvent(idx=0, times=[1.0, 2.0, 3.0], lbl='stim_A')

# Compute PSTH (peri-stimulus time histogram)
event.get_psth(
    sess_spike_obj=session.spike_obj,
    window=[-0.5, 1.0],
    baseline_dur=0.25,
    zscore_flag=True
)

# Save visualization
event.save_plot_as_svg('figures/', suffix='trial_001')
```

### SessionSpikes
Core class for spike time processing:
```python
from xdetectioncore.ephys.spike_time_utils import SessionSpikes

spike_obj = SessionSpikes(
    spike_times_path='spike_times.npy',
    spike_cluster_path='spike_clusters.npy',
    start_time=0,
    parent_dir='data/'
)

# Get spike raster for specific time window
raster = spike_obj.get_spike_raster(start_time=0, end_time=10)
```

### SessionLicks
Track and analyze lick behavior:
```python
from xdetectioncore.components.licks import SessionLicks

licks = SessionLicks()
lick_times = licks.get_lick_times(lick_data_path='lick_times.csv')
```

### SessionPupil
Analyze pupil dynamics:
```python
from xdetectioncore.components.pupil import SessionPupil

pupil = SessionPupil()
# Process pupil tracking data
```

## 📊 Usage Examples

### Example 1: Single Session PSTH Analysis
```python
from xdetectioncore.session import Session
from xdetectioncore.components.events import SoundEvent
import numpy as np

# Create session
session = Session('exp_001', 'ceph_dir', 'pkl_dir')
session.init_spike_obj('spikes.npy', 'clusters.npy', 0, 'data/')

# Create sound event
sound_times = np.array([1.5, 5.2, 9.8])  # Second times
event = SoundEvent(idx=0, times=sound_times, lbl='tone_1khz')

# Compute and plot PSTH
event.get_psth(
    sess_spike_obj=session.spike_obj,
    window=[-0.5, 2.0],
    baseline_dur=0.25,
    zscore_flag=True,
    title='Tone Response'
)

event.save_plot_as_svg('output/', suffix='psth_analysis')
```

### Example 2: Multi-Session Data Aggregation
```python
from xdetectioncore.behaviour import load_aggregate_td_df
from pathlib import Path

# Load trial data from multiple sessions
td_df = load_aggregate_td_df(
    session_topology=session_info_df,
    home_dir=Path('/home/user/data')
)

# Filter and analyze
learning_window = td_df[td_df['trial_type'] == 'learning']
print(f"Average performance: {learning_window['correct'].mean():.3f}")
```

### Example 3: Cross-Platform Data Access
```python
from xdetectioncore.paths import posix_from_win

# Convert Windows path to Linux HPC path
win_path = 'C:\\data\\recordings\\exp_2024'
linux_path = posix_from_win(win_path, '/nfs/nhome/live/aonih')

# Now use linux_path for HPC analysis
print(linux_path)  # /nfs/nhome/live/aonih/data/recordings/exp_2024
```

## 📁 Project Structure

```
XdetectionCore/
├── xdetectioncore/
│   ├── __init__.py              # Package exports
│   ├── session.py               # Central Session class
│   ├── behaviour.py             # Behavioral data processing
│   ├── io_utils.py              # File I/O utilities
│   ├── paths.py                 # Cross-platform path handling
│   ├── plotting.py              # Visualization utilities
│   ├── stats.py                 # Statistical functions
│   ├── components/
│   │   ├── events.py            # SoundEvent class
│   │   ├── licks.py             # SessionLicks class
│   │   ├── pupil.py             # SessionPupil class
│   │   └── utils.py             # Component utilities
│   ├── decoding/
│   │   └── decoding_funcs.py    # Neural decoding algorithms
│   └── ephys/
│       ├── __init__.py
│       ├── spike_time_utils.py  # SessionSpikes class
│       └── generate_synthetic_spikes.py  # Data generation
├── pyproject.toml               # Project metadata
├── setup.py                     # Setup configuration
├── README.md                    # This file
└── LICENSE                      # License file
```

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the License specified in the [LICENSE](LICENSE) file.

## 🔗 Related Projects

- **Neo**: Neural data standards and I/O
- **Elephant**: Electrophysiology analysis toolkit
---

**Maintained by the Akrami Lab (LIM Lab)**  
For issues, questions, or suggestions, please open an issue on the GitHub repository or contact the lab.