import pytest
import numpy as np
import xdetectioncore

def test_numpy_version():
    """Ensure we aren't using the broken NumPy 2.0."""
    major_version = int(np.__version__.split('.')[0])
    assert major_version < 2, f"NumPy version {np.__version__} is incompatible with Matplotlib!"

def test_package_import():
    """Verify the package is installed and has a valid file path."""
    assert xdetectioncore.__file__ is not None
    assert "site-packages" in xdetectioncore.__file__ or "XdetectionCore" in xdetectioncore.__file__

def test_path_conversion():
    """Verify the Windows-to-POSIX engine works."""
    from xdetectioncore.paths import posix_from_win
    win_path = r"X:\Data\test"
    # Adjust the expected output based on your specific lab mapping
    posix_path = posix_from_win(win_path)
    assert posix_path.root == ''
    assert str(posix_path).startswith('Data'), f"Path conversion failed: {posix_path}"

def test_plotting_dependencies():
    """Verify that matplotlib and internal plotting modules load without crashing."""
    try:
        from xdetectioncore import plotting
        import matplotlib.pyplot as plt
    except Exception as e:
        pytest.fail(f"Plotting import failed (Check NumPy version!): {e}")

def test_session_init():
    """Verify the main Session class is exposed correctly."""
    from xdetectioncore import Session
    # We don't need to run it, just check if the class exists
    assert Session is not None
