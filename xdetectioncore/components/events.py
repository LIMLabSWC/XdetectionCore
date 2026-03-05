import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING
from ..plotting import plot_psth
from ..ephys.spike_time_utils import get_event_psth
from .utils import zscore_by_unit

if TYPE_CHECKING:
    from ..ephys.spike_time_utils import SessionSpikes

class SoundEvent:
    def __init__(self, idx, times, lbl):
        self.idx = idx
        self.times = times
        self.lbl = lbl

        self.psth = None
        self.psth_plot = None
        self.synth_params = None

    def get_psth(self, sess_spike_obj: 'SessionSpikes', window, title='', redo_psth=False, redo_psth_plot=False,
                 baseline_dur=0.25, zscore_flag=False, iti_zscore=None, reorder_idxs=None, synth_data=None):

        if not self.psth or redo_psth:
            self.psth = get_event_psth(sess_spike_obj, self.idx, self.times, window, self.lbl,
                                       baseline_dur=baseline_dur, zscore_flag=zscore_flag, iti_zscore=None,
                                       synth_data=synth_data, synth_params=self.synth_params if synth_data else None, )
            if iti_zscore:
                self.psth = (self.psth[0],
                             zscore_by_unit(self.psth[1], unit_means=iti_zscore[0], unit_stds=iti_zscore[1]),
                             self.psth[2])
            if zscore_flag:
                means,stds = sess_spike_obj.unit_means
                self.psth = (self.psth[0], zscore_by_unit(self.psth[1], means, stds), self.psth[2])

        if not self.psth_plot or redo_psth or redo_psth_plot:
            self.psth_plot = plot_psth(self.psth[1], self.lbl, window, title=title)
            if zscore_flag:
                self.psth_plot[2].ax.set_ylabel('zscored firing rate (au)', rotation=270)

    def save_plot_as_svg(self, figdir: Path, suffix=''):
        filename = figdir / f'{self.lbl}_{suffix}.svg'
        if self.psth_plot:
            self.psth_plot[0].savefig(filename, format='svg')
            print(f"Plot saved as {filename}")
        else:
            print("No plot to save. Call 'plot_psth' first.")