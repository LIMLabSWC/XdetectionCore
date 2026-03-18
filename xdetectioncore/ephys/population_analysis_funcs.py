# matplotlib.use('TkAgg')
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt, lines as mlines
from scipy.linalg import orthogonal_procrustes
from scipy.stats import sem
from sklearn.cross_decomposition import CCA

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import euclidean

from sklearn.model_selection import cross_val_score
from os import cpu_count
from itertools import combinations

from ..plotting import plot_ts_var, format_axis, unique_legend

class PopPCA:

    def __init__(self, responses_by_cond: dict):
        self.scatter_plot = None
        self.proj_2d_plot = None
        self.proj_3d_plot = None
        self.eig_vals = None
        self.projected_pca_ts_by_cond = None
        self.pca_ts_plot = None
        self.Xa_trial_averaged_pca = None
        assert isinstance(list(responses_by_cond.values())[0], dict)
        self.responses_by_cond = responses_by_cond
        self.conds = list(responses_by_cond.keys())
        self.events = list(responses_by_cond[self.conds[0]].keys())
        self.event_concatenated_responses = self.get_event_concatenated_responses()
        self.get_eig_vals()

    @staticmethod
    def group_by_parts(strings, parts_to_match):
        """
        Group strings by specific parts (0-based index).

        parts_to_match: list of indexes, e.g. [1] for 2nd part, [0,2] for 1st+3rd.
        """
        groups = defaultdict(list)

        if parts_to_match is None:
            return strings
        for s in strings:
            parts = s.split("_")
            # Build the key using only selected parts
            try:
                key = tuple(parts[i] for i in parts_to_match)
                groups[key].append(s)
            except IndexError:
                pass  # ignore malformed strings

        return groups

    def get_event_concatenated_responses(self):
        event_concatenated_responses = np.hstack(
            [np.hstack(
                [e_responses for e_name, e_responses in cond_responses.items()])
                for cond_responses in self.responses_by_cond.values()])
        event_concatenated_responses = np.squeeze(event_concatenated_responses)
        event_concatenated_responses = event_concatenated_responses - np.nanmean(event_concatenated_responses, axis=1,
                                                                                 keepdims=True)
        return event_concatenated_responses

    def get_eig_vals(self):
        self.eig_vals = compute_eig_vals(self.event_concatenated_responses, plot_flag=True)
        self.eig_vals[2][1].set_xlim(0, 30)
        self.eig_vals[2][1].set_ylabel('PC component')
        self.eig_vals[2][1].set_xlabel('Proportion of variance explained')
        # self.eig_vals[2][0].show()

    def get_trial_averaged_pca(self, n_components=15, standardise=True):
        self.Xa_trial_averaged_pca = compute_trial_averaged_pca(self.event_concatenated_responses,
                                                                n_components=n_components, standardise=standardise)

    def get_projected_pca_ts(self, standardise=True):
        self.projected_pca_ts_by_cond = {cond: {event_name: project_pca(event_response, self.Xa_trial_averaged_pca,
                                                                        standardise=standardise)
                                                for event_name, event_response in cond_responses.items()}
                                         for cond, cond_responses in self.responses_by_cond.items()}

    def _get_proj_full(self, prop):
        """
        Return events (list[str]) and dict[event -> np.ndarray [n_pcs x T]].
        Accepts either dict-of-PCs or already-stacked ndarray per event.
        """
        by_event = self.projected_pca_ts_by_cond[prop]
        events = list(by_event.keys())
        if not events:
            raise ValueError("No events to plot.")
        proj_full = {}
        for e in events:
            data = by_event[e]
            if isinstance(data, dict):
                pcs = sorted(data.keys(), key=int)
                arr = np.vstack([data[k] for k in pcs])
            else:
                arr = data  # assume [n_pcs x T]
            proj_full[e] = arr
        return events, proj_full

    def _make_time_axis(self, event_window, T, x_ser_arg):
        """
        Build/validate time axis and a nearest-index helper robust to float equality.
        """
        import numpy as np
        if x_ser_arg is None:
            x_ser = np.round(np.linspace(event_window[0], event_window[1], T), 6)
        else:
            x_ser = np.asarray(x_ser_arg)

        def nearest_idx(t):
            return int(np.argmin(np.abs(x_ser - float(t))))

        return x_ser, nearest_idx

    def _prepare_proj_for_plot(
            self, prop, event_window, pca_comps_2plot, *, smoothing=0,
            align_trajs=False, align_method="orthogonal", align_pip_grouping=None,
            align_window=None, x_ser=None
    ):
        """
        Common pipeline:
          1) build full matrices, 2) make time axis, 3) (optional) align on ALL PCs,
          4) select PCs to plot, 5) (optional) smooth.
        Returns: proj_plot{event->[k x T]}, x_ser, nearest_idx
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter1d

        events, proj_full = self._get_proj_full(prop)
        T = next(iter(proj_full.values())).shape[-1]
        x_ser, nearest_idx = self._make_time_axis(event_window, T, x_ser)

        # Align on ALL PCs (before selecting/smoothing)
        if align_trajs:
            aw = align_window if align_window is not None else (0.0, event_window[1])
            proj_full = align_pca_groups(
                proj_full, events, x_ser,
                parts_to_match=align_pip_grouping,  # e.g. [0], [1], [0,2] or None
                name_grouper=self.group_by_parts,  # your helper
                align_method=align_method,  # 'orthogonal' or 'cca'
                align_window=aw
            )

        # Select PCs to plot
        proj_plot = {e: proj_full[e][pca_comps_2plot, :] for e in events}

        # Smooth after selection
        if smoothing and float(smoothing) > 0:
            for e in events:
                for i in range(proj_plot[e].shape[0]):
                    proj_plot[e][i, :] = gaussian_filter1d(proj_plot[e][i, :], smoothing)

        return events, proj_plot, x_ser, nearest_idx

    def plot_pca_ts(self, event_window, n_comp_toplot=5, plot_separately=False, fig_kwargs=None,
                    conds2plot=None, **kwargs):
        if conds2plot is None:
            conds2plot = self.conds
        if kwargs.get('events2plot', None) is None:
            events2plot = {cond: list(self.projected_pca_ts_by_cond[cond].keys()) for cond in conds2plot}
        else:
            events2plot = kwargs.get('events2plot')
        if kwargs.get('plot', None) is None:
            self.pca_ts_plot = plt.subplots(len(self.events) if plot_separately else 1, n_comp_toplot, squeeze=False,
                                            **(fig_kwargs if fig_kwargs is not None else {}))
        else:
            self.pca_ts_plot = kwargs.get('plot')

        axes = self.pca_ts_plot[1] if plot_separately else [self.pca_ts_plot[1][0]] * len(self.events)
        lss = kwargs.get('lss', ['-', '--', ':', '-.'])
        plt_cols = kwargs.get('plt_cols', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
        [[plot_pca_ts([projected_responses], [f'{cond} {event}'], event_window, n_components=n_comp_toplot,
                      plot=[self.pca_ts_plot[0], axes[ei]], plot_kwargs={'ls': lss[cond_i],  # 'c': plt_cols[ei],
                                                                         'label': f'{cond} {event}'})
          for ei, (event, projected_responses) in
          enumerate(zip(events2plot[cond], [self.projected_pca_ts_by_cond[cond][e] for e in events2plot[cond]]))]
         for cond_i, cond in enumerate(conds2plot)]
        [row_axes[0].set_ylabel('PC component') for row_axes in self.pca_ts_plot[1]]
        [row_axes[0].legend(loc='upper center', ncol=4) for row_axes in self.pca_ts_plot[1].T]
        # [ax.legend() for ax in self.pca_ts_plot[1]]
        [ax.set_xlabel('Time from stimulus onset (s)') for ax in self.pca_ts_plot[1][-1]]
        self.pca_ts_plot[0].show()

    def scatter_pca_points(self, prop: str, t_s: list, x_ser: np.ndarray, **kwargs):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import lines as mlines

        pca_comps_2plot = kwargs.get('pca_comps_2plot', [0, 1])

        # Reuse pipeline but with no smoothing / no alignment by default
        events, proj_plot, x_ser_eff, nearest_idx = self._prepare_proj_for_plot(
            prop, event_window=(x_ser.min(), x_ser.max()),
            pca_comps_2plot=pca_comps_2plot,
            smoothing=kwargs.get('smoothing', 0),
            align_trajs=kwargs.get('align_trajs', False),
            align_method=kwargs.get('align_method', 'orthogonal'),
            align_pip_grouping=kwargs.get('align_pip_grouping', None),
            align_window=kwargs.get('align_window', (x_ser.min(), x_ser.max())),
            x_ser=x_ser
        )

        if kwargs.get('plot', None) is None:
            fig, ax = plt.subplots(**kwargs.get('fig_kwargs', {}))
        else:
            fig, ax = kwargs.get('plot')

        t_idxs = [nearest_idx(t) for t in t_s]
        markers = kwargs.get('markers', list(mlines.Line2D.markers.keys())[:len(t_idxs)])

        for pi, e in enumerate(events):
            XY = proj_plot[e]  # [2 x T]
            for ti, t_idx in enumerate(t_idxs):
                ax.scatter(XY[0, t_idx], XY[1, t_idx], marker=markers[ti], c=f'C{pi}', label=e, s=50)

        unique_legend((fig, ax))
        format_axis(ax)
        ax.set_xlabel(f'PC {pca_comps_2plot[0]}')
        ax.set_ylabel(f'PC {pca_comps_2plot[1]}')

        fig.set_layout_engine('tight')
        self.scatter_plot = (fig, ax)
        return fig, ax

    def plot_3d_pca_ts_old(self, prop, event_window, **kwargs):

        pca_comps_2plot = kwargs.get('pca_comps_2plot', [0, 1, 2])
        minimal_axes = kwargs.get('minimal_axes', True)
        show_triad = kwargs.get('show_triad', True)
        triad_len_frac = kwargs.get('triad_len_frac', 0.12)
        triad_off_frac = kwargs.get('triad_off_frac', 0.04)
        smoothing = kwargs.get('smoothing', 3)

        t0_time_s = kwargs.get('t0_time', 0.0)
        t_end_s = kwargs.get('t_end', event_window[1])

        # Reuse common prep
        events, proj_plot, x_ser, nearest_idx = self._prepare_proj_for_plot(
            prop, event_window, pca_comps_2plot,
            smoothing=smoothing,
            align_trajs=kwargs.get('align_trajs', False),
            align_method=kwargs.get('align_method', 'orthogonal'),
            align_pip_grouping=kwargs.get('align_pip_grouping', None),
            align_window=kwargs.get('align_window', (t0_time_s, t_end_s)),
            x_ser=kwargs.get('x_ser', None),
        )

        t0_time_idx = nearest_idx(t0_time_s)
        t_end_idx = nearest_idx(t_end_s)

        # Set up plot
        if kwargs.get('plot', None) is None:
            fig, axes = plt.subplots(ncols=kwargs.get('n_cols', 1),
                                     subplot_kw={"projection": "3d"},
                                     figsize=(10, 10))
        else:
            fig, axes = kwargs.get('plot')
        if isinstance(axes, plt.Axes):
            axes = [axes]

        idxs_in_event = (x_ser >= t0_time_s) & (x_ser <= t_end_s)
        in_event_ls, out_event_ls = (kwargs.get('in_event_ls', ['-', '--']) + ['--', '--'])[:2]
        plot_out_event = kwargs.get('plot_out_event', True)

        # Plot per event
        for ei, e in enumerate(events):
            XYZ = proj_plot[e]  # [3 x T]
            for ax in axes:
                # in-event
                x_in, y_in, z_in = (arr.copy() for arr in XYZ)
                x_in[~idxs_in_event] = y_in[~idxs_in_event] = z_in[~idxs_in_event] = np.nan
                ax.plot(x_in, y_in, z_in, c=f'C{ei}', ls=in_event_ls, label=e)

                # out-of-event
                if plot_out_event:
                    x_out, y_out, z_out = (arr.copy() for arr in XYZ)
                    x_out[idxs_in_event] = y_out[idxs_in_event] = z_out[idxs_in_event] = np.nan
                    ax.plot(x_out, y_out, z_out, c=f'C{ei}', ls=out_event_ls)

                # start/end markers
                scatter_kwargs = dict(kwargs.get('scatter_kwargs', {}))
                markers = scatter_kwargs.pop('markers', ['v', 's'])
                size = scatter_kwargs.pop('size', 20)
                ax.scatter(XYZ[0, t0_time_idx], XYZ[1, t0_time_idx], XYZ[2, t0_time_idx], c=f'C{ei}', marker=markers[0],
                           s=size)
                ax.scatter(XYZ[0, t_end_idx], XYZ[1, t_end_idx], XYZ[2, t_end_idx], c=f'C{ei}', marker=markers[1],
                           s=size)

                # optional extra times
                t_pnts = kwargs.get('scatter_times')
                if t_pnts is not None:
                    if not isinstance(t_pnts, (list, tuple, np.ndarray)):
                        t_pnts = [t_pnts]
                    for tt in t_pnts:
                        tidx = nearest_idx(tt)
                        ax.scatter(XYZ[0, tidx], XYZ[1, tidx], XYZ[2, tidx], c=f'C{ei}', **scatter_kwargs)

                # axis labels
                ax.set_xlabel(f'PC{pca_comps_2plot[0]}')
                ax.set_ylabel(f'PC{pca_comps_2plot[1]}')
                ax.set_zlabel(f'PC{pca_comps_2plot[2]}')

        # view / legend
        axes[0].view_init(elev=22, azim=30)
        axes[0].legend()

        # minimal styling (unchanged)
        fig.patch.set_facecolor('white')
        for ax in axes:
            ax.set_facecolor('white')
            ax.grid(False)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                try:
                    axis.pane.fill = False
                    axis.pane.set_edgecolor('white')
                    axis.line.set_color((1, 1, 1, 0))
                except Exception:
                    pass
            for attr in ('w_xaxis', 'w_yaxis', 'w_zaxis'):
                if hasattr(ax, attr):
                    getattr(ax, attr).line.set_visible(False)
            ax.set_xticks([])
            ax.set_xticklabels([])
            if minimal_axes:
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
            if show_triad:
                xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
                dx, dy, dz = (xl[1] - xl[0]), (yl[1] - yl[0]), (zl[1] - zl[0])
                x0 = xl[0] + triad_off_frac * dx
                y0 = yl[0] + triad_off_frac * dy
                z0 = zl[0] + triad_off_frac * dz
                Lx, Ly, Lz = triad_len_frac * dx, triad_len_frac * dy, triad_len_frac * dz
                ax.quiver(x0, y0, z0, Lx, 0, 0, arrow_length_ratio=0.2, color='k')
                ax.quiver(x0, y0, z0, 0, Ly, 0, arrow_length_ratio=0.2, color='k')
                ax.quiver(x0, y0, z0, 0, 0, Lz, arrow_length_ratio=0.2, color='k')
                ax.text(x0 + Lx, y0, z0, 'PCA 1', ha='left', va='center')
                ax.text(x0, y0 + Ly, z0, 'PCA 2', ha='left', va='center')
                ax.text(x0, y0, z0 + Lz, 'PCA 3', ha='left', va='bottom')

        self.proj_3d_plot = (fig, axes)
        fig.tight_layout()
        fig.show()
        return fig, axes

    def plot_3d_pca_ts(self, prop, event_window, **kwargs):
        """
        3D PCA trajectories with optional grouping and mean±SEM visualization.

        Kwargs (new / aligned with 2D):
          plot_group: {'individual','mean_sem','both'} (default 'individual')
          plot_pip_grouping: list[int] or None (grouping used for plotting colors/means)
          sem_mode: {'ellipsoids', None} (default None)
          ellipsoid_every: int (default 10) - draw one ellipsoid every N timepoints
          ellipsoid_alpha: float (default 0.06)
          ellipsoid_scale: float (default 1.0) - multiply SEM radii
          sem_scale: float (default 1.0) - additional multiplier (kept for parity with 1D)

        Existing kwargs respected:
          pca_comps_2plot, smoothing, align_trajs, align_method, align_pip_grouping,
          align_window, t0_time, t_end, plot_out_event, in_event_ls,
          scatter_kwargs, scatter_times, minimal_axes, show_triad, etc.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import sem

        # --- config
        pca_comps_2plot = kwargs.get('pca_comps_2plot', [0, 1, 2])
        if len(pca_comps_2plot) != 3:
            raise ValueError("pca_comps_2plot must have exactly three components for 3D plotting.")
        pcx, pcy, pcz = pca_comps_2plot

        smoothing = kwargs.get('smoothing', 3)

        plot_group = kwargs.get('plot_group', 'individual')  # 'individual' | 'mean_sem' | 'both'
        indiv_alpha = float(kwargs.get('indiv_alpha', 0.5))

        sem_mode = kwargs.get('sem_mode', None)  # None | 'ellipsoids'
        ellipsoid_every = int(kwargs.get('ellipsoid_every', 10))
        ellipsoid_alpha = float(kwargs.get('ellipsoid_alpha', 0.06))
        ellipsoid_scale = float(kwargs.get('ellipsoid_scale', 1.0))
        sem_scale = float(kwargs.get('sem_scale', 1.0))

        t0_time_s = float(kwargs.get('t0_time', 0.0))
        t_end_s = float(kwargs.get('t_end', event_window[1]))

        minimal_axes = kwargs.get('minimal_axes', True)
        show_triad = kwargs.get('show_triad', True)
        triad_len_frac = kwargs.get('triad_len_frac', 0.12)
        triad_off_frac = kwargs.get('triad_off_frac', 0.04)

        in_event_ls, out_event_ls = (kwargs.get('in_event_ls', ['-', '--']) + ['--', '--'])[:2]
        plot_out_event = bool(kwargs.get('plot_out_event', True))
        lw = float(kwargs.get('lw', 2.0))

        # --- prep (reuse existing pipeline)
        events, proj_plot, x_ser, nearest_idx = self._prepare_proj_for_plot(
            prop, event_window, pca_comps_2plot,
            smoothing=smoothing,
            align_trajs=kwargs.get('align_trajs', False),
            align_method=kwargs.get('align_method', 'orthogonal'),
            align_pip_grouping=kwargs.get('align_pip_grouping', None),
            align_window=kwargs.get('align_window', (t0_time_s, t_end_s)),
            x_ser=kwargs.get('x_ser', None),
        )

        idxs_in_event = (x_ser >= t0_time_s) & (x_ser <= t_end_s)
        idxs_out_event = np.logical_or(x_ser < t0_time_s, x_ser > t_end_s)

        t0_time_idx = nearest_idx(t0_time_s)
        t_end_idx = nearest_idx(t_end_s)

        # --- plotting groups (mirror 2D behavior)
        parts_to_match = kwargs.get('align_pip_grouping', None)
        plots_group_parts2match = kwargs.get('plot_pip_grouping', parts_to_match)
        if plots_group_parts2match is None:
            plot_groups = self.group_by_parts(events, [0])
        else:
            if not isinstance(plots_group_parts2match, (list, tuple)):
                plots_group_parts2match = [plots_group_parts2match]
            plot_groups = self.group_by_parts(events, plots_group_parts2match)

        # --- axes
        if kwargs.get('plot', None) is None:
            fig = plt.figure(figsize=kwargs.get('figsize', (8, 7)))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = kwargs.get('plot')

        # --- helpers
        def plot_masked_line(ax, X, mask, color, ls, alpha=1.0, lw=1.5, label=None):
            x = X[0].copy()
            y = X[1].copy()
            z = X[2].copy()
            x[~mask] = np.nan
            y[~mask] = np.nan
            z[~mask] = np.nan
            ax.plot(x, y, z, c=color, ls=ls, alpha=alpha, lw=lw, label=label)

        def add_ellipsoid(ax, center, radii, color, alpha):
            # small param mesh; keep light for speed
            u = np.linspace(0, 2 * np.pi, 12)
            v = np.linspace(0, np.pi, 10)
            cu, su = np.cos(u), np.sin(u)
            cv, sv = np.cos(v), np.sin(v)
            # broadcast to (len(v), len(u))
            x = center[0] + radii[0] * np.outer(sv, cu)
            y = center[1] + radii[1] * np.outer(sv, su)
            z = center[2] + radii[2] * np.outer(cv, np.ones_like(u))
            ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=False)

        # --- individual traces
        if plot_group in ('individual', 'both'):
            for gi, (gkey, evs) in enumerate(plot_groups.items()):
                color = f"C{gi}"
                for e in evs:
                    XYZ = proj_plot[e]  # shape (3, T)
                    plot_masked_line(ax, XYZ, idxs_in_event, color, in_event_ls, alpha=indiv_alpha, lw=lw, label=None)
                    if plot_out_event:
                        plot_masked_line(ax, XYZ, idxs_out_event, color, out_event_ls, alpha=indiv_alpha, lw=lw,
                                         label=None)

        # --- group mean + SEM
        if plot_group in ('mean_sem', 'both'):
            for gi, (gkey, evs) in enumerate(plot_groups.items()):
                color = f"C{gi}"

                stack = np.stack([proj_plot[e] for e in evs], axis=0)  # (N, 3, T)
                meanXYZ = np.mean(stack, axis=0)  # (3, T)
                semXYZ = sem(stack, axis=0) * sem_scale  # (3, T)

                # mean lines
                plot_masked_line(ax, meanXYZ, idxs_in_event, color, in_event_ls, alpha=1.0, lw=lw, label=str(gkey))
                if plot_out_event:
                    plot_masked_line(ax, meanXYZ, idxs_out_event, color, out_event_ls, alpha=1.0, lw=lw, label=None)

                # start/end markers (mean)
                scatter_kwargs = dict(kwargs.get('scatter_kwargs', {}))
                markers = scatter_kwargs.pop('markers', ['v', 's'])
                size = scatter_kwargs.pop('size', 30)
                ax.scatter(meanXYZ[0, t0_time_idx], meanXYZ[1, t0_time_idx], meanXYZ[2, t0_time_idx],
                           c=color, marker=markers[0], s=size, edgecolor='k', linewidth=0.25, zorder=5)
                ax.scatter(meanXYZ[0, t_end_idx], meanXYZ[1, t_end_idx], meanXYZ[2, t_end_idx],
                           c=color, marker=markers[-1], s=size, edgecolor='k', linewidth=0.25, zorder=5)

                # SEM ellipsoids
                if sem_mode == 'ellipsoids':
                    # choose indices (typically only in-event; optional out-event too)
                    idxs = np.where(idxs_in_event)[0]
                    if idxs.size == 0:
                        idxs = np.arange(meanXYZ.shape[1])
                    idxs = idxs[::max(1, ellipsoid_every)]

                    for ti in idxs:
                        center = meanXYZ[:, ti]
                        radii = np.maximum(1e-12, semXYZ[:, ti]) * ellipsoid_scale
                        add_ellipsoid(ax, center, radii, color=color, alpha=ellipsoid_alpha)

                    if plot_out_event:
                        idxs2 = np.where(idxs_out_event)[0][::max(1, ellipsoid_every)]
                        for ti in idxs2:
                            center = meanXYZ[:, ti]
                            radii = np.maximum(1e-12, semXYZ[:, ti]) * ellipsoid_scale
                            add_ellipsoid(ax, center, radii, color=color, alpha=ellipsoid_alpha)

                # scatter_times (mean)
                scatter_ts = kwargs.get('scatter_times', [])
                if scatter_ts:
                    if not isinstance(scatter_ts, (list, tuple)):
                        scatter_ts = [scatter_ts]
                    for t in scatter_ts:
                        ti = nearest_idx(t)
                        ax.scatter(meanXYZ[0, ti], meanXYZ[1, ti], meanXYZ[2, ti],
                                   fc='lightgrey', marker='*', s=size, edgecolor='k', linewidth=0.25, zorder=6)

        # --- labels / legend
        ax.set_xlabel(f"PC{pcx}")
        ax.set_ylabel(f"PC{pcy}")
        ax.set_zlabel(f"PC{pcz}")
        unique_legend((fig, ax))
        ax.set_title(f"PCs: {pcx}, {pcy}, {pcz}")

        # minimal axes styling (optional)
        if minimal_axes:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

        fig.show()
        self.proj_3d_plot = (fig, ax)
        return fig, ax

    def plot_1d_pca_ts(self, prop, event_window, **kwargs):
        """
        1D PCA trajectories vs time with optional grouping and SEM shading.

        Kwargs:
        pca_comp: int (default 0) — which PC to plot
        plot_group: {'individual','mean_sem','both'} (default 'individual')
        smoothing: int (default 3)

        t0_time: float (default 0.0)
        t_end: float (default event_window[1])

        plot_out_event: bool (default True)
        in_event_ls: str or list (default ['-', '--'] -> uses first)
        out_event_ls: str or list (default ['-', '--'] -> uses second)

        indiv_alpha: float (default 0.5)

        sem_mode: {'band', 'bars', None} (default 'band')
        sem_alpha: float (default 0.15)
        sem_every: int (default 10) — for 'bars' mode
        sem_linewidth: float (default 0.8)
        sem_scale: float (default 1.0) — multiplier for SEM

        scatter_kwargs: dict, supports:
            markers: list like ['v','s'] for start/end
            size: int (default 25)

        scatter_times: list of times (floats) to mark on the mean trace
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import sem  # <- matches your 2D usage

        # --- config
        pca_comp = int(kwargs.get('pca_comp', 0))
        smoothing = kwargs.get('smoothing', 3)
        plot_group = kwargs.get('plot_group', 'individual')  # 'individual' | 'mean_sem' | 'both'
        indiv_alpha = float(kwargs.get('indiv_alpha', 0.5))

        t0_time_s = float(kwargs.get('t0_time', 0.0))
        t_end_s = float(kwargs.get('t_end', event_window[1]))

        sem_mode = kwargs.get('sem_mode', 'band')  # 'band' | 'bars' | None
        sem_alpha = float(kwargs.get('sem_alpha', 0.15))

        sem_scale = float(kwargs.get('sem_scale', 1.0))

        # --- shared prep (reuse your existing projector)
        # We call _prepare_proj_for_plot with a single component; it should return proj_plot[e] shaped (1, T)
        events, proj_plot, x_ser, nearest_idx = self._prepare_proj_for_plot(
            prop, event_window, [pca_comp],
            smoothing=smoothing,
            align_trajs=kwargs.get('align_trajs', False),
            align_method=kwargs.get('align_method', 'orthogonal'),
            align_pip_grouping=kwargs.get('align_pip_grouping', None),
            align_window=kwargs.get('align_window', (t0_time_s, t_end_s)),
            x_ser=kwargs.get('x_ser', None),
        )

        idxs_in_event = (x_ser >= t0_time_s) & (x_ser <= t_end_s)
        t0_time_idx = nearest_idx(t0_time_s)
        t_end_idx = nearest_idx(t_end_s)

        # --- groups (mirror your logic)
        parts_to_match = kwargs.get('align_pip_grouping',)

        plots_group_parts2match = kwargs.get('plot_pip_grouping', parts_to_match)
        if plots_group_parts2match is None:
            plot_groups = self.group_by_parts(events, [0])
        else:
            plot_groups = self.group_by_parts(events, plots_group_parts2match)

        # Create axes if needed
        ax = kwargs.get('ax', None)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        in_event_ls, out_event_ls = (kwargs.get('in_event_ls', ['-', '--']) + ['--', '--'])[:2]
        lw = kwargs.get('lw', 1.5)
        plot_out_event = bool(kwargs.get('plot_out_event', True))

        # --- individual traces
        if plot_group in ('individual', 'both'):
            for gi, (gkey, evs) in enumerate(plot_groups.items()):
                color = f'C{gi}'
                for e in evs:
                    Y = proj_plot[e][0]  # (T,)
                    yin = Y.copy()
                    yin[~idxs_in_event] = np.nan
                    ax.plot(x_ser, yin, c=color, ls=in_event_ls, alpha=indiv_alpha)

                    if plot_out_event:
                        yout = Y.copy()
                        yout[idxs_in_event] = np.nan
                        ax.plot(x_ser, yout, c=color, ls=out_event_ls, alpha=indiv_alpha)

        # --- group mean + SEM
        if plot_group in ('mean_sem', 'both'):
            for gi, (gkey, evs) in enumerate(plot_groups.items()):
                color = f'C{gi}'

                Ys = np.array([proj_plot[e][0] for e in evs])  # (N, T)
                meanY = np.mean(Ys, axis=0)
                semY = sem(Ys, axis=0) * sem_scale

                # plot mean trajectory (in-event + optional out-event)
                yin = meanY.copy()
                yin[~idxs_in_event] = np.nan
                ax.plot(x_ser, yin, c=color, ls=in_event_ls, lw=lw, label=str(gkey))

                if plot_out_event:
                    yout = meanY.copy()
                    yout[idxs_in_event] = np.nan
                    ax.plot(x_ser, yout, c=color, ls=out_event_ls, lw=lw)


                # SEM rendering
                if sem_mode == 'band':
                    lo = (meanY - semY).copy()
                    hi = (meanY + semY).copy()
                    lo[~idxs_in_event] = np.nan
                    hi[~idxs_in_event] = np.nan
                    ax.fill_between(x_ser, lo, hi, facecolor=color, alpha=sem_alpha, linewidth=0)

                    if plot_out_event:
                        lo2 = (meanY - semY).copy()
                        hi2 = (meanY + semY).copy()
                        lo2[idxs_in_event] = np.nan
                        hi2[idxs_in_event] = np.nan
                        ax.fill_between(x_ser, lo2, hi2, facecolor=color, alpha=sem_alpha, linewidth=0)


                for t in kwargs.get('scatter_times', []):
                    # safer than exact equality
                    ti = nearest_idx(t)
                    ax.axvline(x_ser[ti],c='k', ls='--',)

        # --- finalize
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'PC{pca_comp}')
        unique_legend((fig, ax))
        ax.set_title(f'PC: {pca_comp}')
        fig.tight_layout()

        self.pca_ts_plot = (fig, ax)
        fig.show()
        return fig, ax

    
    def plot_2d_pca_ts(self, prop, event_window, **kwargs):
        """
        2D PCA trajectories with optional grouping and 2D SEM ellipses.

        Kwargs:
          plot_group: {'individual','mean_sem','both'} (default 'individual')
          sem_mode: {'ellipses', None} — draw 2D SEM ellipses when 'ellipses'
          ellipse_every: int, default 10 — draw ellipse every N timepoints
          ellipse_alpha: float, default 0.25 — transparency of fill
          ellipse_edge_alpha: float, default 0.8 — transparency of edge
          ellipse_linewidth: float, default 0.8
          ellipse_scale: float, default 1.0 — multiplier for SEM radii
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        # --- config
        pca_comps_2plot = kwargs.get('pca_comps_2plot', [0, 1])
        if len(pca_comps_2plot) != 2:
            raise ValueError("pca_comps_2plot must have exactly two components for 2D plotting.")
        pcx, pcy = pca_comps_2plot

        smoothing = kwargs.get('smoothing', 3)
        plot_group = kwargs.get('plot_group', 'individual')  # 'individual' | 'mean_sem' | 'both'
        indiv_alpha = float(kwargs.get('indiv_alpha', 1))

        t0_time_s = kwargs.get('t0_time', 0.0)
        t_end_s = kwargs.get('t_end', event_window[1])

        # --- shared prep
        events, proj_plot, x_ser, nearest_idx = self._prepare_proj_for_plot(
            prop, event_window, pca_comps_2plot,
            smoothing=smoothing,
            align_trajs=kwargs.get('align_trajs', False),
            align_method=kwargs.get('align_method', 'orthogonal'),
            align_pip_grouping=kwargs.get('align_pip_grouping', None),
            align_window=kwargs.get('align_window', (t0_time_s, t_end_s)),
            x_ser=kwargs.get('x_ser', None),
        )

        T = next(iter(proj_plot.values())).shape[-1]
        idxs_in_event = (x_ser >= t0_time_s) & (x_ser <= t_end_s)
        t0_time_idx = nearest_idx(t0_time_s)
        t_end_idx = nearest_idx(t_end_s)

        # --- groups
        parts_to_match = kwargs.get('align_pip_grouping',)
        if kwargs.get('align_trajs', False):
            if parts_to_match is None:
                groups = {"all": events}
            else:
                if not isinstance(parts_to_match, (list, tuple)):
                    parts_to_match = [parts_to_match]
                groups = self.group_by_parts(events, parts_to_match)
        else:
            if parts_to_match is None:
                groups = {"all": events}
            else:
                groups = self.group_by_parts(events, plot_group)
            # groups = self.group_by_parts(events, parts_to_match)
            # groups = {"all": events}

        plots_group_parts2match = kwargs.get('plot_pip_grouping', parts_to_match)
        if plots_group_parts2match is None:
            plot_groups = self.group_by_parts(events, [0])
        else:
            plot_groups = self.group_by_parts(events, plots_group_parts2match)

        fig, ax = plt.subplots(figsize=(5, 4))
        in_event_ls, out_event_ls = (kwargs.get('in_event_ls', ['-', '--']) + ['--', '--'])[:2]
        plot_out_event = kwargs.get('plot_out_event', True)

        # --- individual lines
        if plot_group in ('individual', 'both'):
            for gi, (gkey, evs) in enumerate(plot_groups.items()):
                color = f'C{gi}'
                for e in evs:
                    XY = proj_plot[e]
                    xin, yin = XY[0].copy(), XY[1].copy()
                    xin[~idxs_in_event] = np.nan;
                    yin[~idxs_in_event] = np.nan
                    ax.plot(xin, yin, c=color, ls=in_event_ls, alpha=indiv_alpha,lw=1,label=str(gkey))

                    if plot_out_event:
                        xout, yout = XY[0].copy(), XY[1].copy()
                        xout[idxs_in_event] = np.nan;
                        yout[idxs_in_event] = np.nan
                        ax.plot(xout, yout, c=color, ls=out_event_ls, alpha=indiv_alpha)
                    # start/end markers
                    scatter_kwargs = dict(kwargs.get('scatter_kwargs', {}))
                    markers = scatter_kwargs.pop('markers', ['v', 's'])
                    size = scatter_kwargs.pop('size', 25)
                    ax.scatter(XY[0][t0_time_idx], XY[1][t0_time_idx],
                               c=color, marker=markers[0], s=size, zorder=3,lw=0.25,ec='k')
                    ax.scatter(XY[0][t_end_idx], XY[1][t_end_idx],
                               c=color, marker=markers[-1], s=size, zorder=3,lw=0.25,ec='k')

        # --- group mean and SEM ellipses
        if plot_group in ('mean_sem', 'both'):
            for gi, (gkey, evs) in enumerate(plot_groups.items()):
                color = f'C{gi}'
                meanX = np.mean([proj_plot[e][0] for e in evs],axis=0)
                semX = sem([proj_plot[e][0] for e in evs],axis=0)

                meanY = np.mean([proj_plot[e][1] for e in evs],axis=0)
                semY = sem([proj_plot[e][1] for e in evs],axis=0)

                # plot mean trajectory
                xin, yin = meanX.copy(), meanY.copy()
                xin[~idxs_in_event] = np.nan
                yin[~idxs_in_event] = np.nan
                ax.plot(xin, yin, c=color, ls=in_event_ls, lw=2, label=str(gkey))
                if plot_out_event:
                    idxs_post_event = x_ser > t_end_s
                    xout, yout = meanX.copy(), meanY.copy()
                    xout[~idxs_post_event] = np.nan
                    yout[~idxs_post_event] = np.nan
                    ax.plot(xout, yout, c=color, ls=out_event_ls, lw=2)

                # start/end markers
                scatter_kwargs = dict(kwargs.get('scatter_kwargs', {}))
                markers = scatter_kwargs.pop('markers', ['v', 's'])
                size = scatter_kwargs.pop('size', 25)
                ax.scatter(meanX[t0_time_idx], meanY[t0_time_idx],
                           fc=color, marker=markers[0], s=size, zorder=3,edgecolor='k',linewidth=0.25)
                ax.scatter(meanX[t_end_idx], meanY[t_end_idx],
                           fc=color, marker=markers[-1], s=size, zorder=3,edgecolor='k',linewidth=0.25)

                # --- SEM ellipses (true SEM in both PCs)

                sem_y_in = semY.copy()
                sem_y_in[~idxs_in_event] = np.nan
                # ax.fill_between(xin,yin-sem_y_in,yin+sem_y_in,fc=color,alpha=0.1)

                sem_x_in = semX.copy()
                sem_x_in[~idxs_in_event] = np.nan
                # ax.fill_betweenx(yin,xin-sem_x_in,xin+sem_x_in,fc=color,alpha=0.1)
                # plot_sem_tube(xin,yin,sem_x=sem_x_in,sem_y=sem_y_in,ax=ax,color=color)

                for x, y, sx, sy in zip(xin, yin, sem_x_in, sem_y_in):
                    e = Ellipse((x, y), width=2 * sx, height=2 * sy, fc=color, alpha=0.05,edgecolor='none')
                    ax.add_patch(e)

                if plot_out_event:
                    idxs_post_event = x_ser > t_end_s
                    xout, yout = meanX.copy(), meanY.copy()
                    xout[~idxs_post_event] = np.nan
                    yout[~idxs_post_event] = np.nan
                    sem_y_out = semY.copy()
                    sem_y_out[~idxs_post_event] = np.nan

                    sem_x_out = semX.copy()
                    sem_x_out[~idxs_post_event] = np.nan

                    for x, y, sx, sy in zip(xout, yout, sem_x_out, sem_y_out):
                        e = Ellipse((x, y), width=2 * sx, height=2 * sy, fc=color, alpha=0.05, edgecolor='none')
                        ax.add_patch(e)

                # --- Scatter extra points
                scatter_ts = kwargs.get('scatter_times',[])
                for t in scatter_ts:
                    scatter_t_idx = np.where(x_ser==t)[0][0]
                    x,y = meanX[scatter_t_idx], meanY[scatter_t_idx]
                    ax.scatter(x, y, fc='lightgrey', marker='*', s=size, zorder=3, edgecolor='k',linewidth=0.25)


        # --- finalize
        ax.set_xlabel(f'PC{pcx}')
        ax.set_ylabel(f'PC{pcy}')
        unique_legend((fig, ax))
        ax.set_title(f'PC: {pca_comps_2plot}')
        fig.tight_layout()
        self.proj_2d_plot = (fig, ax)
        fig.show()
        return fig, ax

    def pcspace_distances(
            self,
            prop: str,
            event_window: tuple,
            times,
            *,
            n_pcs: int | None = None,
            x_ser: np.ndarray | None = None,
            align_trajs: bool = False,
            align_method: str = "orthogonal",
            align_pip_grouping=None,
            global_align_window: tuple | None = None,
            reference: str | None = None,
            metric: str = "cosine",  # <-- NEW
            return_squareform: bool = True
    ):
        """
        Compute distances between events in PCA space for given timepoints/windows.
        Supports metrics: 'euclidean', 'cosine', 'mahalanobis'.
        """

        import numpy as np
        from scipy.spatial.distance import pdist, squareform, cdist

        # --------------------------
        # BUILD DATA
        # --------------------------
        events, proj_full = self._get_proj_full(prop)
        T = next(iter(proj_full.values())).shape[-1]
        x_ser_eff, nearest_idx = self._make_time_axis(event_window, T, x_ser)

        # restrict PCs
        if n_pcs is not None:
            proj_full = {e: mat[:n_pcs, :] for e, mat in proj_full.items()}

        # normalize timespecs → [(t0,t1)]
        if not isinstance(times, (list, tuple)):
            times = [times]

        specs = []
        rep_times = []
        for item in times:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                t0, t1 = float(item[0]), float(item[1])
            else:
                t = float(item)
                if global_align_window:
                    t0, t1 = global_align_window
                else:
                    t0 = t1 = t
            if t1 < t0:
                t0, t1 = t1, t0
            specs.append((t0, t1))
            rep_times.append((t0 + t1) / 2)

        def idx_window(t0, t1):
            i0 = nearest_idx(t0)
            i1 = nearest_idx(t1)
            return (i0, i1) if i0 <= i1 else (i1, i0)

        # --------------------------
        # COMPUTE MEAN VECTOR PER EVENT PER WINDOW
        # --------------------------
        per_spec_vectors = []

        for (t0, t1) in specs:
            i0, i1 = idx_window(t0, t1)

            # align per window if requested
            if align_trajs:
                aligned = align_pca_groups(
                    proj_full, events, x_ser_eff,
                    parts_to_match=align_pip_grouping,
                    name_grouper=self.group_by_parts,
                    align_method=align_method,
                    align_window=(t0, t1)
                )
                use = aligned
            else:
                use = proj_full

            rowlist = []
            for e in events:
                seg = use[e][:, i0:i1 + 1]
                if seg.ndim == 1:
                    meanv = seg
                else:
                    meanv = np.mean(seg, axis=1)
                rowlist.append(meanv)
            per_spec_vectors.append(np.vstack(rowlist))  # shape (N,D)

        # --------------------------
        # MAHALANOBIS SUPPORT
        # --------------------------
        if metric == "mahalanobis":
            # covariance of PCs across events (D x D)
            allX = np.vstack(per_spec_vectors)  # (K*N, D)
            cov = np.cov(allX, rowvar=False)
            cov += np.eye(cov.shape[0]) * 1e-9
            invcov = np.linalg.inv(cov)

            # use cdist with mahalanobis
            def pairwise(X):
                return cdist(X, X, metric="mahalanobis", VI=invcov)

            def to_reference(X, r):
                ref = X[[r], :]
                return cdist(np.delete(X, r, 0), ref, metric="mahalanobis", VI=invcov)[:, 0]

        # --------------------------
        # COSINE SUPPORT
        # --------------------------
        elif metric == "cosine":
            def pairwise(X):
                return squareform(pdist(X, metric="cosine"))

            def to_reference(X, r):
                ref = X[[r], :]
                return cdist(np.delete(X, r, 0), ref, metric="cosine")[:, 0]

        # --------------------------
        # EUCLIDEAN DEFAULT
        # --------------------------
        else:
            def pairwise(X):
                return squareform(pdist(X, metric="euclidean"))

            def to_reference(X, r):
                ref = X[[r], :]
                return cdist(np.delete(X, r, 0), ref, metric="euclidean")[:, 0]

        # --------------------------
        # BUILD OUTPUTS
        # --------------------------
        if reference is not None:
            ref_idx = events.index(reference)
            outs = []
            for X in per_spec_vectors:
                outs.append(to_reference(X, ref_idx))
            D = np.stack(outs)
        else:
            mats = []
            for X in per_spec_vectors:
                mats.append(pairwise(X))
            D = np.stack(mats)

        return {
            "events": events,
            "specs": specs,
            "times": np.array(rep_times),
            "distances": D
        }

    def plot_pcspace_distances(
            self,
            prop: str,
            event_window: tuple,
            times,
            *,
            n_pcs: int | None = None,
            x_ser: np.ndarray | None = None,
            align_trajs: bool = False,
            align_method: str = "orthogonal",
            align_pip_grouping=None,
            global_align_window: tuple | None = None,
            reference: str | None = None,
            reduce_when_pairwise: str = "per_event_mean",  # 'per_event_mean' | 'global_mean'
            labels_map: dict | None = None,  # optional renaming of event labels
            figsize: tuple = (7, 4),
            ax=None,
            legend: bool = True,
            title: str | None = None,
            lw: float = 2.0,
            alpha: float = 0.95
    ):
        """
        Plot Euclidean distances in PC space across specified time points/windows.

        Uses `pcspace_distances(...)` under the hood. If `reference` is provided,
        plots one line per other event (distance to reference). Otherwise, reduces
        the full pairwise matrix over events either to:
          - 'per_event_mean': one line per event (mean distance to others)
          - 'global_mean': a single line (mean of all pairwise distances)

        Parameters
        ----------
        prop, event_window, times, n_pcs, x_ser, align_trajs, align_method,
        align_pip_grouping, global_align_window, reference : see pcspace_distances.
        reduce_when_pairwise : str
            Only used when `reference is None`. Choose 'per_event_mean' or 'global_mean'.
        labels_map : dict
            Optional mapping {original_event_name: display_name}.
        figsize : tuple
            Figure size if `ax` is None.
        ax : matplotlib axis or None
            If provided, draw on it; otherwise create a new fig/ax.
        legend : bool
            Show legend.
        title : str or None
            Custom plot title; if None, a sensible default is used.
        lw : float
            Line width.
        alpha : float
            Line alpha.

        Returns
        -------
        fig, ax
        """
        import numpy as np
        import matplotlib.pyplot as plt

        res = self.pcspace_distances(
            prop=prop,
            event_window=event_window,
            times=times,
            n_pcs=n_pcs,
            x_ser=x_ser,
            align_trajs=align_trajs,
            align_method=align_method,
            align_pip_grouping=align_pip_grouping,
            global_align_window=global_align_window,
            reference=reference,
            return_squareform=True  # easier for reductions; we handle our own reductions
        )

        events = res["events"]
        t = np.asarray(res["times"])
        D = res["distances"]  # shape: (K, N, N) if no reference; or (K, N-1) if reference is set

        # Prepare axes
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            created_fig = True
        else:
            fig = ax.figure

        # Helper for labels
        def label_of(e):
            if labels_map and e in labels_map:
                return labels_map[e]
            return e

        if reference is not None:
            # Distances to a reference -> lines for each 'other' event
            if reference not in events:
                raise ValueError(f"reference '{reference}' not in events {events}")
            ref_idx = events.index(reference)

            # Build names excluding reference in the same order used by pcspace_distances
            others = [events[i] for i in range(len(events)) if i != ref_idx]
            Y = D  # (K, N-1)

            for j, e in enumerate(others):
                ax.plot(t, Y[:, j], label=label_of(e), lw=lw, alpha=alpha)

            ttl = title or f"PC-space distance to {label_of(reference)}"
        else:
            # Full pairwise matrices at each time → reduce to lines
            K, N, _ = D.shape
            if reduce_when_pairwise not in {"per_event_mean", "global_mean"}:
                raise ValueError("reduce_when_pairwise must be 'per_event_mean' or 'global_mean'")

            if reduce_when_pairwise == "global_mean":
                # mean of upper triangle (excluding diag) per time
                # each D[k] is (N,N)
                vals = []
                for k in range(K):
                    mat = np.asarray(D[k])
                    iu = np.triu_indices(N, k=1)
                    vals.append(np.mean(mat[iu]) if iu[0].size > 0 else np.nan)
                ax.plot(t, np.asarray(vals), label="Global mean pairwise distance", lw=lw, alpha=alpha)
                ttl = title or "PC-space global mean pairwise distance"
            else:
                # per_event_mean: mean distance from each event to all others
                # one line per event
                for i, e in enumerate(events):
                    means = []
                    for k in range(K):
                        mat = np.asarray(D[k])
                        if N > 1:
                            # mean of row i excluding diagonal
                            m = (np.sum(mat[i, :]) - mat[i, i]) / (N - 1)
                        else:
                            m = np.nan
                        means.append(m)
                    ax.plot(t, np.asarray(means), label=label_of(e), lw=lw, alpha=alpha)
                ttl = title or "PC-space mean distance to others (per event)"

        # Cosmetics
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Euclidean distance in PC space")
        ax.set_title(ttl)
        if legend:
            ax.legend(frameon=False, fontsize=9, ncol=1)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        return fig, ax

    def scatter_2d_pca(
            self,
            prop,
            event_window,
            pca_comps_2plot=(0, 1),
            labels=None,
            colors=None,
            title="2D PCA Scatter",
            xlabel=None,
            ylabel=None,
            figsize=(8, 6),
            show_sem=False,
            sem_scale=1.96,
            marker_size=50,
            alpha=0.6,
            ax=None,
            **kwargs
    ):
        """
        Scatter plot of two principal components with optional SEM visualization.
        
        Parameters
        ----------
        prop : str
            Condition/property key for projected_pca_ts_by_cond
        event_window : tuple
            (t_start, t_end) time window
        pc1 : int, default=0
            Index of first principal component to plot
        pc2 : int, default=1
            Index of second principal component to plot
        labels : array-like, optional
            Labels for each point (for legend)
        colors : array-like, optional
            Colors for each point
        title : str, default="2D PCA Scatter"
            Plot title
        xlabel : str, optional
            Label for x-axis (default: f"PC{pc1+1}")
        ylabel : str, optional
            Label for y-axis (default: f"PC{pc2+1}")
        figsize : tuple, default=(8, 6)
            Figure size
        show_sem : bool, default=False
            Whether to show SEM around points (error bars at mean)
        sem_scale : float, default=1.96
            Scale factor for SEM (1.96 for 95% CI, 1.0 for SEM)
        marker_size : int, default=50
            Size of scatter markers
        alpha : float, default=0.6
            Transparency of markers
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on
        **kwargs : dict
            Additional options passed to _prepare_proj_for_plot (smoothing, align_trajs, etc.)
        
        Returns
        -------
        fig, ax : matplotlib figure and axes objects
        """
        
        pc1, pc2 = pca_comps_2plot
        
        # Reuse common prep pipeline
        events, proj_plot, x_ser, nearest_idx = self._prepare_proj_for_plot(
            prop, event_window, pca_comps_2plot,
            smoothing=kwargs.get('smoothing', 0),
            align_trajs=kwargs.get('align_trajs', False),
            align_method=kwargs.get('align_method', 'orthogonal'),
            align_pip_grouping=kwargs.get('align_pip_grouping', None),
            align_window=(event_window[0], event_window[1]),
            x_ser=kwargs.get('x_ser', None),
        )


        # Create axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        x_mask = np.logical_and(x_ser>=event_window[0], x_ser<=event_window[1])

        # Set default labels
        if xlabel is None:
            xlabel = f"PC{pc1+1}"
        if ylabel is None:
            ylabel = f"PC{pc2+1}"

        markers = kwargs.get('markers', None)

        for ei,e in enumerate(events):
            XY = proj_plot[e]  # [2 x T]
            x = XY[0][x_mask].mean()
            y = XY[1][x_mask].mean()

            # Scatter plot
            scatter = ax.scatter(
                x, y,
                c=colors,
                s=marker_size,
                alpha=1,
                marker=markers[ei] if markers is not None else None,
                edgecolors='black',
                linewidth=0.25
            )

            # Add SEM visualization if requested
            if show_sem:
                sem_x = sem(x)
                sem_y = sem(y)

                mean_x = np.mean(x)
                mean_y = np.mean(y)

                # Draw error bars around mean
                ax.errorbar(
                    mean_x, mean_y,
                    xerr=sem_x * sem_scale,
                    yerr=sem_y * sem_scale,
                    fmt='none',
                    ecolor='k',
                    elinewidth=2,
                    capsize=5,
                    label=f'SEM ({sem_scale}σ)'
                )
                ax.plot(mean_x, mean_y, 'r+', markersize=12, markeredgewidth=2)

        format_axis(ax,xlabel=xlabel,ylabel=ylabel,title=title)

        if labels is not None and colors is None:
            ax.legend(labels, loc='best')
        elif show_sem:
            ax.legend(loc='best')

        self.scatter_plot = (fig, ax)
        return fig, ax

def get_event_response(event_response_dict, event):
    event_responses = np.array_split(event_response_dict[event], event_response_dict[event].shape[0], axis=0)

    event_responses = [np.squeeze(e) for e in event_responses]

    return event_responses


def align_pca_groups(
    proj_full,
    events,
    x_ser,
    *,
    parts_to_match=None,         # e.g. [0], [1], [0,2] or None (align all together)
    name_grouper=None,           # callable(events, parts_to_match) -> dict[group_key] = [events]
    align_method='orthogonal',   # 'orthogonal' | 'cca'
    align_window=None            # (t0, t1) in same units as x_ser
):
    """
    Align per-event PCA time series (all PCs) within groups of events.

    Parameters
    ----------
    proj_full : dict[str, np.ndarray]
        Mapping event -> [n_pcs x T] arrays.
    events : list[str]
        Event names present in proj_full.
    x_ser : np.ndarray shape (T,)
        Time vector aligned to columns of proj_full arrays.
    parts_to_match : list[int] | None
        Indices of name parts to group by. If None, all events are one group.
    name_grouper : callable
        Function like `group_by_parts(events, parts_to_match)` that returns
        dict[group_key] -> list[event_names].
    align_method : {'orthogonal','cca'}
        Alignment method to use between trajectories.
    align_window : tuple[float, float] | None
        Time window to use for alignment mask. If None, use full x_ser.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping event -> aligned [n_pcs x T] arrays (reference events unchanged).
    """

    if align_window is None:
        align_mask = np.ones_like(x_ser, dtype=bool)
    else:
        a0, a1 = align_window
        align_mask = (x_ser >= a0) & (x_ser <= a1)

    # Build groups
    if parts_to_match is None:
        groups = {"all": events}
    else:
        if name_grouper is None:
            raise ValueError("name_grouper must be provided when parts_to_match is not None")
        groups = name_grouper(events, parts_to_match)

    proj_aligned = dict(proj_full)  # start from originals

     # stack pips
    stacked_events = [np.concatenate([proj_aligned[e] for e in pip_group],axis=0) for pip_group in groups.values()]
    ref_stacked = stacked_events[0]  # 0 for exemplars 2 for abstraction
    aligned_stacks = [procrustes_align_pca_timeseries(ref_stacked, mat, align_mask)[1]# if ai > 0 else mat
    # aligned_stacks = [cca_align_pca_timeseries(ref_stacked, mat, align_mask)[1] if ai > 0 else mat
                      for ai, mat in enumerate(stacked_events)]

    unstacked_events_by_group = [np.split(mat,len(pip_group)) for mat,pip_group in zip(aligned_stacks, groups.values())]
    proj_aligned = {}
    for grouped_events, unstacked_events in zip(groups.values(),unstacked_events_by_group):
        for e,aligned_mat in zip(grouped_events, unstacked_events):
            proj_aligned[e] = aligned_mat

    # for _, grouped_events in groups.items():
    #     if not grouped_events:
    #         continue
    #     ref_e = grouped_events[0]
    #     ref_mat = proj_full[ref_e]
    #     proj_aligned[ref_e] = ref_mat  # keep reference unchanged
    #
    #     for e in grouped_events[1:]:
    #         mat = proj_full[e]
    #         if align_method == 'orthogonal':
    #             aligned = procrustes_align_pca_timeseries(ref_mat, mat, align_mask)[1]
    #         elif align_method == 'cca':
    #             aligned = cca_align_pca_timeseries(ref_mat, mat, align_mask)[1]
    #         else:
    #             raise NotImplementedError(f"Unknown align_method: {align_method}")
    #         proj_aligned[e] = aligned

    return proj_aligned

def align_pca_groups_new_maybe(
    proj_full, events, x_ser,
    *, parts_to_match=None, name_grouper=None,
    align_method="orthogonal", align_window=None
):
    if align_window is None:
        align_mask = np.ones_like(x_ser, dtype=bool)
    else:
        a0, a1 = align_window
        align_mask = (x_ser >= a0) & (x_ser <= a1)

    # groups
    if parts_to_match is None:
        groups = {"all": events}
    else:
        if name_grouper is None:
            raise ValueError("name_grouper must be provided when parts_to_match is not None")
        groups = name_grouper(events, parts_to_match)

    proj_aligned = dict(proj_full)

    for gkey, pip_group in groups.items():
        if len(pip_group) < 2:
            continue

        ref_e = pip_group[0]
        Xr = proj_full[ref_e].T            # (T, n_pcs)
        Xr_sub = Xr[align_mask]            # (n_sub, n_pcs)

        # concatenate all other events' samples under the same feature space (PCs)
        for e in pip_group[1:]:
            Xt = proj_full[e].T
            Xt_sub = Xt[align_mask]

            if align_method == "orthogonal":

                R, _ = orthogonal_procrustes(Xt_sub, Xr_sub)   # (n_pcs, n_pcs)

                Xt_rot = (Xt @ R).T                             # back to (n_pcs, T)
                proj_aligned[e] = Xt_rot

            elif align_method == "cca":
                # CCA will change dimensionality unless you force full-rank & keep it.
                # For your stacking/unstacking pipeline, CCA is the wrong primitive.
                raise NotImplementedError("CCA not supported for full-dim per-event alignment. Use orthogonal.")
            else:
                raise NotImplementedError

        proj_aligned[ref_e] = proj_full[ref_e]

    return proj_aligned


def compute_eig_vals(X,plot_flag=False):
    c = np.cov(X, rowvar=True)  # covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(c)
    srt = np.argsort(eig_vals)[::-1]
    print(srt)
    eig_vals = eig_vals[srt]
    eig_vecs = eig_vecs[:, srt]
    fig, ax = plt.subplots()
    if plot_flag:
        ax.plot(np.cumsum(eig_vals / eig_vals.sum()), label='cumulative % variance explained')
        ax.plot(eig_vals / eig_vals.sum(), label='% variance explained')
        ax.set_ylim([0, 1])
        n_comp_to_thresh = np.argwhere(np.cumsum(eig_vals / eig_vals.sum()) > 0.9)[0][0]
        ax.plot([n_comp_to_thresh] * 2, [0, 0.9], color='k', ls='--', )
        ax.plot([0, n_comp_to_thresh], [0.9, 0.9], color='k', ls='--', )
        ax.legend()
        # fig.show()

    return eig_vals, eig_vecs,(fig,ax)


def compute_trial_averaged_pca(X_trial_averaged,n_components=15,standardise=False):
    # Xa = z_score(X_trial_averaged)
    if standardise:
        X_std = StandardScaler().fit_transform(X_trial_averaged)
    else:
        X_std = X_trial_averaged
    pca = PCA(n_components=n_components)
    pca.fit(X_std.T)

    return pca


def project_pca(X_trial,pca,standardise=False):
    # ss = StandardScaler(with_mean=True, with_std=True)
    if standardise:
        trial_sc = StandardScaler().fit_transform(X_trial)
    else:
        trial_sc = X_trial
    proj_trial = pca.transform(trial_sc.T).T
    return proj_trial


def plot_pca_ts(X_proj_by_event, events, window, plot=None, n_components=3,plot_kwargs=None):
    if not plot:
        fig, axes = plt.subplots(1, n_components, figsize=[20, 4],)
    else:
        fig,axes = plot

    x_ser = np.linspace(window[0], window[1], X_proj_by_event[0][0].shape[-1])
    # smooth
    if plot_kwargs.get('smoothing', None):
        smoothing = plot_kwargs.get('smoothing')
        for ei, event in enumerate(events):
            for comp in range(n_components):
                X_proj_by_event[ei][comp] = gaussian_filter1d(X_proj_by_event[ei][comp], smoothing)

    for comp in range(n_components):
        ax = axes[comp]
        for ei, event in enumerate(events):
            projected_trials = np.array(X_proj_by_event[ei])
            if projected_trials.ndim == 3:
                projected_trials_comp = projected_trials[:, comp, :]
                pc_mean_ts =  projected_trials.mean(axis=0)
            else:
                projected_trials_comp = projected_trials[comp]
                pc_mean_ts = projected_trials_comp
            kwargs2use = plot_kwargs if plot_kwargs is not None else {}
            if plot_kwargs.get('ls',{}):
                if isinstance(plot_kwargs.get('ls'),list):
                    kwargs2use['ls']=plot_kwargs.get('ls',{})[ei]
                else:
                    kwargs2use['ls']=plot_kwargs.get('ls',{})
            if plot_kwargs.get('c'):
                if isinstance(plot_kwargs.get('c'),list):
                    kwargs2use['c']=plot_kwargs.get('c')[ei]
                else:
                    kwargs2use['c']=plot_kwargs.get('c')

            ax.plot(x_ser,pc_mean_ts,**kwargs2use)
            if projected_trials.ndim == 3:
                plot_ts_var(x_ser,projected_trials_comp,kwargs2use.get('c',f'C{ei}'),ax)

        ax.set_ylabel(f'PC {comp+1}')
        ax.set_xlabel('Time (s)')
        ax.axvline(0, color='k', ls='--')
        # ax.legend(ncol=len(events))

    # axes[-1].legend(ncol=len(events))
    return fig, axes



def cca_align_pca_timeseries(
    X_ref, X_tgt, time_mask, n_components=None, standardise=False, fix_signs=False
):
    """
    Align two PCA×time trajectories via CCA trained on a subset of timepoints,
    then apply the learned transforms to the full trajectories.

    Parameters
    ----------
    X_ref : (n_pcs, n_times) array
        Reference subject PCA time series.
    X_tgt : (n_pcs, n_times) array
        Target subject PCA time series to align to the reference.
    time_mask : (n_times,) boolean array
        True for the timepoints used to FIT CCA (alignment window).
    n_components : int or None
        Number of canonical components to extract. If None, uses
        min(n_pcs_ref, n_pcs_tgt).
    standardise : bool
        If True, z-score features using stats computed on the subset; then
        apply the same scaling to the full series.
    fix_signs : bool
        If True, flips canonical components so correlations on the subset are positive.

    Returns
    -------
    X_ref_c_full : (n_components, n_times) array
        Reference canonical time series for the full time axis.
    X_tgt_c_full : (n_components, n_times) array
        Target canonical time series, aligned to reference space, full time axis.
    cca : fitted sklearn.cross_decomposition.CCA
        The fitted CCA model (you can reuse it to transform other events).
    meta : dict
        Useful info: scalers, applied sign flips, feature indices, etc.
    """
    # --- Validate & harmonize feature dims ---
    n_pcs_ref, n_times_ref = X_ref.shape
    n_pcs_tgt, n_times_tgt = X_tgt.shape
    if n_times_ref != n_times_tgt:
        raise ValueError(f"Time dimension mismatch: {n_times_ref} vs {n_times_tgt}")
    if time_mask.dtype != bool or time_mask.shape[0] != n_times_ref:
        raise ValueError("time_mask must be boolean of length n_times")

    # Use common feature count if they differ
    n_feat = min(n_pcs_ref, n_pcs_tgt)
    # n_feat = 3
    n_sub = int(time_mask.sum())
    # n_feat = min(n_feat, n_sub)

    if n_pcs_ref != n_pcs_tgt:
        # Keep leading PCs; adjust here if you prefer another selection
        X_ref = X_ref[:n_feat, :]
        X_tgt = X_tgt[:n_feat, :]

    # Transpose to samples×features for CCA
    Xr = X_ref.T          # (n_times, n_feat)
    Xt = X_tgt.T          # (n_times, n_feat)
    Xr_sub = Xr[time_mask]
    Xt_sub = Xt[time_mask]

    # Standardize using subset stats, apply to full series
    if standardise:
        scaler_r = StandardScaler().fit(Xr_sub)
        scaler_t = StandardScaler().fit(Xt_sub)
        Xr_full_std = scaler_r.transform(Xr)
        Xt_full_std = scaler_t.transform(Xt)
        Xr_sub_std  = Xr_full_std[time_mask]
        Xt_sub_std  = Xt_full_std[time_mask]
    else:
        scaler_r = scaler_t = None
        Xr_full_std, Xt_full_std = Xr, Xt
        Xr_sub_std,  Xt_sub_std  = Xr_sub, Xt_sub

    # Number of canonical components
    if n_components is None:
        n_components = n_feat
    n_components = min(n_components, n_feat)

    # Fit CCA on the subset only
    cca = CCA(n_components=n_components)
    Zr_sub, Zt_sub = cca.fit(Xr_sub_std, Xt_sub_std).transform(Xr_sub_std, Xt_sub_std)

    # Transform FULL time series with the fitted model
    Zr_full, Zt_full = cca.transform(Xr_full_std, Xt_full_std)  # (n_times, n_components)

    # Optional: fix signs to make subset correlations positive component-wise
    flips = np.ones(n_components)
    if fix_signs:
        # compute Pearson sign on subset and flip Zt to match Zr
        for k in range(n_components):
            r = np.corrcoef(Zr_sub[:, k], Zt_sub[:, k])[0, 1]
            if np.isnan(r) or r < 0:
                Zt_full[:, k] *= -1
                Zt_sub[:, k]  *= -1
                flips[k] = -1.0

    # Return in (components × time) like your input
    X_ref_c_full = Zr_full.T
    X_tgt_c_full = Zt_full.T

    meta = {
        "scaler_ref": scaler_r,
        "scaler_tgt": scaler_t,
        "n_components": n_components,
        "n_features_used": n_feat,
        "sign_flips_tgt": flips,
        "time_mask": time_mask.copy(),
    }
    return X_ref_c_full, X_tgt_c_full, cca, meta

def procrustes_align_pca_timeseries(
    X_ref, X_tgt, time_mask, n_components=None, standardise=False, fix_signs=False
):
    """
    Align two PCA×time trajectories via Orthogonal Procrustes (rotation-only)
    trained on a subset of timepoints, then apply to the full trajectories.

    Parameters
    ----------
    X_ref : (n_pcs, n_times) array
        Reference PCA time series.
    X_tgt : (n_pcs, n_times) array
        Target PCA time series to align to the reference.
    time_mask : (n_times,) boolean array
        True for the timepoints used to FIT the rotation (alignment window).
    n_components : int or None
        Number of PCs to keep after alignment (from the reference ordering).
        If None, uses min(n_pcs_ref, n_pcs_tgt).
    standardise : bool
        If True, z-score each series using stats computed on the subset, then
        apply the same scaling to the full series. Returned results are mapped
        back to the reference's ORIGINAL units.
    fix_signs : bool
        If True, flips individual aligned target axes so correlations with the
        reference on the subset are positive (after rotation).

    Returns
    -------
    X_ref_full : (n_components, n_times)
        Reference time series (possibly standardised then *inverse* transformed
        back to original units), truncated to n_components.
    X_tgt_aligned_full : (n_components, n_times)
        Target time series rotated into the reference space (and mapped back to
        the reference's units), full time axis.
    R : (n_feat, n_feat) ndarray
        The orthogonal rotation matrix learned on the subset (acts on features).
    meta : dict
        {"scaler_ref","scaler_tgt","n_components","n_features_used",
         "sign_flips_tgt","time_mask","op_scale"}
    """
    # --- validate shapes ---
    X_ref = np.asarray(X_ref)
    X_tgt = np.asarray(X_tgt)
    time_mask = np.asarray(time_mask, dtype=bool)

    n_pcs_ref, n_times_ref = X_ref.shape
    n_pcs_tgt, n_times_tgt = X_tgt.shape
    if n_times_ref != n_times_tgt:
        raise ValueError(f"Time dimension mismatch: {n_times_ref} vs {n_times_tgt}")
    if time_mask.shape[0] != n_times_ref:
        raise ValueError("time_mask must be boolean of length n_times")

    # --- harmonize feature count ---
    n_feat = min(n_pcs_ref, n_pcs_tgt)
    n_sub = int(time_mask.sum())
    # n_feat = min(n_feat, n_sub)

    X_ref = X_ref[:n_feat, :]
    X_tgt = X_tgt[:n_feat, :]
    if n_components is None:
        n_components = n_feat
    n_components = min(n_components, n_feat)

    # Work in samples×features form
    Xr = X_ref.T            # (n_times, n_feat)
    Xt = X_tgt.T            # (n_times, n_feat)
    Xr_sub = Xr[time_mask]  # (n_sub, n_feat)
    Xt_sub = Xt[time_mask]

    # Standardize using subset stats; keep objects for inverse-transform
    if standardise:
        scaler_r = StandardScaler().fit(Xr_sub)
        scaler_t = StandardScaler().fit(Xt_sub)
        Xr_full_std = scaler_r.transform(Xr)
        Xt_full_std = scaler_t.transform(Xt)
        Xr_sub_std  = Xr_full_std[time_mask]
        Xt_sub_std  = Xt_full_std[time_mask]
    else:
        scaler_r = scaler_t = None
        Xr_full_std, Xt_full_std = Xr, Xt
        Xr_sub_std,  Xt_sub_std  = Xr_sub, Xt_sub

    # --- learn orthogonal rotation on the subset ---
    # Find R such that Xt_sub_std @ R ≈ Xr_sub_std
    R, op_scale = orthogonal_procrustes(Xt_sub_std, Xr_sub_std)  # R: (n_feat, n_feat)

    # Apply rotation to FULL time axis (still in standardized units if used)
    Xt_full_rot = Xt_full_std @ R
    Xt_sub_rot  = Xt_sub_std  @ R

    # Optional: make each aligned component positively correlated with reference on the subset
    flips = np.ones(n_feat)
    if fix_signs:
        for k in range(n_feat):
            r = np.corrcoef(Xr_sub_std[:, k], Xt_sub_rot[:, k])[0, 1]
            if np.isnan(r) or r < 0:
                Xt_full_rot[:, k] *= -1
                Xt_sub_rot[:, k]  *= -1
                flips[k] = -1.0

    # Map to reference's ORIGINAL units if we standardised
    if standardise:
        X_ref_full_units = scaler_r.inverse_transform(Xr_full_std)     # reference back to its units
        X_tgt_full_units = scaler_r.inverse_transform(Xt_full_rot)     # target in ref units
    else:
        X_ref_full_units = Xr_full_std
        X_tgt_full_units = Xt_full_rot

    # Keep requested number of components (use reference ordering)
    X_ref_full = X_ref_full_units[:, :n_components].T        # (n_components, n_times)
    X_tgt_aligned_full = X_tgt_full_units[:, :n_components].T

    meta = {
        "scaler_ref": scaler_r,
        "scaler_tgt": scaler_t,
        "n_components": n_components,
        "n_features_used": n_feat,
        "sign_flips_tgt": flips[:n_components],
        "time_mask": time_mask.copy(),
        "op_scale": op_scale,          # sum of singular values (diagnostic)
        "R": R,                        # rotation (acts on features/PCs)
    }
    return X_ref_full, X_tgt_aligned_full, R, meta