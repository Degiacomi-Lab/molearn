from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

plt.rcParams.update({
    "axes.titlesize": 16,    # Figure title
    "axes.labelsize": 14,    # X and Y labels
    "xtick.labelsize": 12,   # X-axis tick labels
    "ytick.labelsize": 12,   # Y-axis tick labels
    "legend.fontsize": 14    # Legend font size
})


def _ensure_path(path: Optional[Path | str]) -> Optional[Path]:
    if path is None:
        return None
    return path if isinstance(path, Path) else Path(path)


def _flatten(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def _to_numpy(values):
    if isinstance(values, np.ndarray):
        return values
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    return np.asarray(values)


def _collect_metric_entries(
    MA,
    plot_data: Sequence[Tuple[str, str, str]],
    metric_fetcher: Callable[[str], Dict[str, Dict[str, np.ndarray]]],
    dataset_key: str,
    decoded_key: str,
) -> Tuple[List[Dict[str, object]], List[str]]:
    """
    Resolve metric dictionaries for datasets and decodes to a unified format.

    :param MolearnAnalysis MA: Analysis object used to resolve metric data for the requested keys.
    :param Sequence[Tuple[str, str, str]] plot_data: Iterable of ``(key, label, colour)`` tuples describing which datasets to plot,
                                                      the legend label to apply, and the colour associated with decoded structures.
    :param Callable[[str], Dict[str, Dict[str, np.ndarray]]] metric_fetcher: Callback returning a mapping of metric group names (e.g. ``dataset_bondlen``)
                                                                              to dictionaries of metric arrays for the provided ``key``.
    :param str dataset_key: Dictionary key under each metric group corresponding to the original dataset values.
    :param str decoded_key: Dictionary key under each metric group corresponding to decoded structure values.

    :return: Tuple of (entries, metric_names) where entries is a list of dicts containing dataset/decoded metric dictionaries
             and legend metadata for downstream plotting, and metric_names is an ordered list of metric names present in the decoded metric dictionary.
    """
    entries: List[Dict[str, object]] = []
    metric_keys: Optional[Iterable[str]] = None
    for key, label, decoded_color in plot_data:
        metric = metric_fetcher(key)
        dataset_metric = metric.get(dataset_key)
        decoded_metric = metric.get(decoded_key)
        if decoded_metric is None:
            raise ValueError(f"Decoded metric '{decoded_key}' missing for key '{key}'.")
        if metric_keys is None:
            metric_keys = decoded_metric.keys()
        entries.append(
            {
                "label": label,
                "dataset": dataset_metric,
                "decoded": decoded_metric,
                "colors": ("gray" if dataset_metric is not None else None, decoded_color),
            }
        )
    return entries, list(metric_keys or [])


def _plot_metric_histograms(
    entries: Sequence[Dict[str, object]],
    metric_names: Sequence[str],
    bins: int,
    wkdir: Optional[Path],
    filename_prefix: str,
    xlabel: str,
    xlim: Optional[Tuple[float, float]] | Dict[str, Tuple[float, float]] = None,
    density: bool = True,
    legend_suffix: str = "",
    save_kwargs: Optional[Dict] = None,
):
    save_kwargs = save_kwargs or {}
    figures = []
    for metric_name in metric_names:
        fig, axes = plt.subplots(len(entries), 1, figsize=(8, 4 * len(entries)), sharex=True)
        figures.append(fig)
        axes = np.atleast_1d(axes)
        for ax, entry in zip(axes, entries):
            dataset_metric = entry["dataset"]
            decoded_metric = entry["decoded"]
            dataset_color, decoded_color = entry["colors"]
            label = entry["label"]
            if dataset_metric is not None:
                ax.hist(
                    _flatten(dataset_metric[metric_name]),
                    bins=bins,
                    color=dataset_color,
                    alpha=0.5,
                    density=density,
                    hatch="//",
                    label=f"{label} Dataset{legend_suffix}",
                )
            ax.hist(
                _flatten(decoded_metric[metric_name]),
                bins=bins,
                color=decoded_color,
                alpha=0.5,
                density=density,
                label=f"{label} Decoded{legend_suffix}",
            )
            ax.set_title(f"{label} - {metric_name}")
            ax.set_ylabel("Density" if density else "Count")
            if isinstance(xlim, dict):
                limits = xlim.get(metric_name)
            else:
                limits = xlim
            if limits is not None:
                ax.set_xlim(*limits)
            ax.set_xlabel(xlabel)
            ax.legend()
            ax.tick_params(labelbottom=True, labelleft=False)
        plt.tight_layout()
        if wkdir is not None:
            plt.savefig(wkdir / f"{filename_prefix}_{metric_name}.pdf", **save_kwargs)
        plt.show()
    return figures


def _latent_edge(values: np.ndarray) -> np.ndarray:
    return np.append(values, (2 * values[-1] - values[-2]))


def _overlay_latent_points(ax, MA, plot_data):
    if not plot_data:
        return []
    legend_handles = []
    for key, label, colour, plot_type in plot_data:
        coords = _to_numpy(MA.get_encoded(key)).squeeze()
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(f"Encoded coordinates for '{key}' must be of shape (N, 2)")
        x, y = coords[:, 0], coords[:, 1]
        if plot_type == "scatter":
            ax.scatter(x=x, y=y, marker="o", s=1, c=colour, label=label)
            legend_handles.append(mpl.lines.Line2D([0], [0], color=colour, linestyle="", marker="o", markersize=8, label=label))
        elif plot_type == "kde":
            sns.kdeplot(x=x, y=y, levels=7, color=colour, ax=ax, label=label)
            legend_handles.append(mpl.lines.Line2D([0], [0], color=colour, lw=2, label=label))
        else:
            raise ValueError(f"Unknown plot_type '{plot_type}' for key '{key}'.")
    return legend_handles


def plot_bondlength_hist(MA, plot_data=None, bins: int = 100, wkdir=None, **kwargs):
    """
    Plot bond-length distributions for original and decoded structures.
    
    :param MolearnAnalysis MA: A MolearnAnalysis object with datasets loaded.
    :param list plot_data: A list of tuples containing the data to plot. Each tuple should contain
                           the key to a original/encoded dataset in the MolearnAnalysis object, a label for the legend,
                           and a colour for the decoded data plot.
                           Format: [(key, label, colour), ...]
    :param int bins: No of bins in the histogram.
    :param Path wkdir: A directory where figures are saved.

    :return: None
    """

    if plot_data is None:
        raise ValueError("plot_data must be provided.")

    wkdir = _ensure_path(wkdir)
    entries, metric_names = _collect_metric_entries(
        MA,
        plot_data,
        MA.get_bondlengths,
        dataset_key="dataset_bondlen",
        decoded_key="decoded_bondlen",
    )
    _plot_metric_histograms(
        entries,
        metric_names,
        bins=bins,
        wkdir=wkdir,
        filename_prefix="BL",
        xlabel="Bond length (Å)",
        xlim=(0.0, 2.5),
        save_kwargs=kwargs,
    )

def plot_dihedral_hist(MA, plot_data=None, bins: int = 100, wkdir=None, **kwargs):
    """
    Plot backbone dihedral distributions for dataset and decoded structures.
    
    :param MolearnAnalysis MA: A MolearnAnalysis object with datasets loaded.
    :param list plot_data: A list of tuples containing the data to plot. Each tuple should contain
                           the key to a original/encoded dataset in the MolearnAnalysis object, a label for the legend,
                           and a colour for the decoded data plot.
                           Format: [(key, label, colour), ...]
    :param int bins: No of bins in the histogram.
    :param Path wkdir: A directory where figures are saved.

    :return: None
    """

    if plot_data is None:
        raise ValueError("plot_data must be provided.")

    wkdir = _ensure_path(wkdir)
    entries, metric_names = _collect_metric_entries(
        MA,
        plot_data,
        MA.get_dihedrals,
        dataset_key="dataset_dihedrals",
        decoded_key="decoded_dihedrals",
    )
    _plot_metric_histograms(
        entries,
        metric_names,
        bins=bins,
        wkdir=wkdir,
        filename_prefix="Dihed",
        xlabel="Radians",
        xlim=(-np.pi, np.pi),
        save_kwargs=kwargs,
    )


def plot_angle_hist(MA, plot_data=None, bins: int = 100, wkdir=None, **kwargs):
    """
    Plot bond-angle distributions for original and decoded structures.
    
    :param MolearnAnalysis MA: A MolearnAnalysis object with datasets loaded.
    :param list plot_data: A list of tuples containing the data to plot. Each tuple should contain
                           the key to a original/encoded dataset in the MolearnAnalysis object, a label for the legend,
                           and a colour for the decoded data plot.
                           Format: [(key, label, colour), ...]
    :param int bins: No of bins in the histogram.
    :param Path wkdir: A directory where figures are saved.

    :return: None
    """

    if plot_data is None:
        raise ValueError("plot_data must be provided.")

    wkdir = _ensure_path(wkdir)
    entries, metric_names = _collect_metric_entries(
        MA,
        plot_data,
        MA.get_angles,
        dataset_key="dataset_angles",
        decoded_key="decoded_angles",
    )
    _plot_metric_histograms(
        entries,
        metric_names,
        bins=bins,
        wkdir=wkdir,
        filename_prefix="Agl",
        xlabel="Radians",
        xlim=(0.0, np.pi),
        save_kwargs=kwargs,
    )


def plot_inversion_hist(MA, plot_data, fname=None, **kwargs):
    """
    Plot distributions of number of D-amino acids in each structure in datasets as bar plots

    :param MolearnAnalysis MA: A MolearnAnalysis object with datasets loaded.
    :param list plot_data: A list of tuples containing the data to plot. Each tuple should contain
                           the key to a original/encoded dataset in the MolearnAnalysis object, a label for the legend,
                           and a colour for the decoded data plot.
                           Format: [(key, label, colour), ...]
    :param Path fname: File name of the plot.
    :param bool refine: if True, refine structures before calculating DOPE score
    :param dict kwargs: Additional keyword arguments to pass to plt.savefig.

    :return: None
    """

    if plot_data is None:
        raise ValueError("plot_data must be provided.")
    
    data_pairs = []
    labels = []
    colors = []
    dataset_color = "gray"  # Set dataset bars to gray
    
    for key, label, decoded_color in plot_data:
        result = MA.get_inversions(key)
        dataset_inversions = result.get('dataset_inversions')
        decoded_inversions = result.get('decoded_inversions')
        if decoded_inversions is None:
            raise ValueError(f"Decoded inversions unavailable for key '{key}'")
        data_pairs.append((dataset_inversions, decoded_inversions))
        labels.append(label)
        colors.append((dataset_color if dataset_inversions is not None else None, decoded_color))
    
    num_plots = len(data_pairs)
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 3 * num_plots), sharex=True)
    
    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable if there’s only one plot
    
    for ax, (dataset_data, decoded_data), label, (dataset_color, decoded_color) in zip(axes, data_pairs, labels, colors):
        
        if dataset_data is not None:
            dataset_values, dataset_counts = np.unique(dataset_data, return_counts=True)
            ax.bar(dataset_values, dataset_counts, color=dataset_color, alpha=0.7, edgecolor='black', hatch='//', label=f'{label} dataset')
        
        decoded_values, decoded_counts = np.unique(decoded_data, return_counts=True)
        ax.bar(decoded_values, decoded_counts, color=decoded_color, alpha=0.7, edgecolor='black', label=f'{label} decoded')
        
        ax.set_ylabel("Count")
        ax.set_title(label, fontsize=10)
        ax.legend()
    
    axes[-1].set_xlabel("D-amino acids count")
    
    plt.tight_layout()
    
    if fname is not None:
        plt.savefig(fname, **kwargs)
    plt.show()
    

def plot_dope_hist(MA, plot_data=None, fname=None, refine=True, **kwargs):
    """
    Plot DOPE score distributions as split violin plots comparing datasets and decoded structures.
    
    :param MolearnAnalysis MA: A MolearnAnalysis object with datasets loaded.
    :param list plot_data: A list of tuples containing the data to plot. Each tuple should contain
                           the key to an original/decoded dataset in the MolearnAnalysis object, a label for the legend,
                           and a colour for the decoded data plot.
                           Format: [(key, label, colour), ...]
    :param Path fname: File name to save the plot.
    :param bool refine: If True, refine structures before calculating DOPE score. Can also be 'both' to plot both refined and unrefined.

    :return: None
    """

    if plot_data is None:
        raise ValueError("plot_data must be provided.")

    if refine not in (True, False, "both"):
        raise ValueError("refine must be True, False, or 'both'.")

    def _score_dict(values):
        arr = np.asarray(values)
        if refine == "both" and arr.ndim == 2 and arr.shape[1] == 2:
            return {"Raw": _flatten(arr[:, 0]), "Refined": _flatten(arr[:, 1])}
        return {"DOPE": _flatten(arr)}

    def _fetch(key: str):
        metrics: Dict[str, Dict[str, np.ndarray]] = {}
        decoded_scores = MA.get_all_dope_score(MA.get_decoded(key, scale=True), refine=refine)
        metrics["decoded"] = _score_dict(decoded_scores)
        if key in MA._datasets:
            dataset_scores = MA.get_all_dope_score(MA.get_dataset(key, scale=True), refine=refine)
            metrics["dataset"] = _score_dict(dataset_scores)
        return metrics

    entries, metric_names = _collect_metric_entries(
        MA,
        plot_data,
        _fetch,
        dataset_key="dataset",
        decoded_key="decoded",
    )

    dataset_colour = "#B3B3B3"
    figures: List[plt.Figure] = []
    for metric_name in metric_names:
        data_pairs: List[np.ndarray] = []
        color_pairs: List[str] = []
        labels: List[str] = []
        decoded_handles: Dict[str, mpatches.Patch] = {}

        for entry in entries:
            dataset_metric = entry["dataset"]
            if dataset_metric is None:
                raise ValueError(f"Dataset DOPE scores unavailable for key '{entry['label']}'.")
            decoded_metric = entry["decoded"]
            dataset_values = _flatten(dataset_metric[metric_name])
            decoded_values = _flatten(decoded_metric[metric_name])
            data_pairs.extend([dataset_values, decoded_values])
            labels.append(entry["label"])
            decoded_color = entry["colors"][1]
            color_pairs.extend([dataset_colour, decoded_color])
            if entry["label"] not in decoded_handles:
                decoded_handles[entry["label"]] = mpatches.Patch(color=decoded_color, label=f"{entry['label']} decoded")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            data=data_pairs,
            split=True,
            inner="quart",
            palette=color_pairs,
            ax=ax,
            gap=-0.05,
            cut=0,
            width=0.9,
            bw=0.2,
            native_scale=True,
            dodge=False,
        )

        legend_patches = [mpatches.Patch(color=dataset_colour, label="Dataset")]
        legend_patches.extend(decoded_handles.values())
        ax.legend(handles=legend_patches, loc='upper right')

        ax.set_ylabel('DOPE score')
        ax.set_xticks(np.arange(len(labels)) * 2 + 0.5)
        ax.set_xticklabels(labels, rotation=0)
        title_suffix = f" ({metric_name})" if len(metric_names) > 1 else ""
        ax.set_title(f'Distribution of DOPE scores{title_suffix}')
        ax.grid(True, linestyle='--', alpha=0.6)

        figures.append(fig)
        plt.show()

    # Save the plot if fname is provided
    if fname:
        plt.savefig(fname, **kwargs)
    plt.show()



def plot_rmsd_hist(MA, plot_data=None, fname=None, **kwargs):
    """
    Plot distributions of RMSD scores of the chosen datasets.

    :param MolearnAnalysis MA: A MolearnAnalysis object with datasets loaded.
    :param list plot_data: A list of tuples containing the data to plot. Each tuple should contain
                           a pair of keys corresponding to two datasets in MolearnAnalysis object
                           which would be plotted on a split violin, a label for the legend, and 
                           keywords indicating if datasets are train or test. 
                           Format: [(key1, key2, label, kw1, kw2), ...]
    :param Path fname: The file name to save the plots.
    :param dict kwargs: Additional keyword arguments to pass to plt.savefig.

    :return: None
    """
    
    data_pairs = []
    labels = []
    color_pairs = []
    train_colour = "#FDBFCA"
    test_colour = "#AFC2DC"

    for key1, key2, label, kw1, kw2 in plot_data:
        if kw1 == 'train': c1 = train_colour
        elif kw1 == 'test': c1 = test_colour
        else: raise ValueError(f"Invalid keyword {kw1}.")
        if kw2 == 'train': c2 = train_colour
        elif kw2 == 'test': c2 = test_colour
        else: raise ValueError(f"Invalid keyword {kw2}.")
        error1 = MA.get_error(key1)
        error2 = MA.get_error(key2)
        data_pairs.extend([error1, error2])
        labels.append(label)
        color_pairs.extend([c1, c2])

    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.violinplot(data=data_pairs,
                   split=True, inner="quart", 
                   palette=color_pairs, 
                   ax=ax, gap=-0.05, cut=0, width=0.9, bw=0.2,
                   native_scale=True, dodge=False)
    
    legend_patches = [mpatches.Patch(color=train_colour, label="Train"),
                      mpatches.Patch(color=test_colour, label="Test")]
    ax.legend(handles=legend_patches, loc='upper right')

    ax.set_ylabel('RMSD (Å)')
    ax.set_xticks(np.arange(len(labels))*2+0.5)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title('Distribution of RMSD')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot if fname is provided
    if fname:
        plt.savefig(fname, **kwargs)
    plt.show()



def plot_network_rmsd_surface(MA, plot_data=None, cmap='gist_heat_r', fname=None, **kwargs):
    """
    Plot the latent grid coloured by reconstruction RMSD.

    :param MolearnAnalysis MA: Analysis instance with latent grid axes and a precomputed
                               ``surfaces['Network_RMSD']`` entry (produced by :meth:`MolearnAnalysis.scan_error`).
    :param list plot_data: Optional overlays given as ``(key, label, colour, plot_type)`` tuples. ``plot_type``
                           must be either ``'scatter'`` or ``'kde'`` and each ``key`` must correspond to an
                           encoded dataset available through :meth:`MolearnAnalysis.get_encoded`.
    :param str cmap: Matplotlib colormap name for the background surface.
    :param Path fname: Destination path for the saved figure. When omitted the plot is displayed only.

    :return: None
    """

    if 'Network_RMSD' not in MA.surfaces:
        raise ValueError("Run MA.scan_error before plotting the network RMSD surface.")

    cmap_obj = mpl.cm.get_cmap(name=cmap)
    data = MA.surfaces['Network_RMSD']
    xvals = _latent_edge(MA.xvals)
    yvals = _latent_edge(MA.yvals)

    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(xvals, yvals, data, cmap=cmap_obj, shading='auto')

    handles = _overlay_latent_points(ax, MA, plot_data or [])
    if handles:
        ax.legend(handles=handles, loc='upper right')

    ax.set_xlim(MA.xvals.min(), MA.xvals.max())
    ax.set_ylim(MA.yvals.min(), MA.yvals.max())
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')
    ax.grid(False)
    ax.set_aspect('equal')

    cbar_ax = fig.add_axes([
        ax.get_position().x1 + 0.02,
        ax.get_position().y0,
        0.02,
        ax.get_position().height,
    ])
    cb = fig.colorbar(mesh, cax=cbar_ax)
    cb.ax.tick_params(left=False, right=True)
    cb.ax.set_ylabel('RMSD (Å)')

    if fname is not None:
        plt.savefig(fname, **kwargs)
    plt.show()


def plot_dope_surface(MA, refine=True, truncate_at=None, plot_data=None, cmap='gist_heat_r', fname=None, **kwargs):
    """
    Plot the latent grid coloured by decoded DOPE scores.

    :param MolearnAnalysis MA: Analysis instance containing latent grid axes and precomputed DOPE surfaces
                               via :meth:`MolearnAnalysis.scan_dope`.
    :param bool refine: When ``True`` the refined DOPE surface (``'DOPE_refined'``) is rendered; otherwise
                        the unrefined surface (``'DOPE_unrefined'``) is used.
    :param float truncate_at: Upper bound for the colour scale. Defaults to the maximum value in the selected surface.
    :param list plot_data: Optional overlays given as ``(key, label, colour, plot_type)`` tuples, matching the
                           semantics described for :func:`plot_network_rmsd_surface`.
    :param str cmap: Matplotlib colormap name applied to the surface.
    :param Path fname: Destination path for the saved figure. When omitted the plot is displayed only.

    :return: None
    """
    surface_key = 'DOPE_refined' if refine else 'DOPE_unrefined'
    if surface_key not in MA.surfaces:
        raise ValueError("Run MA.scan_dope() before plotting the DOPE surface.")

    cmap_obj = mpl.cm.get_cmap(name=cmap)
    cmap_obj.set_over(cmap_obj(1.0))
    data = MA.surfaces[surface_key]
    xvals = _latent_edge(MA.xvals)
    yvals = _latent_edge(MA.yvals)

    if truncate_at is None:
        truncate_at = data.max()

    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(
        xvals,
        yvals,
        data,
        vmin=data.min(),
        vmax=truncate_at,
        cmap=cmap_obj,
        shading='auto',
    )

    handles = _overlay_latent_points(ax, MA, plot_data or [])
    if handles:
        ax.legend(handles=handles, loc='upper right')

    ax.set_xlim(MA.xvals.min(), MA.xvals.max())
    ax.set_ylim(MA.yvals.min(), MA.yvals.max())
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')
    ax.grid(False)
    ax.set_aspect('equal')

    cbar_ax = fig.add_axes([
        ax.get_position().x1 + 0.02,
        ax.get_position().y0,
        0.02,
        ax.get_position().height,
    ])
    cb = fig.colorbar(mesh, cax=cbar_ax)

    cb.ax.tick_params(left=False, right=True)
    cb.ax.set_ylabel('DOPE score')

    if fname is not None:
        plt.savefig(fname, **kwargs)
    plt.show()


def plot_inversion_surface(MA, plot_data=None, levels=10, cmap='gist_heat_r', fname=None, **kwargs):
    """
    Plot the latent grid coloured by predicted D-amino acid counts.

    :param MolearnAnalysis MA: Analysis instance with latent grid axes and chirality surface data produced by
                               :meth:`MolearnAnalysis.scan_ca_chirality`.
    :param list plot_data: Optional overlays given as ``(key, label, colour, plot_type)`` tuples, matching the
                           semantics described for :func:`plot_network_rmsd_surface`.
    :param int levels: Number of discrete contour levels used for the colour map.
    :param str cmap: Matplotlib colormap name applied to the surface.
    :param Path fname: Destination path for the saved figure. When omitted the plot is displayed only.

    :return: None
    """

    if 'Chirality' not in MA.surfaces:
        raise ValueError("Run MA.scan_ca_chirality before plotting the inversion surface.")

    cmap_obj = mpl.cm.get_cmap(name=cmap, lut=levels + 1)
    cmap_obj.set_under(cmap_obj(0))
    cmap_obj.set_bad(cmap_obj(0))
    cmap_obj.set_over(cmap_obj(levels))

    data = MA.surfaces['Chirality']
    masked_data = np.ma.masked_where(data == 0, data)
    xvals = _latent_edge(MA.xvals)
    yvals = _latent_edge(MA.yvals)

    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(
        xvals,
        yvals,
        masked_data,
        vmin=0,
        vmax=levels + 1,
        cmap=cmap_obj,
        shading='auto',
    )

    handles = _overlay_latent_points(ax, MA, plot_data or [])
    if handles:
        ax.legend(handles=handles, loc='upper right')

    ax.set_xlim(MA.xvals.min(), MA.xvals.max())
    ax.set_ylim(MA.yvals.min(), MA.yvals.max())
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')
    ax.grid(False)
    ax.set_aspect('equal')

    cbar_ax = fig.add_axes([
        ax.get_position().x1 + 0.02,
        ax.get_position().y0,
        0.02,
        ax.get_position().height,
    ])
    cb = fig.colorbar(mesh, cax=cbar_ax)

    cb.ax.tick_params(left=False, right=True)
    cb.ax.set_ylabel(r'Number of of D-amino acids')
    tick_labels = cb.ax.yaxis.get_ticklabels()
    tick_labels[-2].set_text(rf'$\geq${levels}')
    cb.ax.yaxis.set_ticklabels(tick_labels)
    ticks = cb.ax.get_yticks()
    ticks = [t + 0.5 for t in ticks[:-1]]
    cb.ax.yaxis.set_ticks(ticks)

    if fname is not None:
        plt.savefig(fname, **kwargs)
    plt.show()


def plot_analysis_surface(MA, dataset, cmap='gist_heat_r', fname=None, **kwargs):
    """
    Plot a specific dataset overlayed on its analysis surface.

    :param MolearnAnalysis MA: Analysis instance containing latent grid axes and the requested surface entry.
    :param str dataset: Key identifying the surface to render. Use :meth:`MolearnAnalysis.scan_dataset`
                        (or similar) to populate ``MA.surfaces[dataset]`` beforehand.
    :param str cmap: Matplotlib colormap name applied to the surface.
    :param Path fname: Destination path for the saved figure. When omitted the plot is displayed only.

    :return: None
    """

    if dataset not in MA.surfaces:
        raise ValueError(
            f"Dataset {dataset} not present in surfaces. Run MA.scan_dataset('{dataset}') first."
        )

    cmap_obj = mpl.cm.get_cmap(name=cmap)
    data = MA.surfaces[dataset]
    xvals = _latent_edge(MA.xvals)
    yvals = _latent_edge(MA.yvals)

    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(xvals, yvals, data, cmap=cmap_obj, shading='auto')

    for encoded in MA._encoded.values():
        coords = _to_numpy(encoded).squeeze()
        ax.scatter(x=coords[:, 0], y=coords[:, 1], marker='o', s=0.5, c='white')

    focal = _to_numpy(MA.get_encoded(dataset)).squeeze()
    ax.scatter(x=focal[:, 0], y=focal[:, 1], marker='o', s=1, c='black', label=dataset)
    ax.legend()

    ax.set_xlim(MA.xvals.min(), MA.xvals.max())
    ax.set_ylim(MA.yvals.min(), MA.yvals.max())
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')
    ax.grid(False)
    ax.set_aspect('equal')

    cbar_ax = fig.add_axes([
        ax.get_position().x1 + 0.02,
        ax.get_position().y0,
        0.02,
        ax.get_position().height,
    ])
    cb = fig.colorbar(mesh, cax=cbar_ax)
    cb.ax.tick_params(left=False, right=True)
    cb.ax.set_ylabel('RMSD (Å)')

    if fname is not None:
        plt.savefig(fname, **kwargs)
    plt.show()
