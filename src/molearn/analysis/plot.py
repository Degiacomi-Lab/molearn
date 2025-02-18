import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


plt.rcParams.update({
    "axes.titlesize": 16,    # Figure title
    "axes.labelsize": 14,    # X and Y labels
    "xtick.labelsize": 12,   # X-axis tick labels
    "ytick.labelsize": 12,   # Y-axis tick labels
    "legend.fontsize": 14    # Legend font size
})


def plot_bondlength_hist(MA, plot_data=None, wkdir=None, **kwargs):
    """
    Plot distributions of bond lengths of the chosen datasets.

    :param MolearnAnalysis MA: A MolearnAnalysis object with datasets loaded.
    :param list plot_data: A list of tuples containing the data to plot. Each tuple should contain
                           the key to a original/encoded dataset in the MolearnAnalysis object, a label for the legend,
                           and a colour for the decoded data plot.
                           Format: [(key, label, colour), ...]
    :param Path wkdir: The directory to save the plots.
    :param bool refine: if True, refine structures before calculating DOPE score
    :param dict kwargs: Additional keyword arguments to pass to plt.savefig.

    :return: None
    """

    data_pairs = []
    labels = []
    colors = []
    
    for key, label, decoded_color in plot_data:
        dataset_color = "gray"  # Set dataset distribution as gray
        if key in MA._datasets.keys():
            dataset_bondlen = MA.get_bondlengths(key)['dataset_bondlen']
            decoded_bondlen = MA.get_bondlengths(key)['decoded_bondlen']
        elif key in MA._encoded.keys():
            decoded_bondlen = MA.get_bondlengths(key)['decoded_bondlen']
            dataset_bondlen, dataset_color = None, None
        else:
            raise ValueError(f"{key} not found in datasets or encoded data")
        data_pairs.append((dataset_bondlen, decoded_bondlen))
        labels.append(label)
        colors.append((dataset_color, decoded_color))

    num_plots = len(data_pairs)
    
    for key in data_pairs[0][1].keys():
        fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots), sharex=True)
        
        if num_plots == 1:
            axes = [axes]  # Ensure axes is iterable if there’s only one plot
        
        for ax, (dataset_data, decoded_data), label, (dataset_color, decoded_color) in zip(axes, data_pairs, labels, colors):
            
            if dataset_data is not None:
                ax.hist(dataset_data[key].flatten(), bins=100, color=dataset_color, label=f'{label} Dataset', alpha=0.5, density=True, hatch='//')
            
            ax.hist(decoded_data[key].flatten(), bins=100, color=decoded_color, label=f'{label} Decoded', alpha=0.5, density=True)
            
            ax.set_title(f'{label} - {key} bond lengths')
            ax.set_ylabel('Density')
            ax.legend()
        
        axes[-1].set_xlabel('Bond length (Å)')
        
        plt.tight_layout()
        
        if wkdir is not None:
            plt.savefig(wkdir / f'{key}_bond_len_dist.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    return None


def plot_inversion_hist(MA, plot_data=None, fname=None, **kwargs):
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
        
        if key in MA._datasets.keys():
            dataset_inversions = MA.get_inversions(key)['dataset_inversions']
            decoded_inversions = MA.get_inversions(key)['decoded_inversions']
            data_pairs.append((dataset_inversions, decoded_inversions))
            labels.append(label)
            colors.append((dataset_color, decoded_color))
        elif key in MA._encoded.keys():
            decoded_inversions = MA.get_inversions(key)['decoded_inversions']
            data_pairs.append((None, decoded_inversions))
            labels.append(label)
            colors.append((None, decoded_color))
        else:
            raise ValueError(f"{key} not found in datasets or encoded data")
    
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
    
    return None


def plot_dope_hist(MA, plot_data=None, fname=None, refine=True, **kwargs):
    """
    Plot distributions of DOPE scores of the chosen datasets.

    :param MolearnAnalysis MA: A MolearnAnalysis object with datasets loaded.
    :param list plot_data: A list of tuples containing the data to plot. Each tuple should contain
                           the key to a original/encoded dataset in the MolearnAnalysis object, a label for the legend,
                           and a colour for the decoded data plot.
                           Format: [(key, label, colour), ...]
    :param Path fname: The directory to save the plots.
    :param bool refine: if True, refine structures before calculating DOPE score
    :param dict kwargs: Additional keyword arguments to pass to plt.savefig.

    :return: None
    """
    
    data_pairs = []
    labels = []
    color_pairs = []
    
    for key, label, decoded_color in plot_data:
        dataset_color = "gray"  # Set dataset distribution as gray
        if key in MA._datasets.keys():
            dataset_dope = MA.get_all_dope_score(MA.get_dataset(key), refine=refine)
            decoded_dope = MA.get_all_dope_score(MA.get_decoded(key), refine=refine)
        elif key in MA._encoded.keys():
            dataset_dope, dataset_color = None, None        
            decoded_dope = MA.get_all_dope_score(MA.get_decoded(key), refine=refine)
        else:
            raise ValueError(f"Dataset with key {key} not found in MolearnAnalysis object.")
        
        data_pairs.append((dataset_dope, decoded_dope))
        labels.append(label)
        color_pairs.append((dataset_color, decoded_color))

    # Plotting
    data = []
    palette = []    
    for i, (data_pair, label, color_pair) in enumerate(zip(data_pairs, labels, color_pairs)):
        if data_pair[0] is not None:
            data.extend([data_pair[0], data_pair[1]])
            palette.extend(color_pair)
        else:
            data.append([data_pair[1], data_pair[1]])
            palette.extend([decoded_color, decoded_color])


    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.violinplot(data=data,
                   split=True, inner="quart", 
                   palette=palette, 
                   ax=ax, gap=-0.05, cut=0, width=0.9, bw=0.2,
                   native_scale=True, dodge=False)
        
    legend_patches = [mpatches.Patch(color=dataset_color, label="Dataset")]
    ax.legend(handles=legend_patches, loc='upper right')

    ax.set_ylabel("DOPE score")
    ax.set_xticks(np.arange(len(labels))*2+0.5)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title('Distribution of DOPE Scores')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot if wkdir is provided
    if fname:
        plt.savefig(fname, bbox_inches='tight', **kwargs)
    plt.show()

    return None


def plot_rmsd_hist(MA, plot_data=None, fname=None, refine=True, **kwargs):
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
        plt.savefig(fname, bbox_inches='tight', **kwargs)
    plt.show()

    return None


def plot_network_rmsd_surface(MA, plot_data=None, cmap='gist_heat_r', fname=None, **kwargs):
    """
    Plot the latent landscape colored by the network RMSD, the RMSD between 
    the decoded grid and the grid encoded and decoded again. 

    Parameters
    ----------
    MA : MolearnAnalysis
        A MolearnAnalysis object containing the latent grid and chirality surface.
    plot_data : list of tuple
        A list of tuples specifying the data to plot. Each tuple should contain:
        - key (str): The key to the data in the MolearnAnalysis object.
        - label (str): A label for the legend.
        - colour (str): The color for the plot.
        - plot_type (str): The type of plot ('scatter' or 'kde').
    cmap : str
        The name of the Matplotlib colormap to use.
    fname : str
        The filename to save the plots.
    **kwargs : dict
        Additional keyword arguments to pass to plt.savefig
    """

    assert 'Network_RMSD' in MA.surfaces.keys(), "Run MA.scan_error first"

    cmap = mpl.cm.get_cmap(name=cmap)
    data = MA.surfaces['Network_RMSD']
    xvals = np.append(MA.xvals, (2*MA.xvals[-1] - MA.xvals[-2]))
    yvals = np.append(MA.yvals, (2*MA.yvals[-1] - MA.yvals[-2]))

    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(xvals, yvals, data, cmap=cmap) 

    legend_handles = []
    if plot_data is not None:
        for key, label, colour, plot_type in plot_data:
            if plot_type == 'scatter':
                ax.scatter(x=MA._encoded[key].squeeze()[:,0],
                        y=MA._encoded[key].squeeze()[:,1],
                        marker='o', s=1,
                        c=colour, label=label)
                legend_handles.append(mpl.lines.Line2D([0], [0], color=colour, linestyle='', marker='o', markersize=10, label=label))
            elif plot_type == 'kde':
                sns.kdeplot(x=MA._encoded[key].squeeze()[:,0],
                            y=MA._encoded[key].squeeze()[:,1],
                            levels=7, color=colour,
                            ax=ax, label=label)
                legend_handles.append(mpl.lines.Line2D([0], [0], color=colour, lw=2, label=label))
            else:
                pass
        ax.legend(handles=legend_handles, loc='upper right')

    ax.set_xlim(MA.xvals.min(), MA.xvals.max())
    ax.set_ylim(MA.yvals.min(), MA.yvals.max())
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')
    ax.grid(False)
    ax.set_aspect('equal')
    
    cbar_ax = fig.add_axes([ax.get_position().x1+0.02,
                            ax.get_position().y0,
                            0.02,
                            ax.get_position().height])
    cb = fig.colorbar(mesh, cax=cbar_ax)

    cb.ax.tick_params(left=False, right=True)
    cb.ax.set_ylabel('RMSD (Å)')

    if fname is not None:
        plt.savefig(fname, **kwargs)
    plt.show()

    return None


def plot_dope_surface(MA, refine=True, truncate_at=None, plot_data=None, cmap='gist_heat_r', fname=None, **kwargs):
    """
    Plot the latent landscape colored by decoded DOPE scores.

    Parameters
    ----------
    MA : MolearnAnalysis
        A MolearnAnalysis object containing the latent grid and chirality surface.
    refine : bool
        Whether to plot the refined or unrefined DOPE scores.
    truncate_at : float
        The maximum value to plot in the color map.
    plot_data : list of tuple
        A list of tuples specifying the data to plot. Each tuple should contain:
        - key (str): The key to the data in the MolearnAnalysis object.
        - label (str): A label for the legend.
        - colour (str): The color for the plot.
        - plot_type (str): The type of plot ('scatter' or 'kde').
    cmap : str
        The name of the Matplotlib colormap to use.
    fname : str
        The filename to save the plots.
    **kwargs : dict
        Additional keyword arguments to pass to plt.savefig
    """
    if refine==True:
        key='DOPE_refined'
    elif refine==False:
        key='DOPE_unrefined'
    assert key in MA.surfaces.keys(), "Run MA.scan_dope() first"

    cmap = mpl.cm.get_cmap(name=cmap)
    data = MA.surfaces[key]
    xvals = np.append(MA.xvals, (2*MA.xvals[-1] - MA.xvals[-2]))
    yvals = np.append(MA.yvals, (2*MA.yvals[-1] - MA.yvals[-2]))
    cmap.set_over(cmap(1.0))

    if truncate_at is None:
        truncate_at = data.max()

    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(xvals, yvals, data, vmin=data.min(), vmax=truncate_at, cmap=cmap) 

    legend_handles = []
    if plot_data is not None:
        for key, label, colour, plot_type in plot_data:
            if plot_type == 'scatter':
                ax.scatter(x=MA._encoded[key].squeeze()[:,0],
                        y=MA._encoded[key].squeeze()[:,1],
                        marker='o', s=1,
                        c=colour, label=label)
                legend_handles.append(mpl.lines.Line2D([0], [0], color=colour, linestyle='', marker='o', markersize=10, label=label))
            elif plot_type == 'kde':
                sns.kdeplot(x=MA._encoded[key].squeeze()[:,0],
                            y=MA._encoded[key].squeeze()[:,1],
                            levels=7, color=colour,
                            ax=ax, label=label)
                legend_handles.append(mpl.lines.Line2D([0], [0], color=colour, lw=2, label=label))
            else:
                pass
        ax.legend(handles=legend_handles, loc='upper right')

    ax.set_xlim(MA.xvals.min(), MA.xvals.max())
    ax.set_ylim(MA.yvals.min(), MA.yvals.max())
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')
    ax.grid(False)
    ax.set_aspect('equal')
    
    cbar_ax = fig.add_axes([ax.get_position().x1+0.02,
                            ax.get_position().y0,
                            0.02,
                            ax.get_position().height])
    cb = fig.colorbar(mesh, cax=cbar_ax)

    cb.ax.tick_params(left=False, right=True)
    cb.ax.set_ylabel('DOPE score')

    if fname is not None:
        plt.savefig(fname, **kwargs)
    plt.show()

    return None


def plot_inversion_surface(MA, plot_data=None, levels=10, cmap='gist_heat_r', fname=None, **kwargs):
    """
    Plot the latent landscape colored by the number of D-amino acids.

    .. note::
        The colorbar tick labels are incorrect when ``levels`` is odd.

    Parameters
    ----------
    MA : MolearnAnalysis
        A MolearnAnalysis object containing the latent grid and chirality surface.
    plot_data : list of tuple
        A list of tuples specifying the data to plot. Each tuple should contain:
        
        - key (str): The key to the data in the MolearnAnalysis object.
        - label (str): A label for the legend.
        - colour (str): The color for the plot.
        - plot_type (str): The type of plot ('scatter' or 'kde').

    levels : int
        The number of levels to plot in the colormap.
    cmap : str
        The name of the Matplotlib colormap to use.
    fname : str
        The filename to save the plots.
    **kwargs : dict
        Additional keyword arguments to pass to plt.savefig
    """

    assert 'Chirality' in MA.surfaces.keys(), "Run scan_ca_chirality first"

    cmap = mpl.cm.get_cmap(name=cmap, lut=levels+1)
    cmap.set_under(cmap(0))
    cmap.set_bad(cmap(0))
    cmap.set_over(cmap(1))

    data = MA.surfaces['Chirality']
    masked_data = np.ma.masked_where(data == 0, data)
    xvals = np.append(MA.xvals, (2*MA.xvals[-1] - MA.xvals[-2]))
    yvals = np.append(MA.yvals, (2*MA.yvals[-1] - MA.yvals[-2]))

    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(xvals, yvals, masked_data, vmin=0, vmax=levels+1, cmap=cmap)

    legend_handles = []
    if plot_data is not None:
        for key, label, colour, plot_type in plot_data:
            if plot_type == 'scatter':
                ax.scatter(x=MA._encoded[key].squeeze()[:,0],
                        y=MA._encoded[key].squeeze()[:,1],
                        marker='o', s=1,
                        c=colour, label=label)
                legend_handles.append(mpl.lines.Line2D([0], [0], color=colour, linestyle='', marker='o', markersize=10, label=label))
            elif plot_type == 'kde':
                sns.kdeplot(x=MA._encoded[key].squeeze()[:,0],
                            y=MA._encoded[key].squeeze()[:,1],
                            levels=7, color=colour,
                            ax=ax, label=label)
                legend_handles.append(mpl.lines.Line2D([0], [0], color=colour, lw=2, label=label))
            else:
                pass
        ax.legend(handles=legend_handles, loc='upper right')

    ax.set_xlim(MA.xvals.min(), MA.xvals.max())
    ax.set_ylim(MA.yvals.min(), MA.yvals.max())
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')
    ax.grid(False)
    ax.set_aspect('equal')

    cbar_ax = fig.add_axes([ax.get_position().x1+0.02,
                            ax.get_position().y0,
                            0.02,
                            ax.get_position().height])
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

    return None