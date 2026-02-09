
import os
from typing import Optional, Tuple
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.linalg import matrix_balance
from scipy.stats import mannwhitneyu
from scipy.optimize import curve_fit
import numpy as np
from scipy.spatial.distance import squareform

from config import CONFIG

from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
import seaborn as sns
from preprocessed_data import PreprocessedData

class PltConfig:
    def __init__(self, name:str):
        self.name = name
        self.itr = 0
    def save(self):
        self.itr += 1
        if CONFIG.do_logging:
            dir = CONFIG.run_dir if CONFIG.do_logging else CONFIG.out_dir
            CONFIG.logger.info(f"saving {self.name} to {dir}")
            plt.savefig(os.path.join(dir, f"{self.name}_{self.itr}.png"))

VIZ_CONFIG = {id: PltConfig(id) for id in [
    'line_chart',
    'noise',
    'heatmap',
    'heatmap_color',
    'histogram',
    'ML',
    'powerfit',
    'dendogram',
    'colormap',
    'colormap_legend',
]}

def ML_plotting(training_loss, validation_loss, validation_accuracies, num_epochs):
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), training_loss, label="Training Loss", color='blue')
    if validation_loss:
        val_epochs, val_losses = zip(*validation_loss)
        plt.plot(val_epochs, val_losses, label="Validation Loss", color='orange', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.grid(True)
    plt.legend()

    val_epochs, val_accs = zip(*validation_accuracies)
    plt.subplot(1, 2, 2)
    plt.plot(val_epochs, val_accs, label="Validation Accuracy", color='green', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy over Epochs")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    VIZ_CONFIG['ML'].save()


def histogram(data, xlabel="", title=""):
    min_size = min(data)
    max_size = max(data)
    bins = list(range(min_size, max_size + 2)) 

    plt.hist(data, bins=bins, align='left', rwidth=0.8)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(f"{title} ; max={max(data)}")
    plt.xticks(range(min_size, max_size + 1))
    #plt.show()
    VIZ_CONFIG['histogram'].save()


def powerfit(x, y, xlabel="", ylabel="", title="", dofit=None):
    def power_law(x, a, b):
        return a * np.power(x, b)
    def exponential_law (x, a, b):
        return a * np.exp(b * x)
    ymax = np.max(y)
    mask = x <= 50
    x_masked = x[mask]
    y_masked = y[mask]

    # Fit the curve
    x_fit = np.linspace(min(x_masked), max(x_masked), 2000)
    if dofit is not None and dofit['power']:
        (a_fit, b_fit), _ = curve_fit(power_law, x_masked, y_masked / ymax, p0=[1, -1])
        a_fit *= ymax
        y_fit = power_law(x_fit, a_fit, b_fit)
    if dofit is not None and dofit['exponential']:
        (a_fit_e, b_fit_e), _ = curve_fit(exponential_law, x_masked, y_masked / ymax, p0=[1, -0.5])
        a_fit_e *= ymax
        y_fit_e = exponential_law(x_fit, a_fit_e, b_fit_e)

    # Plot
    plt.scatter(x, y, label="Data", color="blue", s=2)
    if dofit is not None and dofit['power']:
        plt.plot(x_fit, y_fit, label=f"Fit: y = {a_fit:.3f} x^{b_fit:.3f}", color="red")
    if dofit is not None and dofit['exponential']:
        plt.plot(x_fit, y_fit_e, label=f"Fit: y = {a_fit_e:.3f} exp({b_fit_e:.3f} x)", color="green")
    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    #plt.show()
    VIZ_CONFIG['powerfit'].save()
    plt.close()

def dendogram_2d(all_labels: list[Tuple[PreprocessedData, PreprocessedData]], all_scores, title):
    # Unique label strings (dedupe by label text)
    unique_label_names = sorted({x.get_short_label() for pair in all_labels for x in pair})
    n = len(unique_label_names)
    label_to_idx = {name: i for i, name in enumerate(unique_label_names)}

    # Build similarity matrix (indexing by label strings)
    sim_matrix = np.zeros((n, n))
    for (a, b), s in zip(all_labels, all_scores):
        i, j = label_to_idx[a.get_short_label()], label_to_idx[b.get_short_label()]
        sim_matrix[i, j] = s
        sim_matrix[j, i] = s
    np.fill_diagonal(sim_matrix, 1.0)
    dist_matrix = 1 - sim_matrix

    # Compute linkage once and pass it into clustermap (keeps dendrogram consistent)
    linkage_mat = linkage(dist_matrix, method='average')

    # Dataframe: rows/cols correspond to unique_label_names (no manual re-ordering)
    df_sim = pd.DataFrame(sim_matrix, index=unique_label_names, columns=unique_label_names)

    sns.set(font_scale=0.6)
    g = sns.clustermap(
        df_sim,
        row_cluster=True,
        col_cluster=True,
        row_linkage=linkage_mat,
        col_linkage=linkage_mat,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        #cbar=False
    )
    # if hasattr(g, "cax") and g.cax is not None:
    #     g.cax.set_visible(False)
    #     g.cax.remove()
    # Adjust colorbar position and size
    # if hasattr(g, "cax") and g.cax is not None:
    # box = g.cax.get_position()
    # Move slightly left and up, and shrink size
    # g.cax.set_position((
    #     box.x0 - 2.0,   # shift left
    #     box.y0 + 2.0,   # shift up
    #     box.width * 0.5, # make narrower
    #     box.height * 0.5 # make shorter
    # ))

    # Build map: label string -> category (take first seen object for each label)
    label_to_cat = {}
    for (a, b) in all_labels:
        for x in (a, b):
            lbl = x.get_short_label()
            if lbl not in label_to_cat:
                label_to_cat[lbl] = x.non_rep

    # Auto-assign colors to categories
    unique_cats = sorted(set(label_to_cat.values()))
    palette = sns.color_palette("hls", len(unique_cats))
    cat_to_color = {cat: palette[i] for i, cat in enumerate(unique_cats)}

    # Helper: color tick labels based on their actual text (robust to clustermap reordering)
    def color_ticks(ticklabels):
        for txt in ticklabels:
            lbl_text = txt.get_text().strip()
            cat = label_to_cat.get(lbl_text)
            if cat is None:
                # fallback: try without stripping or set neutral color
                cat = label_to_cat.get(txt.get_text(), None)
            if cat is None:
                txt.set_color("gray")
            else:
                txt.set_color(cat_to_color[cat])

    # Apply to both axes (use actual tick labels returned by seaborn)
    color_ticks(g.ax_heatmap.get_xticklabels())
    color_ticks(g.ax_heatmap.get_yticklabels())

    # --- Put title clearly above the plot ---
    g.fig.suptitle(title, fontsize=16, y=0.95, weight="bold")

    # Manually adjust top spacing instead of tight_layout (prevents clipping)
    g.fig.subplots_adjust(top=0.90)

    # Force draw before reposition
    plt.draw()

    # Access colorbar axis directly and reposition
    cbar_ax = g.ax_cbar
    cbar_ax.set_position([0.05, 0.8, 0.02, 0.1])

    VIZ_CONFIG['dendogram'].save()
    plt.close()





def test_bar_chart(rep_scores, non_rep_scores, method:str, exp:str):
    """
    Use Matplotlib to visualize the difference between replicate scores and non-replicate scores with a bar chart
    """
    groups = [rep_scores, non_rep_scores]
    _, ax = plt.subplots(figsize=(10, 5))
    ax.set_ylim(0, 1)

    gap = 3
    x_pos_group1 = np.arange(len(groups[0]))
    x_pos_group2 = np.arange(len(groups[1])) + len(groups[0]) + gap

    ax.bar(x_pos_group1, groups[0], color='tab:blue', label='Replicates')
    ax.bar(x_pos_group2, groups[1], color='tab:orange', label='Non-replicates')

    ax.set_xticks([len(groups[0])/2 - 0.5, len(groups[0]) + gap + len(groups[1])/2 - 0.5])
    ax.set_xticklabels(['Replicates', 'Non-replicates'])

    ax.set_ylabel(f'{method} Score')
    chr = CONFIG.cfg['chr']
    ax.set_title(f'{method} + {exp} similarity at bin_size={int(CONFIG.bin_size/1e3)}kb on chr{chr}')

    plt.tight_layout()
    #plt.show()


def test_line_chart(rep_scores_lst, non_rep_scores_lst, title, labels, reps_labels, nonreps_labels, bShow=False):
    """
    Use Matplotlib to visualize the difference between replicate scores and non-replicate scores with 1D line chart.
    Returns: updated ax. Usage:
        [calls ...]
        plt.tight_layout()
        plt.show()
    """
    label_size=5
    label_stagger=.026
    stagger_order=17
    s=2

    _, ax = plt.subplots(figsize=(12, 3.5 + max(len(labels)*1.5, 3.)))
    #if len(labels) > 1:
    for i, exp in enumerate(labels):
        x_rep = np.array(rep_scores_lst[i])
        x_nonrep = np.array(non_rep_scores_lst[i])
        l = i+1

        ax.scatter(x_rep, [-l+.05] * len(x_rep), color='tab:blue', alpha=0.7, label='Replicates' if i == 0 else "", s=s)
        ax.scatter(x_nonrep, [-l+.1] * len(x_nonrep), color='tab:red', alpha=0.7, label='Non-replicates' if i == 0 else "", s=s)
        ax.text(1.05, -l, exp, verticalalignment='center')

        R, B = x_nonrep, x_rep
        score_height = 1
        def display_score(name, score, p=None):
            nonlocal score_height
            if p is None or (not np.isnan(p)):
                title_sub = f'{name}:{score:.3f}' if p is None else f'{name}:{score:.3f} p:{p:.3f}'
                CONFIG.logger.info(f'Sep. Result: {title_sub}')
                score_clamp = max(0, min(1, score))
                color = mcolors.to_hex((1 - score_clamp, score_clamp, 0))
                ax.text(1.05+.4, -l+.04*score_height, title_sub, verticalalignment='center', color=color)
                score_height += 1
        CONFIG.logger.info(f'Sep. Section: {exp}')

        verbose = CONFIG.cfg['visualization']['verbose_labels']
        if verbose and reps_labels is not None and nonreps_labels is not None:
            rep_lbls = reps_labels[i]
            nonrep_lbls = nonreps_labels[i]
            offset = 0
            for j, val in enumerate(x_rep):
                offset = 0.03 + label_stagger * (j % stagger_order)
                ax.text(val, -l - offset, str(rep_lbls[j]), fontsize=label_size, rotation=0, verticalalignment='bottom', color='darkblue')
            prev_offset = offset
            for j, val in enumerate(x_nonrep):
                offset = prev_offset + label_stagger * (j % stagger_order)
                ax.text(val, -l - offset, str(nonrep_lbls[j]), fontsize=label_size, rotation=0, verticalalignment='bottom', color='darkred')

    ax.set_yticks([])
    #if len(labels) == 1:
        #ax.set_xlim(-1, 1.3)
    ax.set_xlim(-1, 1.3)
    ax.set_xlabel("Reproducibility")
    ax.set_title(title+get_encoding_str())
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    plt.tight_layout()
    VIZ_CONFIG['line_chart'].save()
    if bShow: 
        plt.show()
    
def get_encoding_str():
    return f"\n[encoding:{CONFIG.cfg['data_adjustment']['encoding']};p:{CONFIG.cfg['data_adjustment']['encoding_p']};a={CONFIG.cfg['data_adjustment']['encoding_alpha']}]"


def test_noiselevel_analysis(scores_list, title, labels, reps_labels):
    plt.figure(figsize=(10, 6))

    for scores, method_label, noise_labels in zip(scores_list, labels, reps_labels):
        noise_values = [float(label.split('=')[1]) for label in noise_labels]
        plt.plot(noise_values, scores, marker='o', label=method_label, markersize=2)

    plt.title(title)
    plt.xlabel(f"Noise Level {get_encoding_str()}")
    plt.ylabel("Score")
    plt.legend(title="Method")
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    VIZ_CONFIG['noise'].save()
    


def matrix_heatmap(tensor : np.ndarray, title="", res=1, aspect=1, vmax=1, bAdjustRes=True, bShow=True, BI=None):
    """
    Prints out a heatmap of a 2D tensor for visualization/debugging. 
    Input:  res: 0 < res <= 1. Lower values mean larger binning. Descrease if visualization is too sparse. 
            aspect: Scales the Y axis. 
            vmax: 0 < vmax <= 1. Sets upper bound on intensity. vmax=1 gives no effect. 
            bAdjustRes: If false, plot the raw matrix without resizing. 
    """
    if vmax < 1:
        tensor = np.clip(tensor, a_min=None, a_max=tensor.max() * vmax)
    n, m = tensor.shape
    if bAdjustRes:
        res *= CONFIG.bin_size / 1e6
        n, m = int(n * res), int(m * res * aspect)

    y_idx, x_idx = np.nonzero(tensor) 
    values = tensor[y_idx, x_idx]

    heatmap, _, _ = np.histogram2d(
        x_idx, y_idx,
        bins=[m, n], 
        weights=values
    )
    if bShow:
        plt.imshow(heatmap, origin='upper', cmap='hot', interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        #plt.tight_layout()
        #plt.show()
        VIZ_CONFIG['heatmap'].save()
    else:
        return heatmap, title
    

def heatmap_multi(tensors, titles=None, ncols=2, res=1, aspect=1, vmax=1, bAdjustRes=False, BI=[None]):
    import matplotlib.pyplot as plt
    
    has_intensity = BI[0] is not None
    n_heatmaps = len(tensors)
    nrows = (n_heatmaps + ncols - 1) // ncols
    #fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows+1))
    #axes = axes.flatten() if n_heatmaps > 1 else [axes]
    fig, axes = plt.subplots(
        nrows=2 * nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows + 1),
        gridspec_kw={"height_ratios": [5, 1] * nrows}  # ADDED
    )
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    axes = axes.reshape((2 * nrows, ncols))  # ADDED

    for i, tensor in enumerate(tensors):
        row = (i // ncols) * 2 
        col = i % ncols       
        heatmap, title = matrix_heatmap(
            tensor.toarray(), 
            title=titles[i] if titles else "", 
            res=res, 
            aspect=aspect, 
            vmax=vmax, 
            bAdjustRes=bAdjustRes, 
            bShow=False,
            BI=BI[i] if has_intensity else None
        )
        ax = axes[row, col]
        im = ax.imshow(heatmap, origin='upper', cmap='hot', interpolation='nearest')
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

        if has_intensity and BI[i] is not None:
            ax_bi = axes[row + 1, col]
            bi_vec = BI[i].toarray().flatten() if hasattr(BI[i], 'toarray') else np.asarray(BI[i])
            ax_bi.bar(range(len(bi_vec)), bi_vec, color='red', width=1.0, align='edge')
            ax_bi.set_title("Binding Intensity")
            ax_bi.set_xlim(0, len(bi_vec)) 
        else:
            axes[row + 1, col].axis('off')

    # for j in range(i+1, len(axes)):
    #     axes[j].axis('off')
    if has_intensity:
        total_plots = 2 * ((n_heatmaps + ncols - 1) // ncols) * ncols
        for j in range(i + 1, total_plots // 2):
            try:
                axes[2*j, col].axis('off')       # ADDED
                axes[2*j + 1, col].axis('off')   # ADDED
            except Exception as e:
                CONFIG.logger.error(e)

    plt.tight_layout()
    #plt.show(block=False)
    VIZ_CONFIG['heatmap'].save()


# Depricated
def print_hypergraph_clique(hypergraph, file_name="", genome_range:Optional[Tuple[int, int]]=None):
    """
    Prints a HiC-style representation of the clique-expansion of the hypergraph. Hypergraph must be encoded with [encode_map]. 
    """
    
    acc = {}
    ks = []
    for hyperedge in hypergraph.values():
        for i in range(len(hyperedge)):
            for j in range(i + 1, len(hyperedge)):
                x, y = hyperedge[i], hyperedge[j] 

                bias = 1 + abs(x-y) # scale up distant interations, don't multiply by bin size since we normalize
                k1, k2 = (x,y), (y,x)
                ks.append(int(x))
                acc[k1] = acc.get(k1, 0) + bias
                acc[k2] = acc.get(k2, 0) + bias

    if genome_range is None:
        nbins = int(max(ks))
        A = 0
        B = nbins
    else:
        A = genome_range[0] // CONFIG.bin_size
        B = genome_range[1] // CONFIG.bin_size
        nbins = int(B - A)
    
    X = np.array([k[0] for k in acc.keys()])
    Y = np.array([k[1] for k in acc.keys()])
    Z = np.array(list(acc.values()))

    heatmap, _, _ = np.histogram2d(X, Y, bins=nbins, weights=Z, range=[[A, B], [A, B]])
    heatmap, _ = matrix_balance(heatmap) # Balance matrix
    plt.imshow(
        heatmap, origin='lower', extent=[A, B, A, B],
        cmap='inferno', aspect='auto', vmax=np.max(heatmap)
    )
    plt.title(f"Clique expansion at {CONFIG.bin_size*1e-3} kb for \nf{file_name}")
    plt.colorbar(label="Density")
    #plt.show()
    