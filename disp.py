from copy import deepcopy
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def set_font_size(ax, font_size, legend_font_size=None):
    """Set font_size of all axis text objects to specified value."""
    
    axs = ax
    try:
        iter(axs)
    except TypeError:
        axs = np.array(ax)
    for ax in axs.flatten():
        texts = [ax.title, ax.xaxis.label, ax.yaxis.label] + \
            ax.get_xticklabels() + ax.get_yticklabels()

        for text in texts:
            text.set_fontsize(font_size)

        if ax.get_legend():
            if not legend_font_size:
                legend_font_size = font_size
            for text in ax.get_legend().get_texts():
                text.set_fontsize(legend_font_size)
            

def set_plot(ax, x_lim=None, y_lim=None, x_ticks=None, y_ticks=None, x_tick_labels=None, y_tick_labels=None,
        x_label=None, y_label= None, title=None, font_size=12):
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)
    if y_tick_labels is not None:
        ax.set_yticklabels(y_tick_labels)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    if font_size is not None:
        set_font_size(ax, font_size)
        
        
def set_n_x_ticks(ax, n, x_min=None, x_max=None):
    x_ticks = ax.get_xticks()
    
    x_min = np.min(x_ticks) if x_min is None else x_min
    x_max = np.max(x_ticks) if x_max is None else x_max
    
    ax.set_xticks(np.linspace(x_min, x_max, n))
    
    
def set_n_y_ticks(ax, n, y_min=None, y_max=None):
    y_ticks = ax.get_yticks()
    
    y_min = np.min(y_ticks) if y_min is None else y_min
    y_max = np.max(y_ticks) if y_max is None else y_max
    
    ax.set_yticks(np.linspace(y_min, y_max, n))

    
def set_color(ax, color, box=False):
    """Set colors on all parts of axis."""

    if box:
        ax.spines['bottom'].set_color(color)
        ax.spines['top'].set_color(color)
        ax.spines['left'].set_color(color)
    ax.spines['right'].set_color(color)

    ax.tick_params(axis='x', color=color)
    ax.tick_params(axis='y', color=color)

    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_color(color)

    ax.title.set_color(color)
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)
    
    
def get_spaced_colors(cmap, n, step):
    """step from 0 to 1"""
    cmap = cm.get_cmap(cmap)
    return cmap((np.arange(n, dtype=float)*step)%1)


def get_ordered_colors(cmap, n, lb=0, ub=1):
    cmap = cm.get_cmap(cmap)
    return cmap(np.linspace(lb, ub, n))

    
def fast_fig(n_ax, ax_size, fig_w=15):
    """Quickly make figure and axes objects from number of axes and ax size (h, w)."""
    n_col = int(round(fig_w/ax_size[1]))
    n_row = int(np.ceil(n_ax/n_col))
    
    fig_h = n_row*ax_size[0]
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(fig_w, fig_h), tight_layout=True, squeeze=False)
    return fig, axs.flatten()
