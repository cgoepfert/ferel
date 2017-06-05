import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

font = {'family': 'normal', 'weight': 'bold', 'size': 20}

matplotlib.rc('font', **font)
bmap = sns.color_palette("Set2", 5)
sns.set(style='ticks', palette='Set2')
sns.despine()


def handle_axes(figure=None, axes=None):
    """
    Create figure and/or axes if necessary.
    """
    if axes is None:
        if figure is None:
            figure = plt.figure()
        axes = figure.add_subplot(1, 1, 1)
    return axes


def scatter_2d_hyperplane(data,
                          labels,
                          dim_1,
                          dim_2,
                          normal,
                          bias,
                          figure=None,
                          axes=None):
    """
    Plot a 2-dimensional representation of data together with a hyperplane.
    """
    # sns stuff, not sure where to appropriately put it.
    sns.set(
        context='notebook',
        style='white',
        palette=sns.color_palette("BrBG", 7),
        font='sans-serif',
        font_scale=1,
        color_codes=False,
        rc=None)
    current_palette = sns.color_palette()

    # Determine bounding x coordinates for the hyperplane.
    x = np.array([np.min(data[:, dim_1]), np.max(data[:, dim_1])])

    # Make sure the second normal vector entry is not zero for easy plotting.
    if abs(normal[dim_2]) < 1e-4:
        normal[dim_2] = 1e-4

    # Calculate y coordinates for the hyperplane.
    hyperplane = (bias - normal[dim_1] * x) / normal[dim_2]

    margin_left = (1 + bias - normal[dim_1] * x) / normal[dim_2]
    margin_right = (-1 + bias - normal[dim_1] * x) / normal[dim_2]

    # Set colors according to margin intrusions.
    xis = np.maximum(0, 1 - labels * (np.dot(data, normal) - bias))
    colors = np.zeros(labels.size)
    colors[labels == -1] = 4
    colors[np.logical_and(labels == -1, xis > 0)] = 5
    colors[np.logical_and(labels == -1, xis > 1)] = 6
    colors[labels == 1] = 2
    colors[np.logical_and(labels == 1, xis > 0)] = 1
    colors[np.logical_and(labels == 1, xis > 1)] = 0
    ax = handle_axes(figure, axes)
    ax.scatter(
        data[:, dim_1],
        data[:, dim_2],
        c=np.asarray(current_palette)[colors.astype(int)])
    ax.plot(x, hyperplane)
    ax.plot(x, margin_left)
    ax.plot(x, margin_right)
    epsilon = 0.1
    ax.axis([
        np.min(data[:, dim_1]) - epsilon,
        np.max(data[:, dim_1]) + epsilon,
        np.min(data[:, dim_2]) - epsilon,
        np.max(data[:, dim_2]) + epsilon
    ])
    return ax


def plot_intervals(ranges, figure=None, axes=None):
    """
    Plot a visualization of relevance intervals.
    """
    # Setup figure and axes.
    ax = handle_axes(figure, axes)

    # Prepare intervals.
    n_intervals = len(ranges)
    index = np.arange(n_intervals) + 1
    upper_vals = ranges[:, 1]
    lower_vals = ranges[:, 0]

    # Prepare colors.
    colors = np.asarray(sns.color_palette("Set2", n_intervals))

    # Plot lower bounds.
    ax.bar(
        index,
        lower_vals,
        tick_label=index,
        align="center",
        linewidth=1.3,
        color=colors * 0.66)

    # Plot upper bounds by stacking them on top of the lower bounds.
    ax.bar(
        index,
        upper_vals - lower_vals,
        bottom=lower_vals,
        tick_label=index,
        align="center",
        linewidth=1.3,
        color=colors)

    # Annotate plot.
    plt.ylabel('relevance', fontsize=19)
    plt.xlabel('feature', fontsize=19)

    # Return axes.
    return ax
