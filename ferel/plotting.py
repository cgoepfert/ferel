import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

sns.set(style='whitegrid')


def handle_axes(figure=None, axes=None):
    """
    Create figure and/or axes if necessary.
    """
    if axes is None:
        if figure is None:
            figure = plt.figure(figsize=[6, 6])
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

    # Determine bounding x coordinates for the hyperplane.
    x = np.array([np.min(data[:, dim_1]), np.max(data[:, dim_1])]) * 1.1

    # Make sure the second normal vector entry is not zero for easy plotting.
    if abs(normal[dim_2]) < 1e-4:
        normal[dim_2] = 1e-4

    # Calculate y coordinates for the hyperplane.
    hyperplane = (bias - normal[dim_1] * x) / normal[dim_2]

    margin_left = (1 + bias - normal[dim_1] * x) / normal[dim_2]
    margin_right = (-1 + bias - normal[dim_1] * x) / normal[dim_2]

    # Set colors according to margin intrusions.
    xis = np.maximum(0, 1 - labels * (np.dot(data, normal) - bias))
    color_index = np.zeros(labels.size)
    color_index[labels == -1] = 4
    color_index[np.logical_and(labels == -1, xis > 0)] = 5
    color_index[np.logical_and(labels == -1, xis > 1)] = 6
    color_index[labels == 1] = 2
    color_index[np.logical_and(labels == 1, xis > 0)] = 1
    color_index[np.logical_and(labels == 1, xis > 1)] = 0
    current_palette = sns.color_palette("PuOr", 7)
    colors = np.asarray(current_palette)[color_index.astype(int)]

    # Get axes to plot to.
    ax = handle_axes(figure, axes)

    # Scatter data.
    ax.scatter(data[:, dim_1], data[:, dim_2], c=colors)

    # Draw hyperplane and margin.
    ax.plot(x, hyperplane, c='grey')
    ax.plot(x, margin_left, c=current_palette[0])
    ax.plot(x, margin_right, c=current_palette[6])

    # Zoom in on the data so that it fills the plot but with equal axis.
    epsilon = 0.1
    x_min = np.min(data[:, dim_1]) - epsilon
    x_max = np.max(data[:, dim_1]) + epsilon
    y_min = np.min(data[:, dim_2]) - epsilon
    y_max = np.max(data[:, dim_2]) + epsilon
    xy_min = max(x_min, y_min)
    xy_max = min(x_max, y_max)
    ax.axis([xy_min, xy_max, xy_min, xy_max])  # This may not show all data.

    # Return axes.
    return ax


def plot_intervals(ranges, weights=None, labels=None, figure=None, axes=None):
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

    pal = sns.color_palette()

    # Plot lower bounds.
    lower_bars = ax.bar(
        index,
        lower_vals,
        tick_label=index,
        align="center",
        linewidth=1.3,
        label='lower bound',
        color=pal[0])

    # Plot upper bounds by stacking them on top of the lower bounds.
    upper_bars = ax.bar(
        index,
        upper_vals - lower_vals,
        bottom=lower_vals,
        tick_label=index,
        align="center",
        linewidth=1.3,
        label='upper bound',
        color=pal[1])

    # Annotate plot.
    ax.set_ylabel('relevance')
    ax.set_xlabel('feature')

    # Scatter weights.
    if weights is not None:
        if labels is None:
            labels = [None for w in weights]
        for w, l in zip(weights, labels):
            ax.scatter(index, w, zorder=2, label=l)

    # Draw legend.
    ax.legend()

    # Return axes.
    return ax
