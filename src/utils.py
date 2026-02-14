import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import ceil
from pandas import DataFrame
from matplotlib.pyplot import figure


palette = ['#002D62', '#FDBB30', "#086432", "#4E0068", "#7F092A"]


def plot_feature_rate(dataframe: DataFrame, features: list[str]) -> figure:
    n_cols = min(3, len(features))
    n_rows = ceil(len(features) / n_cols)
    sns.set_style('darkgrid')
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes)

    for i, feature in enumerate(features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[col]

        sns.countplot(data=dataframe, x=feature, ax=ax, palette=palette, hue=feature, legend=False)
        ax.yaxis.set_visible(False)
        ax.set_xlabel('')
        ax.set_title(f'{feature} Rate')
        
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        if set(labels) <= {'0', '1'}:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['No', 'Yes'])

        total = len(dataframe)

        for p in ax.patches:
            height = p.get_height()
            percentage = height / total * 100
            ax.annotate(
                f'{percentage:.1f}%',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', 
                va='bottom',
                fontweight='bold',
                color='black'
            )

    return fig


def plot_feature_distribuition(dataframe: DataFrame, features: list[str], target=None) -> figure:
    n_cols = min(3, len(features))
    n_rows = ceil(len(features) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 3.7 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    sns.set_style('darkgrid')

    for i, feature in enumerate(features):
        ax = axes[i]
        sns.histplot(data=dataframe, x=feature, ax=ax, palette=palette if target is not None else palette, color=palette[0] if target is None else None, kde=True, stat='proportion', hue=target, legend=True)

        if target is not None:
            sns.move_legend(ax, "upper right")

    return fig


def plot_feature_box_plot(dataframe: DataFrame, features: list[str], target=None) -> figure:
    n_cols = min(3, len(features))
    n_rows = ceil(len(features) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    sns.set_style('darkgrid')

    for i, feature in enumerate(features):
        ax = axes[i]
        sns.boxplot(data=dataframe, x=target, y=feature, ax=ax, palette=palette)

    return fig


def plot_feature_target_rate(dataframe: DataFrame, features: list[str], target=None) -> figure:
    n_cols = min(3, len(features))
    n_rows = ceil(len(features) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes)
    sns.set_style('darkgrid')

    for i, feature in enumerate(features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[col]

        sns.barplot(data=dataframe, x=feature, y=target , ax=ax, color="#002D62", ci=None)
        ax.yaxis.set_visible(False)
        ax.set_xlabel('')
        ax.set_title(f'Taxa de Churn por {feature} (%)')

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f'{height*100:.1f}%',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', 
                va='bottom',
                fontweight='bold',
                color='black'
            )

    return fig