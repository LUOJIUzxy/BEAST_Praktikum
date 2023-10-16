import sys
from argparse import ArgumentParser

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    print('Failed to load dependencies. Please ensure that seaborn, matplotlib, and pandas can be loaded.', file=sys.stderr)
    exit(-1)

if __name__ == '__main__':

    parser = ArgumentParser(description='Generate performance charts for ci stage.')
    parser.add_argument('--performance-data', type=str, required=True)
    parser.add_argument('--reference-data', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)

    result = parser.parse_args()

    # Read data
    data = pd.read_fwf(result.performance_data, delimiter=' |')
    ref_data = pd.read_fwf(result.reference_data, delimiter=' |')

    fig = plt.figure(figsize=(16, 6))

    # Measured performance
    axes = plt.subplot(1, 2, 1)
    #print(data)
    sns.lineplot(data=data, x="Data size", y="Latency [ns]", hue="Sum Type", marker='o')
    axes.set_xscale('log', basex=2)
    axes.set_title('Measured Latency [ns]')

    # Reference performance
    axes = plt.subplot(1, 2, 2)

    sns.lineplot(data=ref_data, x="Data size", y="Latency [ns]", hue="Sum Type", marker='o', axes=axes)
    axes.set_xscale('log', basex=2)
    axes.set_title('Reference Latency [ns]')

    # Max ylim
    max_lim = max([ax.get_ylim()[1] for ax in fig.get_axes()])
    for ax in fig.get_axes():
        ax.set_ylim([0, max_lim])

    fig.tight_layout()

    plt.savefig(result.output_file)
    print(f"Wrote output to {result.output_file}")

