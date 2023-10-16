import sys
from argparse import ArgumentParser
import re
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    print('Failed to load dependencies. Please ensure that seaborn, matplotlib, and pandas can be loaded.', file=sys.stderr)
    exit(-1)

if __name__ == '__main__':

    parser = ArgumentParser(description='Generate performance charts for ci stage.')
    parser.add_argument('--performance-data', action='append', default=[], required=True)
    parser.add_argument('--reference-data', action='append', default=[], required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--hue-column', type=str, default='Threads', help='Seaborn hue column used for data frame.')

    args = parser.parse_args()

    perf_data_file = open(args.performance_data[0], 'r')
    perf_data = { "Data size": [], "MFLOPS": []}
    ref_data_file = open(args.reference_data[0], 'r')
    ref_data = {"Data size": [], "MFLOPS": []}

    for line in perf_data_file:
      matchData = re.match("MFLOPS\(N=\s*(\d+)\): (\d+\.\d+)", line)
      perf_data["Data size"].append(int(matchData.group(1)))
      perf_data["MFLOPS"].append(float(matchData.group(2)))

    mesured_data = pd.DataFrame(perf_data)

    for line in ref_data_file:
      matchData_ref = re.match("MFLOPS\(N=\s*(\d+)\): (\d+\.\d+)", line)
      ref_data["Data size"].append(int(matchData_ref.group(1)))
      ref_data["MFLOPS"].append(float(matchData_ref.group(2)))

    reference_data = pd.DataFrame(ref_data)


    fig = plt.figure(figsize=(16, 6))

    # Measured performance
    axes = plt.subplot(1, 2, 1)
    sns.lineplot(data=mesured_data, x="Data size", y="MFLOPS", marker='o')
    axes.set_title('Measured Performance')

    axes = plt.subplot(1, 2, 2)
    sns.lineplot(data=reference_data, x="Data size", y="MFLOPS", marker='o')
    axes.set_title('Reference Performance')


    # Max ylim
    max_lim = max([ax.get_ylim()[1] for ax in fig.get_axes()])
    for ax in fig.get_axes():
        ax.set_ylim([0, max_lim])

    plt.xticks(range(0, 4001, 500))

    fig.tight_layout()

    plt.savefig(args.output_file)
    print(f"Wrote output to {args.output_file}")
