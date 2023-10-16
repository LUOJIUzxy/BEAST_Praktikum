import argparse
import pandas as pd
import matplotlib.style
import matplotlib.pyplot as plt

matplotlib.style.use('seaborn')

frequency = {'instructions-ice2.csv': 3.5,
             'instructions-rome2.csv': 3.4,
             'instructions-thx1.srve.csv': 2.5,
             'instructions-cs2.csv': 1.8}


def latency(df):
    return df.loc[df['chains'] == 1].sort_values(by=['op', 'dtype']).reset_index(drop=True)


def throughput(df):
    return df.loc[df.groupby(['op', 'dtype']).time.idxmin()].sort_values(by=['op', 'dtype']).reset_index(drop=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='make latency/throughput table and barplot from data')
    parser.add_argument('-f,' '--file', dest='infile',
                        help='input csv file', required=True)
    parser.add_argument('-o,' '--out', dest='outfile',
                        help='output file', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.infile)
    df['cycles'] = round(df['time'] * frequency[args.infile], 1)

    df_l = latency(df).set_index(['op', 'dtype'])
    df_t = throughput(df).set_index(['op', 'dtype'])

    fig, ax = plt.subplots(figsize=(10, 6))

    print(df_l)
    print(df_t)

    df_l.plot.bar(stacked=True, y='cycles', ax=ax,
                  color='darkorange', label='non-pipelined (latency)', rot=23)
    df_t.plot.bar(stacked=True, y='cycles', ax=ax,
                  color='darkgreen', label='pipelined (throughput)', rot=23)

    for container in ax.containers:
        ax.bar_label(container)

    ax.set_xlabel('Operation, Data Type')
    ax.set_ylabel('CPI')
    ax.set_title(args.infile)

    plt.savefig(args.outfile, bbox_inches = 'tight')
