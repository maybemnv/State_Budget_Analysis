import pandas as pd
from matplotlib import pyplot as plt

def gen_line_plot(df, filename, show_averages=False):
    df = df.T
    usecols = []
    for col in df.columns:
        if "Total" != col:
            usecols .append(col)
    if not show_averages:
        df.drop("Average (%)", inplace=True)
        for col in usecols:
            plt.plot(df[col], label=col)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(filename)
        plt.xticks(rotation=45)
        plt.grid()
        plt.savefig(f"../plots/{filename}.jpg", bbox_inches='tight')
        plt.close()
    print(df.to_markdown())
    return None