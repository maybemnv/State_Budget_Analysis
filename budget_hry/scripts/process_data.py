import pandas as pd
import sys
import os
import gen_plot
def get_percentage_df(df_raw):
    df = df_raw.copy()
    usecols = []
    for col in df.columns:
        try:
            df[col] /= df[col].sum()
            df[col] *= 100
            df[col] = (df[col]*100).astype(int)/100
            usecols.append(col)
        except Exception as e:
            print(col)
            print(e)
            exit()
    df["Average (%)"] = df[usecols].mean(axis=1)
    return df

def get_totals_and_average_perc(df_raw):
    df = df_raw.copy()
    df2 = get_percentage_df(df_raw)["Average (%)"]
    df = df.join(df2)
    df.loc["Total"] = df.sum()
    return df

if __name__ == "__main__":

    filename = sys.argv[1]
    df = pd.read_csv(f"../raw_data/{filename}.csv")
    df.set_index("Title", inplace=True)
    for col in df.columns:
        df[col] = df[col].astype(float)
    print(df.head().to_markdown())
    print(df.dtypes)
    x = int(input())

    get_totals_and_average_perc(df).to_csv(f"../processed_data/{filename}_total.csv")
    get_percentage_df(df).to_csv(f"../processed_data/{filename}_percentage.csv")

    gen_plot.gen_line_plot(get_totals_and_average_perc(df), f"{filename}_total", False)
    gen_plot.gen_line_plot(get_percentage_df(df), f"{filename}_percentage", False)