import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from plotnine import *
from plotnine.data import *
import seaborn as sns

def getSurprisalDataframe(df_labs: pl.DataFrame,
                       hour_buckets = 1,
                       labs= [
                           'Sodium',
                           'PaO2',
                           'Albumin',
                           'Hemoglobin',
                           'Troponin I'
                           ]):
    """
    Helper function for calculating surprisal values for all (lab, hour_ordered) combinations from raw lab orders
    df_labs: a polars dataframe containing "lab" and "hour_ordered" columns
    hour_ordered = variable defining one hour block lab is ordered in. Can be engineered from date/time ordered
    hour_buckets: int, size of window for analysis
    returns a polars dataframe containing the surprisal values by lab and hour_ordered
    """
    
    df_cross = pl.DataFrame(
        dict(lab=labs)
    ).join(
        pl.DataFrame(dict(hour_ordered=pl.int_range(24, eager=True, step=hour_buckets))),
        how="cross"
    )

    df_labs = df_labs.filter(
        pl.col("lab").is_in(labs)
    )

    df_labs_by_dept = df_labs.group_by(
        "lab", pl.col("hour_ordered").floordiv(hour_buckets).mul(hour_buckets)
    ).agg(
        count_by_lab_hour=pl.len()
    ).join(
        df_cross, on=["hour_ordered","lab"], how="outer", coalesce=True
    ).fill_null(
        1
    ).with_columns(
        prob_by_lab_hour=pl.col("count_by_lab_hour").truediv(df_labs.shape[0])
    ).with_columns(
        surprisal_by_lab_hour=pl.col("prob_by_lab_hour").log().mul(-1)
    ).sort("lab", "hour_ordered")

    return df_labs_by_dept

def plotSurprisalBedComparison(df_labs: pl.DataFrame,
                       dept1="UVHE NICU",
                       bed1="NICB09",
                       dept2="UVHE 4EAS",
                       bed2="4103A",
                       hour_buckets = 1,
                       labs= [
                           'Sodium',
                           'PaO2',
                           'Albumin',
                           'Hemoglobin',
                           'Troponin I'
                           ]):
    """
    Generates a 2x2 panel figure comparing suprisal values across labs and order hours for two beds and their respective units
    df_labs: a polars dataframe containing "lab", "dept", and "hour_ordered" columns
    dept1: department of bed1 (upper right)
    bed1: value for bed1 (upper left)
    dept2: department of bed2(lower right)
    bed1: value for bed1 (lower left)
    hour_buckets: int, size of window for analysis
    returns a polars dataframe containing the surprisal values by lab and hour_ordered
    """

    df = df_labs.filter(pl.col('dept').eq(dept1) & pl.col('bed').eq(bed1))
    df_bed1 = getSurprisalDataframe(df, hour_buckets=hour_buckets)
    df_dept1 = getSurprisalDataframe(df_labs.filter(pl.col('dept').eq(dept1)).sample(df.shape[0]), hour_buckets=hour_buckets)
    
    df = df_labs.filter(pl.col('dept').eq(dept2) & pl.col('bed').eq(bed2))
    df_bed2 = getSurprisalDataframe(df, hour_buckets=hour_buckets)
    df_dept2 = getSurprisalDataframe(df_labs.filter(pl.col('dept').eq(dept2)).sample(df.shape[0]), hour_buckets=hour_buckets)

    fig, axs = plt.subplots(2,2, figsize=(25,10))  
    ((ax1,ax2),(ax3, ax4)) = axs

    glue = df_bed1.to_pandas().pivot(index="lab", columns="hour_ordered", values="surprisal_by_lab_hour")
    ax = sns.heatmap(glue, ax=ax1)
    ax.set_title(f"surprisal for {bed1}")

    glue = df_dept1.to_pandas().pivot(index="lab", columns="hour_ordered", values="surprisal_by_lab_hour")
    ax = sns.heatmap(glue, ax=ax2)
    ax.set_title(f"surprisal for {dept1}")

    glue = df_bed2.to_pandas().pivot(index="lab", columns="hour_ordered", values="surprisal_by_lab_hour")
    ax = sns.heatmap(glue, ax=ax3)
    ax.set_title(f"surprisal for {bed2}")

    glue = df_dept2.to_pandas().pivot(index="lab", columns="hour_ordered", values="surprisal_by_lab_hour")
    ax = sns.heatmap(glue, ax=ax4)
    ax.set_title(f"surprisal for {dept2}")

    return fig, df_bed1, df_dept1, df_bed2, df_dept2

def exampleUVAFigure(df_labs: pl.DataFrame):
    bed1="NICB09"
    dept1="UVHE NICU"
    bed2="4103A"
    dept2="UVHE 4EAS"
    hour_buckets=1
    fig, df_bed1, df_dept1, df_bed2, df_dept2 = plotSurprisalBedComparison(df_labs, bed1=bed1, dept1=dept1, bed2=bed2, dept2=dept2, hour_buckets=hour_buckets)

    fig.savefig(f"out/surprisal_comparison_{hour_buckets}hr_{bed1}_{bed2}.png")
    fig.show()