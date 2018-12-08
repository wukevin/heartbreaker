import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 6]

import shap

import data_loader

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
PLOTS_DPI = 600
assert os.path.isdir(PLOTS_DIR)

def histogram(values, xlabel, title, fname, nbins=40):
    """Plot a histogram with the given parameters and save it to the given filename"""
    fig, ax = plt.subplots()

    ax.hist(values, bins=nbins)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    plt.savefig(fname, bbox_inches='tight', dpi=PLOTS_DPI)

def plot_forward_search(source=os.path.join(os.path.dirname(os.path.dirname(__file__)), "results/forward_search_197_logreg.csv"), plot_fname=os.path.join(PLOTS_DIR, "forward_search_plot.png")):
    """Plots the results of forward search"""
    results = pd.read_csv(source, index_col=0)
    fig, ax = plt.subplots()

    ax.scatter(range(len(results)), results, alpha=0.6, marker=".")
    ax.set_ylabel("F1 score")
    ax.set_xlabel("Num. features (n={})".format(len(results)))
    ax.set_title("Forward feature search")
    
    plt.savefig(plot_fname, bbox_inches='tight', dpi=PLOTS_DPI)

def make_histograms():
    """Generate the plots"""
    data = data_loader.load_all_data()
    
    histogram(data['heart_disease_mortality'], "Rate per 100,000 population", "Heart disease mortality rates", os.path.join(PLOTS_DIR, "mortality_histogram.png"))

    histogram(data['PC_FSRSALES07'], "Dollars", "Expenditures per capita, restaurants, 2007", os.path.join(PLOTS_DIR, "restaurants_per_capita_hist.png"))
    print(np.nanmin(data['PC_FSRSALES07']))
    print(np.nanmedian(data['PC_FSRSALES07']))
    print(np.nanmax(data['PC_FSRSALES07']))

    histogram(data['VEG_FARMS12'], "Count", "Vegetable farms, 2012", os.path.join(PLOTS_DIR, "veg_farms_2012_hist.png"))
    print(np.nanmin(data['VEG_FARMS12']))
    print(np.nanmedian(data['VEG_FARMS12']))
    print(np.nanmax(data['VEG_FARMS12']))

def plot_shap_tree_summary(model, baseline_data, eval_data, output_fname=""):
    """Create a shap summary plot"""
    explainer = shap.TreeExplainer(model, baseline_data)
    shap_values = explainer.shap_values(eval_data)
    shap.summary_plot(
        shap_values,
        eval_data,
        class_names=["Low risk", "High risk"],
        show=False if output_fname else False  # Allows for saving below
    )
    if output_fname:
        plt.savefig(
            output_fname,
            bbox_inches='tight',
            dpi=PLOTS_DPI,
        )

if __name__ == "__main__":
    # make_histograms()
    plot_forward_search()
