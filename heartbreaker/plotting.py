import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 6]

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

def main():
    """Generate the plots"""
    data = data_loader.load_all_data()
    
    histogram(data['heart_disease_mortality'], "Rate per 100,000 population", "Heart disease mortality rates", os.path.join(PLOTS_DIR, "mortality_histogram.png"))

if __name__ == "__main__":
    main()
