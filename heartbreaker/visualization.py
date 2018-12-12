import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn import tree

import data_loader
import util
import numpy as np
import graphviz

def tsne_visualization(X, y):
    scaler = StandardScaler()
    scaler.fit(X)
    
    X_scaled = scaler.transform(X)
    
    pca = PCA(n_components=50)
    X_50 = pca.fit(X_scaled).transform(X_scaled)
    
    X_embedded = TSNE(n_components=2).fit_transform(X_50)
    
    # based on PCA example @
    # https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py
    
    target_names = ["low risk", "high risk"]
    colors = ['blue', 'red']
    
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('t-SNE of combined datasets')
    
    plt.savefig('../plots/tsne.png', dpi=450)

# Run command: dot -Tpng -Gsize=9,15\! -Gdpi=450 -ofoo.png test.gv
# manually on output .gv file to get high-res output 
def decision_tree_visualization(X, y):
    y_copy = np.copy(y)
    
    # need to flip to get coloring right
    # (by default graphviz export colors True nodes blue and there is no easy
    # override)
    for i in range(len(y)):
        y_copy[i] = not(y[i])
        
    
    clf = tree.DecisionTreeClassifier(
        criterion="entropy",
        min_samples_leaf=.01,
        max_leaf_nodes=10,
        max_depth=3,
        class_weight='balanced'
    )
    clf.fit(X,y_copy)
    
    
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names = list(X),
                                    class_names=["high risk", "low risk"], filled=True,
                                    rounded=True, special_characters=True, proportion=True,
                                    impurity=False, rotate=True, node_ids=True)
    
    with open("../results/dt.gv", "w+") as file:
        file.write(dot_data)
        
def plot_two_features(X, y, f1, f2):
    vs = X[[f1, f2]].values

    plt.figure()
    colors = ['blue', 'red']
    target_names = ["low risk", "high risk"]
    lw = 2
    
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(vs[y == i,0], vs[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.savefig("%s_v_%s.png" % (f1, f2), dpi=450)

def main():
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    rates = data.pop('heart_disease_mortality')
    rates = util.continuous_to_categorical(rates, percentile_cutoff=75)
    X = data
    y = rates
    
    tsne_visualization(X, y)
    
    decision_tree_visualization(X, y)