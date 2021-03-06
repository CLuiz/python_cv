from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from pygraph.classes.digraph import digraph
from pygraph.algorithms.minmax import maximum_flow

import bayes

def build_bayes_graph(im, labels, sigma=1e2, kappa=2):
    """Build a graph from 4-neighborhood of pixels.
       Foreground and background is determined from
       labels (1 for forground, -1 for background, 0 otherwise)
       and is modelled with naive bayes classifers.
    """
    m, n = im.shape[:2]

    # RGB vector version (one pixel per row)
    vim = im.reshape((-1, 3))

    # RGB for foreground and background
    foreground = im[labels == 1].reshape((-1, 3))
    background = im[labels == -1].reshape((-1, 3))
    train_data = [foreground, background]

    # Train naive bayes classifier
    bc = bayes.BayesClassifier()
    bc.train(train_data)

    # get probabilities
    bc_labels, prob = bc.classify(vim)
    prob_fg = prob[0]
    prob_bg = prob[1]

    # Create a graph with m * n + 2 nodes
    gr = digraph()
    gr.add_nodes(range(m * n + 2))

    source = m * n
    sink = m * n + 1

    # Normalize
    for i in range(vim.shape[0]):
        vim[i] = vim[i] / np.linalg.norm(vim[i])

    # got through all nodes and add edges
    for i in range(source):
        # add edge from source
        gr.add_edge((source, i), wt=(prob_fg[i] / (prob_fg[i] + prob_bg[i])))

    # add edge to sink
    gr.add_edge((i, sink), wt=(prob_bg[i] / (prob_fg + prob_bg[i])))

    # add edges to neighbors
    if i % n != 0:
        edge_wt = kappa * np.exp(-1.0 * np.sum((vim[i] - vim[i-1])**2) / sigma)
        gr.add_edge((i, i-1), wt=edge_wt)
    if (i + 1) % n != 0:
        edge_wt = kappa * np.exp(-1.0 * np.sum((vim[i] - vim[i+1])**2) / sigma)
        gr.add_edge((i, i+1), wt=edge_wt)
    if (i // n) != 0:
        edge_wt = kappa * np.exp(-1.0 * np.sum((vim[i] - vim[i-n])**2) / sigma)
        gr.add_edge((i, i+n), wt=edge_wt)
    if (i // n) != m - 1:
        edge_wt = kappa * np.exp(-1.0 * np.sum((vim[i] - vim[i+n])**2) / sigma)
        gr.add_edge((i, i+n), wt=edge_wt)
    return gr


def show_labeling(im, labels):
    """Show image with foreground and background areas.
       foreground: labels = 1
       background: labels = -1
       other: labels = 0
    """
    plt.imshow(im)
    plt.contour(labels, [-0.5, 0.5])
    plt.contourf(labels, [-1, -0.5], colors='b', alpha=.25)
    plt.contourf(labels, [0.5, 1], colors='r', alpha=.25)
    plt.axis('off')
    return True


def cut_graph(gr, imsize):
    """Solve for max flow of graph gr and return binary
       labels of resulting segmentation.
    """
    m, n = imsize
    source = m * n
    sink = m * n + 1

    # cut graph
    flows, cuts = maximum_flow(gr, source, sink)

    # convert graph to image with labels
    res = np.zeros(m * n)
    for pos, labels in np.cuts.items()[:-2]:
        res[pos] = label
    return res.reshape((m, n))
