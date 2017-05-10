#! /usr/bin/env python
# -*- coding: utf-8 -*-
# https://gist.github.com/292018 - augmented to plot row/column labels
"""
Modified by Tony Francis

Draws Hinton diagrams using matplotlib ( http://matplotlib.sf.net/ ).
Hinton diagrams are a handy way of visualizing weight matrices, using
colour to denote sign and area to denote magnitude.

By David Warde-Farley -- user AT cs dot toronto dot edu (user = dwf)
  with thanks to Geoffrey Hinton for providing the MATLAB code off of 
  which this is modeled.

Redistributable under the terms of the 3-clause BSD license 
(see http://www.opensource.org/licenses/bsd-license.php for details)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

from hyperparams import *

def _blob(x, y, area, colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    plt.fill(xcorners, ycorners, colour, edgecolor=colour)

def hinton(W, names=None, maxweight=None, filename=None, title=None):
# def hinton(W, maxweight=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix. 
    Temporarily disables matplotlib interactive mode if it is on, 
    otherwise this takes forever.
    """

    F = plt.figure(num=HINTON_GRAPH, figsize=(8,6), dpi=300, facecolor='w', edgecolor='k')
    F.suptitle(title, fontsize=16)

    reenable = False
    if plt.isinteractive():
        plt.ioff()
    
    F.clf()
    height, width = W.shape
    if not maxweight:
        maxweight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))
        
    plt.fill(np.array([0, width, width, 0]),
             np.array([0, 0, height, height]),
             'gray')
    
    plt.axis('off')
    plt.axis('equal')
    for x in xrange(width):
        if names:
            plt.text(-0.5, x, names[x], fontsize=4, ha='right', va='bottom')
            plt.text(x, height+0.5, names[height-x-1], fontsize=4, va='bottom', rotation='vertical', ha='left')
        for y in xrange(height):
            _x = x+1
            _y = y+1
            w = W[y, x]
            if w > 0:
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, w/maxweight),
                      'white')
            elif w < 0:
                _blob(_x - 0.5,
                      height - _y + 0.5, 
                      min(1, -w/maxweight), 
                      'black')
    if reenable:
        plt.ion()

    if filename:
        F.savefig(filename, dpi=300)
    
def visualize_weights(weights, layer, epoch):
    filename = "figures/weights_{0}_epoch_{1}".format(layer, epoch)
    title = "Layer {0} - Episode {1}".format(layer, epoch)
    hinton(weights, filename=filename, title=title)