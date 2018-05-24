import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


def dPTE_heatmap(dPTE, channels, vmin=None, vmax=None):
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    cmap.set_under('.5')

    g = sns.heatmap(dPTE, cmap=cmap, center=0.5, vmin=vmin, vmax=vmax)

    g.set_xticklabels(channels, rotation=90)
    g.set_yticklabels(channels[::-1])


def rawPTE_heatmap(rawPTE, channels, vmin=None, vmax=None):
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    g = sns.heatmap(rawPTE, cmap=cmap, vmin=vmin, vmax=vmax)

    g.set_xticklabels(channels, rotation=90)
    g.set_yticklabels(channels[::-1])


def _overall_vmin_vmax(matrices):
    dmin = list()
    dmax = list()
    for matrix in matrices:
        dmin.append(np.min(matrix.np.nonzero(matrix)))
        dmax.append(np.max(matrix))

    vmin = min(dmin)
    vmax = max(dmax)

    return vmin, vmax


def _replace_with_zeros(array, mask):

    m, n = array.shape
    assert(m==n)

    B = array
    b = np.zeros(m)

    for i in range(m):
        if i in mask:
            B[i,:] = b
        for j in range(m):
            if j in mask:
                B[:,j] = b
    return B


def _insert_zeros(array, mask):

    m, n = array.shape
    c = len(mask)
    l = m+c
    B = np.empty([l,l])
    B[:] = 0
    # print(mask)
    k = 0

    for i in range(m):
        g = 0
        if i in mask:

            # print(i)
            k += 1

        for j in range(m):
            # print(k, g)

            if j in mask:
                # print(j)
                g += 1

            B[i+k,j+g] = array[i,j]
    return B


def _delete_channels(array, mask):
    array = np.delete(array, mask, 0)
    array = np.delete(array, mask, 1)
    return array