from scipy.stats import wilcoxon
import numpy as np
import pandas as pd

def matrix_wilcoxon(x_matrices, y_matrices, alpha = 0.05, bonferroni=True):
    x = list()
    y = list()
    for data_frame in x_matrices:
        x.append(data_frame.as_matrix())
    for data_frame in y_matrices:
        y.append(data_frame.as_matrix())

    p_values = np.zeros_like(x[0])
    p_mask = np.zeros_like(x[0])

    m, n = p_values.shape

    alpha_c = alpha
    if bonferroni:
        alpha_c = alpha*m**2

    for i in range(0, m):
        for j in range(0, m):
            temp_x = list()
            temp_y = list()
            for matrix in x:
                temp_x.append(matrix[i, j])
            for matrix in y:
                temp_y.append(matrix[i, j])
            s, p = wilcoxon(temp_x, temp_y)

            p_values[i, j] = p

            if p < alpha_c:
                p_mask[i, j] = False
            else:
                p_mask[i, j] = True

    p_values_df = pd.DataFrame(p_values, index=x_matrices[0].index, columns=x_matrices[0].columns)
    p_mask_df = pd.DataFrame(p_mask, index=x_matrices[0].index, columns=x_matrices[0].columns)

    return p_values_df, p_mask_df
