import numpy as np


def jansen_rit(g1 = 135, C=None):
    """
    Neural mass model which returns a dynamical system F and Wiener process G
    for a given coupling matrix C

    This function defines:
        all fixed parameters for the Jansen-Rit model
        f : the dynamical system of a particular model
        F : the overall dynamical system of coupled f to be integrated with sdeint
        g : the Wiener process per dynamical system
        G : a stack of g to be integrated with sdeint

    Parameters
    ----------
    g1 : int
        Model parameter which determines what kind of rhythm will be generated
        default : 135 for alpha-waves
    C : ndarray
        coupling matrix must be quadratic. values of the coupling matrix correspond to
        the coupling strength between neural mass models

    Returns
    -------
    F : function
        dynamical system of size m x 8 where m is the number of neural mass models
    G : function
        Wiener process of size m x 8

    """
    if(C is not None):
        m, n = C.shape
        assert(m==n)



    r1 = 0.56
    r2 = 6.0
    e0 = 2.5
    He = 3.25
    Hi = 22
    ke1 = 100
    ke2 = 100
    ke3 = 100
    ki = 50

    g2 = 0.8*g1
    g3 = 0.25*g1
    g4 = 0.25*g1

    u_sdev = 0.0
    u_mean = 0.0
    p_sdev = 100.0/np.sqrt(3.0)
    p_mean = 220

    average_timestep_used_by_jr = 0.0012
    p_sdev = p_sdev*np.sqrt(average_timestep_used_by_jr)

    def S(y):
        return (2.0*e0)/(1.0 + np.exp(r1*(r2-y)))

    def F(y, t):

        dy = np.zeros([m, 8])

        for i in range(m):
            dy[i,:] = f(y[i,:], t)
            if(C is not None):
                dy[i, 4] += ( He * ke2 *np.sum(C[i, :] * y[i,6]) )

        return dy


    def f(y, t):
        dy = np.zeros(8)

        dy[0] = y[3]
        dy[3] = He * ke1 * (g1 * S(y[1] - y[2])) - 2 * ke1 * y[3] - y[0] * ke1 ** 2
        dy[1] = y[4]
        dy[4] = He * ke2 * (g2 * S(y[0])) - 2 * ke2 * y[4] - y[1] * ke2 ** 2
        dy[2] = y[5]
        dy[5] = Hi * ki * g4 * S(y[0]) - 2 * ki * y[5] - y[2] * ki ** 2 + p_sdev
        dy[6] = y[7]
        dy[7] = He * ke3 * g3 * S(y[1] - y[2]) - 2 * ke3 * y[7] - y[6] * ke3 ** 2
        return dy


    def G(v, t):
        dW = np.zeros([m, 8])
        for i in range(m):
            dW[i,:] = g(v[i,:], t)
        return dW

    def g(v, t):
        dw = np.zeros(8)
        dw[4] = ke1 * He * (u_sdev + u_mean)
        dw[5] = ke2 * He * (p_sdev + p_mean)
        return dw


    return F, G
