import numpy as np

def _hillebrand_PTE(xs):

    import sys
    from phase_transfer_entropy.hillebrand.PhaseTE_MF import PhaseTE_MF
    sys.path.append('../phase_transfer_entropy/hillebrand/')
    dPTE, raw_PTE = PhaseTE_MF(xs)
    return dPTE, raw_PTE


def PTE(measurement, method='hillebrand'):

    assert(method in ['hillebrand', 'myPTE'])
    raw_data, times = measurement[:]

    x = np.asanyarray(raw_data)

    xs = np.swapaxes(x, 0, 1)
    if(method=='hillebrand'):
        dPTE, raw_PTE = _hillebrand_PTE(xs)
        return dPTE, raw_PTE
    if(method=='myPTE'):
        import myPTE
        phase = myPTE._get_phase(xs)
        delay = myPTE._get_delay(phase)
        phase2 = phase + np.pi
        binsize = myPTE._get_binsize(phase2)
        bincount = myPTE._get_bincount(binsize)
        dphase = myPTE._get_discretized_phase(phase2, binsize)
        dPTE, raw_PTE = myPTE.compute_dPTE_rawPTE(dphase, delay)
        return dPTE, raw_PTE

