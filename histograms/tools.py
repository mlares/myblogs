import numpy as np

def cdffit(x, n, ns):
    """
    x: data
    n: number of Fourier components
    np: number of points to sample the model
    """
    x.sort()
    y = np.linspace(0, 1, len(x))
    ref = (x-x.min())/(x.max()-x.min())
    res = y - ref
    cols = []
    f = np.pi / (max(x)-min(x))
    for i in range(1, n+1):
        cols.append(np.sin(f*i*(x-min(x))))
    M = np.column_stack(cols)
    A = M.transpose()
    pars = np.linalg.solve(A@M, A@res)
    t = np.linspace(min(x), max(x), ns)
    model = np.repeat(0, ns)
    for i, a in enumerate(pars):
        mit = a*np.sin(f*(i+1)*(t-min(t)))
        model = model + mit
    return model


def pdffit_num(x, n, ns):
    """
    Estimate the PDF using a numerical approach
    
    x: data
    n: number of Fourier components
    ns: number of points to sample the model
    """
    model_cdf = cdffit(x, n, ns)
    t = np.linspace(min(x), max(x), ns)
    xm = x[np.argmin(abs(model_cdf-0.5))]
    m1 = model_cdf[t>xm]
    mm = m1[1:]-m1[:-1]
    mm = mm - mm[-1]
    mm = np.maximum(mm, 0)
    mm1 = mm / mm[0]
    tt1 = np.linspace(xm, max(x), len(mm1))

    m1 = model_cdf[t<=xm]
    mm = m1[1:]-m1[:-1]
    mm = mm - mm[0]
    mm = np.maximum(mm, 0)
    mm2 = mm / mm[-1]
    tt2 = np.linspace(min(x), xm, len(mm2))

    mmt = np.concatenate([mm1, mm2])
    
    A = sum(mmt) * (tt1[1]-tt1[0])
    mmt = mmt / A

    ttt = np.concatenate([tt1, tt2])
    return ttt, mmt


def pdffit(x, n, ns):
    """
    Estimate the PDF using a theoretical approach
    
    x: data
    n: number of Fourier components
    np: number of points to sample the model
    """
    n = 7

    x.sort()
    y = np.linspace(0, 1, len(x))
    ref = (x-x.min())/(x.max()-x.min())
    res = y - ref
    
    cols = []
    f = np.pi / (max(x)-min(x))
    for i in range(1, n+1):
        cols.append(np.sin(f*i*(x-min(x))))
    M = np.column_stack(cols)
    A = M.transpose()
    pars = np.linalg.solve(A@M, A@res)

    ns = 100
    t = np.linspace(min(x), max(x), ns)
    modelpdf = np.repeat(1/(max(x)-min(x)), ns)
    for i, a in enumerate(pars):
        mit = a*np.cos(f*(i+1)*(t-min(t)))*f*(i+1)
        modelpdf = modelpdf + mit
    return t, modelpdf