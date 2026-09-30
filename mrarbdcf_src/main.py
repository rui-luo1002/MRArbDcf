# import library
import numpy
from numpy.typing import *
from typing import *
import finufft
from .utility import normDcf
from time import time
from itertools import product
from psutil import cpu_count
from importlib import import_module

# gpu library
try: import cufinufft, cupy
except ImportError: cufinufft, cupy = None, None

pi = numpy.pi

# interface function
fDbgInfo = False
def setDbgInfo(x:bool):
    """
    Set whether to print debug input. Default to False.

    Args:
        x: True for to print. False for not to.
    """
    global fDbgInfo
    fDbgInfo = bool(x)
    
enInputCheck = True
def setInputCheck(x:bool):
    """
    Set whether to check the input type. Recommend and default to True. Can be set to False for benchmark reason.

    Args:
        x: True for to use. False for not to.
    """
    global enInputCheck
    enInputCheck = bool(x)
    
enWarmStart = 3
def setWarmStart(x:int):
    """
    Set whether to use warm start. Recommend and default to True.
    Note that 

    Args:
        x: 0 for disabled; 1 for radial DCF; 2 for linear DCF; 3 for both
    """
    global enWarmStart
    enWarmStart = int(x)

lstWType = ["poly"]
lstWPara = [2.4]
def setWind(wtype:list[str], wpara:list[float]):
    """
    Set type and shape parameter of the PSF window

    Args:
        wtype (list[str]): window type, can be "poly" or "cos" or "es"
        spara (list[float]): desired shape parameter
    """
    global lstWType, lstWPara
    lstWType = wtype
    lstWPara = wpara
    
def setMode(x:Literal["general", "fast"]):
    """
    Set solver's working mode.
    
    Args:
        x: 0 for general mode, 1 for fast mode
    """
    if x=="general":
        setInputCheck(1)
        setWarmStart(0)
        setWind(["corr", "poly"], [0, 2.4])
        """
        @note: if trajectory-based warm start is 
        impossible, we use a Fourier PD window - autocorrelation window to generate an
        initial guess, because this kind of window
        will not cause DCF cancelation.
        """
    elif x=="fast":
        setInputCheck(0)
        setWarmStart(3)
        setWind(["poly"], [2.4])
    else:
        raise ValueError("x")
    
def backend(sample:Any):
    return import_module(type(sample).__module__)

# main solver
def solve(nPix:int, lstArrK:list[NDArray]) -> list[NDArray]:
    """
    Solve for the density compensation function.
        
    Args:
        nPix: Designed number of pixels of the trajectory.
        lstArrK: list of trajectories, element shape [nK,nAx].
        
    Returns:
        list of density compensation function, element shape [nK,].
    """
    t0 = time()
    
    xp = backend(lstArrK[0])
    arrCatK = xp.concatenate(lstArrK, axis=0).astype(xp.float32)
    nTraj = len(lstArrK)
    arrNRO = xp.array([arrK.shape[0] for arrK in lstArrK])
    arrI0 = xp.zeros((len(arrNRO),), dtype=int)
    arrI0[1:] = xp.cumsum(arrNRO)[:-1]
    
    if enInputCheck: # check
        # unfixable
        if arrCatK.ndim!=2 or arrCatK.shape[1] not in (2,3):
            raise RuntimeError(f"`lstArrK[i].shape` should be `[nK,nDim]`, got `{arrCatK.shape}`")
        arrRho:NDArray = xp.linalg.norm(arrCatK, axis=-1)
        if arrRho.max()>0.51: raise UserWarning("k-range: [-0.5,0.5]")
        # fixable
        if arrCatK.shape[1]==3 and (arrCatK[:,2]==0).all(): arrCatK = arrCatK[:,:2]
    if fDbgInfo: print(f"# input check: {time() - t0:.3f}s"); t0 = time()
    
    # basic parameter
    nPix = int(nPix)
    nK, nAx = arrCatK.shape
    complex = xp.complex64
    float = xp.float32
    scomplex = "complex64"
    
    # data initialize
    arrCatDcf = xp.ones(nK, dtype=complex)
    arrCatDcf = normDcf(arrCatDcf, nAx)
    
    if enWarmStart & 0x01:
        # radial DCF
        arrRho = xp.sum(arrCatK**2, axis=-1, dtype=complex)
        xp.sqrt(arrRho, out=arrRho)
        arrCatDcf *= (arrRho+1/nPix)**(nAx-1)
    if enWarmStart & 0x02:
        # 1D DCF
        arrDcf1D = xp.empty(nK, dtype=complex)
        arrDcf1D[:-1] = xp.sqrt(xp.sum(xp.diff(arrCatK, axis=0)**2, axis=-1))
        arrDcf1D[arrI0-1] = arrDcf1D[arrI0-2]
        arrCatDcf *= arrDcf1D
    
    if fDbgInfo: print(f"# data initialize: {time() - t0:.3f}s"); t0 = time()

    nStep = len(lstWType)
    for iStep in range(nStep):
        wtype = lstWType[iStep]
        wpara = lstWPara[iStep]
        # grid of rho
        coords = xp.ogrid[tuple(slice(0, 1, nPix*1j) for _ in range(nAx))]
        arrGridRho = xp.sqrt(sum(c.astype(float)**2 for c in coords))
        if fDbgInfo: print(f"# grid of rho: {time() - t0:.3f}s"); t0 = time()

        # Nd window
        r = arrGridRho.clip(0,1)
        if wtype=="poly": arrWindNd = 1 - r**wpara # 2.4
        elif wtype=="cos": arrWindNd = 0.5 + 0.5*xp.cos(r*pi).clip(0,1)**wpara # 0.7
        elif wtype=="es": beta=wpara; arrWindNd = xp.exp(beta*xp.sqrt(1-r**2))/xp.exp(beta)
        elif wtype=="gaus": arrWindNd = xp.exp(-r**2/(2*wpara**2))
        elif wtype=="wend": arrWindNd = (1-r)**4 * (4*r + 1)
        elif wtype=="wend4": arrWindNd = ((1-r)**6) * (35*r**2 + 18*r + 3) / 3.0
        elif wtype=="corr":
            if nAx==1: arrWindNd = 1 - r
            elif nAx==2: arrWindNd = 2/pi*(xp.arccos(r) - r*xp.sqrt(1-r**2))
            elif nAx==3: arrWindNd = 1 - 3/2*r + 1/2*r**3
        else: raise NotImplementedError("")
        arrWindNd[arrGridRho>1] = 0
        arrWindNd /= abs(arrWindNd).max()
        del arrGridRho
        
        for iAx in range(nAx):
            tupSli = tuple(0 if iAx==_iAx else slice(None) for _iAx in range(nAx))
            xp.sqrt(arrWindNd[tupSli], out=arrWindNd[tupSli])
        if fDbgInfo: print(f"# Nd window: {time() - t0:.3f}s"); t0 = time()
        
        # deconvolve
        if xp==numpy:
            nufftpara = {"debug":0, "spread_debug":0, "showwarn":0, "upsampfac":1.25, "nthreads":cpu_count(logical=True), "spread_sort":1, "fftw":64}
            fn = finufft
        elif xp==cupy:
            nufftpara = {"upsampfac":1.25}
            fn = cufinufft
        else: raise ValueError("xp")
        
        n_modes = tuple(2*nPix-1 for _ in range(nAx))
        arr2PiKT = xp.array(arrCatK.T, order='C', dtype=float)
        arr2PiKT *= 2*pi
        eps = 1e-3
        
        pNuift = fn.Plan(1, n_modes, eps=eps, dtype=scomplex, **nufftpara)
        pNufft = fn.Plan(2, n_modes, eps=eps, dtype=scomplex, **nufftpara)
        pNuift.setpts(*arr2PiKT)
        pNufft.setpts(*arr2PiKT)
    
        arrPsf = pNuift.execute(arrCatDcf)
        if fDbgInfo: print(f"# nuift: {time() - t0:.3f}s"); t0 = time()
        
        # suppress alias outside of PSF
        sliNeg = slice(nPix-1,None,-1)
        sliPos = slice(nPix-1,None,1)
        for iCorner in product(range(2), repeat=nAx):
            tupSli = tuple(sliNeg if i else sliPos for i in iCorner)
            arrPsf[tupSli] *= arrWindNd
        
        arrDcfApo = pNufft.execute(arrPsf)
        arrCatDcf /= arrDcfApo
        arrCatDcf = normDcf(arrCatDcf, nAx)
        if fDbgInfo: print(f"# nufft: {time() - t0:.3f}s"); t0 = time()
    
    lstArrDcf = [arrCatDcf[arrI0[iTraj].item():arrI0[iTraj+1].item()] for iTraj in range(nTraj-1)]
    lstArrDcf.append(arrCatDcf[arrI0[-1].item():])
    
    return lstArrDcf
    
