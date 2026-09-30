import finufft as fn
from numpy import *
from numpy.fft import *
from numpy.linalg import norm
from matplotlib.pyplot import *
import matplotlib.ticker as ticker

nPix = 256
goldrat = (1+sqrt(5))/2
sWind = "poly" # "cos" "poly" "es"

fft = lambda x: fftshift(fftn(ifftshift(x)))
ift = lambda x: fftshift(ifftn(ifftshift(x)))
nufft = lambda k, x: fn.nufft1d2((2*pi)*k, x, eps=1e-8)
nuift = lambda k, x, n: fn.nufft1d1((2*pi)*k, x, n, eps=1e-8)

def winfun(x, p, sWind):
    if sWind=="cos": arrW = cos(pi/2 * x)**p
    elif sWind=="poly": arrW = 1 - (abs(x))**p
    elif sWind=="es": arrW = exp(p*sqrt(1-x*x))/exp(p)
    else: raise NotImplementedError()
    return arrW

arrP = arange(1.0,3.0,0.1) # arange(1.80, 2.00, 0.01)
arrErr = None
for iTest in range(100):
    print(f"iTest: {iTest}")
    random.seed(iTest)
    # arrK = random.randn(int(sqrt(2*pi)*e**4.5*nPix/6))/6
    '''
    PDF of 3sigma is `1/sqrt(2*pi)/e**4.5`, possibility to drop in `6/nPix` region at `3sigma` is `6/sqrt(2*pi)/e**4.5/nPix`
    '''
    arrK = random.randn(int(sqrt(2*pi)*e**2*nPix/4))/4
    '''
    PDF of 2sigma is `1/sqrt(2*pi)/e**2`, possibility to drop in `4/nPix` region at `2sigma` is `4/sqrt(2*pi)/e**2/nPix`
    '''
    # arrK = random.randn(int(sqrt(2*pi*e)*nPix/2))/2
    '''
    PDF of 1sigma is `1/sqrt(2*pi*e)`, possibility to drop in `2/nPix` region at `1sigma` is `2/sqrt(2*pi*e)/nPix`
    '''
    # figure()
    # hist(arrK, bins=100)
    # savefig("fig.pdf", dpi=300)
    # exit()
    # arrK = random.rand(2*nPix)-0.5
    # arrK = (iTest/nPix + arange(nPix*goldrat)/goldrat)%1-0.5
    arrK[0] = 0
    arrK = arrK[abs(arrK)<0.5]
    print(f"Nk: {arrK.size}, range: {arrK.min():.3f}, {arrK.max():.3f}")

    arrDcf = ones_like(arrK, dtype=complex128) # E
    # arrDcf = abs(arrK)+1/nPix + 0j # E
    arrDcf /= arrDcf.sum()
    arrPsf = nuift(arrK, arrDcf, (2*nPix)-1)/size(arrK) # P

    lstErr = []
    for iP in range(arrP.size):
        p = arrP[iP]
        x = linspace(-nPix,nPix,2*nPix-1)
        arrW = winfun(x/nPix, p, sWind)
        
        arrPsfStar = nuift(arrK, arrDcf/nufft(arrK, arrPsf*arrW), (2*nPix)-1)/size(arrK)
        arrWeightPsf = arrPsfStar * log2(1 + abs(linspace(-(nPix-1),nPix-1,2*nPix-1,1))) / abs(arrPsfStar[nPix-1])
        
        lstErr.append(norm(arrWeightPsf)) # , ord=inf (min-max)
    
    if arrErr is None: arrErr = array(lstErr)
    else: arrErr = maximum(arrErr, array(lstErr))
    
arrErr[isnan(arrErr)] = max(arrErr[~isnan(arrErr)])
iPOpt = argmin(arrErr)
    
# fig = figure(figsize=(5,3), dpi=150)
# ax = fig.add_subplot(111)
# lstC = ["tab:blue"]*arrP.size
# lstC[iPOpt] = "tab:red"
# ax.scatter(arrP, arrQuaErr, c=lstC)
# ax.set_xlabel(r"$p$")
# ax.set_ylabel(r"$\|P^\star_\mathrm{out}(\mathbf{x})\|$")
# ax.set_ylim(0,10)
# ax.text(arrP[iPOpt], arrQuaErr[iPOpt]-0.1, rf"$p={arrP[iPOpt]:.1f}$"+"\n"+rf"$\|P^\star_\mathrm{{out}}(\mathbf{x})\|={arrQuaErr[iPOpt]:.3f}$", horizontalalignment='center', verticalalignment='top')

wFPage = 4.77
fig = figure(figsize=(wFPage, wFPage*3/3), dpi=300)

ax = fig.add_subplot(211)
lstC = ["tab:blue"]*arrP.size
lstC[iPOpt] = "tab:red"
ax.scatter(arrP, arrErr, c=lstC)
ax.set_ylim(2,10) # arrErr.min()*0.9,arrErr.min()*1.1)
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$\|P^\star_\mathrm{out}(\mathbf{x})\|$")
ax.text(arrP[iPOpt], arrErr[iPOpt]-0.1, rf"$p={arrP[iPOpt]:.1f}$"+"\n"+rf"$\|P^\star_\mathrm{{out}}(\mathbf{{x}})\|={arrErr[iPOpt]:.3f}$", horizontalalignment='center', verticalalignment='top')
ax.set_title("(A)", loc="left")

ax = fig.add_subplot(212)
x = linspace(0,1,1000)
y = winfun(x, arrP[iPOpt], sWind)

ax.plot(x, y, "-")
ax.set_xlabel(r"$\|\mathbf{\bar{x}}\|$")
ax.set_ylabel(r"$W^\star(\mathbf{x})$")
if sWind=="cos": sFunc = rf"$W^\star(\mathbf{{x}})=\cos(\pi\mathbf{{\bar{{x}}}}/2)^{{{arrP[iPOpt]:.1f}}}$"
elif sWind=="poly": sFunc = rf"$W^\star(\mathbf{{x}})=1-\|\mathbf{{\bar{{x}}}}\|^{{{arrP[iPOpt]:.1f}}}$"
else: sFunc = rf"$W^\star(\mathbf{{x}})$"
ax.text(1,y[0], sFunc, horizontalalignment='right', verticalalignment='top')
ax.set_title("(B)", loc="left")

fig.subplots_adjust(0.15,0.1,0.99,0.95, wspace=0.0, hspace=0.5)
fig.savefig("temp/figure/PSearch.png", dpi=300)

show()