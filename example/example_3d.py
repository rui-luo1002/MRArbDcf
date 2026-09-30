from numpy import *
from numpy.typing import NDArray
from matplotlib.pyplot import *
import mrarbgrad as mag
import mrarbdcf as mad
from time import time

nPix = 256

# generate a sampling pattern
dtGrad = 10e-6
dtAdc = 2.5e-6

mag.config(dt = dtGrad, enTrajRep=False)

traj = "Yarnball" # "Cones" "Yarnball" 
lstK0GradK1 = mag.scan(traj, nPix, nAcq=None)
lstArrGrad, lstArrK = [], []
for k0, arrGrad, k1 in lstK0GradK1:
    arrK = mag.integrate(arrGrad, dtGrad, dtAdc)
    arrK += k0
    lstArrGrad.append(arrGrad)
    lstArrK.append(arrK)

# solve for the DCF
t = time()
lstArrDcf = mad.solve(nPix, lstArrK)
t = time()-t
print(f"time: {t:.3f}")

# visualization
fig = figure(figsize=(12, 5))
idx = len(lstArrK)//4

ax = fig.add_subplot(121, projection='3d')
ax.plot(*lstArrK[idx].T, '.-')

ax = fig.add_subplot(122)
ax.plot(abs(lstArrDcf[idx]), ".-")
ax.set_xlabel("Index")
ax.set_ylabel("DCF")
ax.grid("both")

filename = __file__.replace(".py","_fig.png")
fig.savefig(filename, dpi=300)
print(f"figure saved to {filename}")
# show()
