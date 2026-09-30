from numpy import *
import mrarbdcf as mad
import mrarbgrad as mag

lstK0GradK1 = mag.scan("Cones", 256, nAcq=None)
lstArrK = []
for k0, arrGrad, k1 in lstK0GradK1:
    arrK = k0 + mag.integrate(arrGrad, 10e-6, 2.5e-6)
    lstArrK.append(arrK)

lstArrDcf = mad.solve(256, lstArrK)
