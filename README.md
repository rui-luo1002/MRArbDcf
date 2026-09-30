# Non-Cartesian MRI Density Compensation Toolbox (MRArbDcf)
Python library for solving for the density compensation function (DCF) for arbitrary MRI trajectories.

## How to Use
### Install
```bash
$ pip install mrarbdcf mrarbgrad
```

### Import Libraries
```python
from numpy import *
import mrarbdcf as mad
import mrarbgrad as mag
```

### Generate a Trajecotry
```python
lstK0GradK1 = mag.scan("Cones", 256, nAcq=None)
lstArrK = []
for k0, arrGrad, k1 in lstK0GradK1:
    arrK = k0 + mag.integrate(arrGrad, 10e-6, 2.5e-6)
    lstArrK.append(arrK)
```

### Solve for the Density Compensation Function
```pyhton
lstArrDcf = mad.solve(256, lstArrK)
```

## Acknowledgements
The algorithm in this library is proposed in:

> [1] Luo R, Hu P, Qi H. Sampling Density Compensation using Fast Fourier Deconvolution. In: Proc. Int. Soc. Magn. Reson. Med. [Internet]. 2026 [cited 2026 Apr 28]. Available from: http://echo.ismrm.org/p/ISMRM2026/461-03-015
>
> [2] Luo R, Hu P, Qi H. Sampling Density Compensation using Fast Fourier Deconvolution [Internet]. arXiv; 2025 [cited 2025 Oct 17]. Available from: http://arxiv.org/abs/2510.14873

Additionally, FINUFFT [3,4] and CUFINUFFT [5] are adopted for NUFFT operators in this package.

> [3] Barnett AH, Magland J, af Klinteberg L. A Parallel Nonuniform Fast Fourier Transform Library Based on an “Exponential of Semicircle" Kernel. SIAM J Sci Comput. 2019 Jan;41(5):C479–504. 
>
> [4] Barnett AH. Aliasing error of the exp(β√(1-z²)) kernel in the nonuniform fast Fourier transform. Applied and Computational Harmonic Analysis. 2021 Mar 1;51:1–16. 
>
> [5] Shih Y hsuan, Wright G, Anden J, Blaschke J, Barnett AH. cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs. 2021 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW). 2021 June;688–97. 
