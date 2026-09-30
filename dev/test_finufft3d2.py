import numpy as np
import finufft

# number of nonuniform points
M = 100

# the nonuniform points
x = 2 * np.pi * np.random.uniform(size=M)
y = 2 * np.pi * np.random.uniform(size=M)
z = 2 * np.pi * np.random.uniform(size=M)

# number of Fourier modes
N1, N2, N3 = 50, 75, 100

# the Fourier mode coefficients
f = (np.random.standard_normal(size=(N1, N2, N3))
     + 1J * np.random.standard_normal(size=(N1, N2, N3)))

# calculate the type-2 NUFFT
c = finufft.nufft3d2(x, y, z, f)