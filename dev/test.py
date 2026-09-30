import numpy as np
import matplotlib.pyplot as plt
import finufft
import mrarbgrad as mag

pi = np.pi


def save_clean_line_image(x, y, out_png, figsize=6, dpi=300,
                          bg="white", fg="black", lw=0.5):
    """Save a trajectory using plot(), with no axes or margins."""
    fig = plt.figure(figsize=(figsize, figsize), dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.plot(x, y, ".-", markersize=8, color=fg, lw=lw)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")

    fig.savefig(out_png, dpi=dpi, pad_inches=0)
    plt.close(fig)


def save_clean_log_image(arr2d, out_png, cmap="gray", dpi=300, floor_db=-60):
    """
    Save a 2D array in log scale, no axes/ticks/title/colorbar/margins.
    """
    mag = np.abs(arr2d).astype(np.float32)
    mag /= mag.max() + 1e-30
    db = 20 * np.log10(mag + 1e-30)
    db = np.clip(db, floor_db, 0.0)

    fig = plt.figure(figsize=(6, 6), dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(db, origin="lower", interpolation="nearest",
              vmin=floor_db, vmax=0.0)
    ax.axis("off")

    fig.savefig(out_png, dpi=dpi, pad_inches=0)
    plt.close(fig)


def make_window_nd(nPix, nAx=2, wtype="poly", wpara=2.4):
    coords = np.ogrid[tuple(slice(0, 1, nPix * 1j) for _ in range(nAx))]
    rho = np.sqrt(sum(c.astype(np.float32) ** 2 for c in coords))

    r = np.clip(rho, 0, 1)

    if wtype == "poly":
        wind = 1 - r**wpara
    elif wtype == "cos":
        wind = 0.5 + 0.5 * np.cos(np.clip(r * pi, 0, 1)) ** wpara
    elif wtype == "es":
        beta = wpara
        wind = np.exp(beta * np.sqrt(1 - r**2)) / np.exp(beta)
    elif wtype == "gaus":
        wind = np.exp(-r**2 / (2 * wpara**2))
    elif wtype == "wend":
        wind = (1 - r) ** 4 * (4 * r + 1)
    elif wtype == "wend4":
        wind = ((1 - r) ** 6) * (35 * r**2 + 18 * r + 3) / 3.0
    elif wtype == "corr":
        if nAx == 1:
            wind = 1 - r
        elif nAx == 2:
            wind = 2 / pi * (np.arccos(r) - r * np.sqrt(1 - r**2))
        elif nAx == 3:
            wind = 1 - 1.5 * r + 0.5 * r**3
        else:
            raise ValueError("corr window only implemented for nAx=1,2,3")
    else:
        raise NotImplementedError(wtype)

    wind[rho > 1] = 0
    wind /= np.abs(wind).max()

    for iAx in range(nAx):
        sli = tuple(0 if iAx == j else slice(None) for j in range(nAx))
        wind[sli] = np.sqrt(wind[sli])

    return wind.astype(np.float32)


def apply_corner_window_2d(psf, wind):
    nPix = wind.shape[0]
    out = psf.copy()

    sliNeg = slice(nPix - 1, None, -1)
    sliPos = slice(nPix - 1, None,  1)

    out[sliPos, sliPos] *= wind
    out[sliPos, sliNeg] *= wind
    out[sliNeg, sliPos] *= wind
    out[sliNeg, sliNeg] *= wind

    return out


def psf_from_traj_2d(arrK, nPix, eps=1e-6):
    """
    Compute PSF from a single 2D trajectory/interleave.
    Output grid size is (2*nPix-1, 2*nPix-1), i.e. 2xFOV support.
    """
    n_modes = (2 * nPix - 1, 2 * nPix - 1)
    arr2PiKT = np.array(arrK.T, order="C", dtype=np.float32)
    arr2PiKT *= 2 * pi

    weights = np.ones(arrK.shape[0], dtype=np.complex64)

    plan = finufft.Plan(
        1, n_modes,
        eps=eps,
        dtype="complex64",
        debug=0,
        spread_debug=0,
        showwarn=0,
        upsampfac=1.25,
    )
    plan.setpts(*arr2PiKT)
    psf = plan.execute(weights)
    return psf


# --------------------------------------------------
# 1) Generate a spiral trajectory, keep first interleave only
# --------------------------------------------------
gamma = 42.5756e6

nPix = 256
fov = 0.5
sLim = 50 * gamma * fov / nPix
gLim = 120e-3 * gamma * fov / nPix
dtGrad = 10e-6
dtADC = 2.5e-6

mag.setMagOverSamp(4)
mag.setGoldAng(False)
gLim = min(gLim, 1 / (dtADC * nPix))

lstArrK0, lstArrGrad = mag.getG_Spiral(
    fov, nPix, sLim, gLim, dtGrad,
    kRhoPhi=0.5 / (8 * pi)
)

# first interleave only
arrK0 = lstArrK0[0]
arrGrad = lstArrGrad[0]

arrK = mag.cvtGrad2Traj(arrGrad, dtGrad, dtADC)[0]
arrK += arrK0
arrK = arrK[:, :2].astype(np.float32)

# Save trajectory using plot()
save_clean_line_image(arrK[:, 0], arrK[:, 1], "spiral_trajectory.png")


# --------------------------------------------------
# 2) Compute PSF over 2xFOV and save in log scale
# --------------------------------------------------
psf = psf_from_traj_2d(arrK, nPix, eps=1e-6)
save_clean_log_image(psf, "spiral_psf_2xfov.png", floor_db=-60)


# --------------------------------------------------
# 3) Apply window and save windowed PSF in log scale
# --------------------------------------------------
wind = make_window_nd(nPix, nAx=2, wtype="poly", wpara=2.4)
psf_windowed = apply_corner_window_2d(psf, wind)
save_clean_log_image(psf_windowed, "spiral_psf_windowed.png", floor_db=-60)