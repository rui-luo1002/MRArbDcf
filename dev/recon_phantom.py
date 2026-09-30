import h5py
from numpy import *
from numpy.linalg import norm
import matplotlib.pyplot as plt
# Import svds (Singular Value Decomposition for sparse/truncated calculation)
from scipy.sparse.linalg import svds 
from scipy.signal.windows import gaussian

# --- Configuration ---
pathRes = "/mnt/d/LProject/DcfBenchmark/resource/"
dsid = "1b197efe-9865-43be-ac24-f237c380513e"
# dsid = "2588bfa8-0c97-478c-aa5a-487cc88a590d"
FILE_PATH = f'{pathRes[:-1]}/{dsid}.h5'
DATASET_PATH = '/dataset/data'
TARGET_NX = 320 # Target readout size after cropping (to remove oversampling)

with h5py.File(FILE_PATH, 'r') as f:
    print(f"Loading data from {DATASET_PATH}...")
    
    dataset = f[DATASET_PATH][:]
    
    data = dataset['data']
    head = dataset['head'] 
    print("kspace_data_field.dtype", data.dtype)
    
    if len(data) == 0:
        raise ValueError("No k-space data found.")
    
    print("kspace_data_field[0].shape:", data[0].shape)
    print("kspace_data_field[0].dtype:", data[0].dtype)
    
    nCh = head[0]['active_channels']
    nRO = data[0].shape[0]
    nX = nRO // nCh // 2
    
    nAcq = len(data)
    
    vecIdxPhEnc1 = head['idx']['kspace_encode_step_1']
    vecIdxPhEnc2 = head['idx']['kspace_encode_step_2'] 
    vecIdxSlice = head['idx']['slice']
    
    print(f"DEBUG: vecIdxPhEnc1 range: {vecIdxPhEnc1.min()} to {vecIdxPhEnc1.max()}")
    print(f"DEBUG: vecIdxPhEnc2 range: {vecIdxPhEnc2.min()} to {vecIdxPhEnc2.max()}")
    print(f"DEBUG: vecIdxSlice range: {vecIdxSlice.min()} to {vecIdxSlice.max()}")
    
    nY = vecIdxPhEnc1.max() + 1 # Ny
    nZ = vecIdxSlice.max() + 1
    
    print(f"Inferred Dimensions: Channels={nCh}, Readout Raw (Nx_raw)={nX}, Phase Y (Ny)={nY}, Phase Z (Nz)={nZ}")
    
    # read rawdata
    kspMulCh = zeros((nCh, nZ, nY, nX), dtype=complex64)
    for i in range(nAcq):
        iY = vecIdxPhEnc1[i]
        iZ = vecIdxSlice[i]
        arrS = data[i]
        
        kspMulCh[:, iZ, iY, :] = (arrS[0::2] + arrS[1::2]*1j).reshape(nCh, nX)
        
    # I write this jumble during debugging, but it succeed to recovory the dataset 
    kspMulCh = fft.fftn(kspMulCh, axes=(-3,))
    kspMulCh = fft.fftshift(kspMulCh, axes=(-3,))
    kspMulCh = fft.ifftn(kspMulCh, axes=(-3,))
    kspMulCh = fft.ifftshift(kspMulCh, axes=(-3,))
    kspMulCh = fft.fftn(kspMulCh, axes=(-3,))
    kspMulCh = fft.fftshift(kspMulCh, axes=(-3,))
    
    print(f"K-space array shape for reconstruction: {kspMulCh.shape}")
    
    # coil sensitivity map
    sigma = 0.01
    arrGauWin = gaussian(nZ, nZ*sigma)
    arrGauWin = multiply.outer(arrGauWin, gaussian(nY, nY*sigma))
    arrGauWin = multiply.outer(arrGauWin, gaussian(nX, nX*sigma))
    kspCsm = kspMulCh*arrGauWin[None,...]
    
    imgCsm = kspCsm.copy(); tupAxes = (-3,-2,-1)
    imgCsm = fft.ifftshift(imgCsm, axes=tupAxes)
    imgCsm = fft.ifftn(imgCsm, axes=tupAxes)
    imgCsm = fft.fftshift(imgCsm, axes=tupAxes)

    # coil images
    imgMulCh = kspMulCh.copy(); tupAxes = (-3,-2,-1)
    imgMulCh = fft.ifftshift(imgMulCh, axes=tupAxes)
    imgMulCh = fft.ifftn(imgMulCh, axes=tupAxes)
    imgMulCh = fft.fftshift(imgMulCh, axes=tupAxes)
    
    def crop_trans(img:ndarray):
        iY0 = nY//2-nZ//2; iY1 = iY0 + nZ
        iX0 = nX//2-nZ//2; iX1 = iX0 + nZ
        img = img[:,:,iY0:iY1,iX0:iX1] # crop
        img = img.transpose(0,3,2,1)[:,::-1,:,:]
        return img
        
    imgCsm = crop_trans(imgCsm)
    imgMulCh = crop_trans(imgMulCh)
    
    print(f"Full Image domain array shape: {imgMulCh.shape}")

    # reconstructed image
    nCh, nZ, nY, nX = imgMulCh.shape
    
    imgCsm *= sqrt(nZ*nY*nX)/norm(imgCsm.ravel())
    imgMulCh *= sqrt(nZ*nY*nX)/norm(imgMulCh.ravel())
    imgRec = sum(imgCsm.conj()*imgMulCh, axis=0) / (sum(abs(imgCsm)**2, axis=0) + 1e0) # stablizer is necessary here
    # imgRec = sqrt(sum(abs(imgMulCh)**2, axis=0))
    imgRec = squeeze(imgRec)
    
    imgRec /= percentile(abs(imgRec), 95)
    save(f"{pathRes[:-1]}/{dsid}.npy", imgRec) # save
    
    print(f"Final combined image volume shape: {imgRec.shape}")
    
    # plot
    # imgRec = imgCsm[0,...] # test
    sliceZ0 = imgRec[nZ//2, :, :]
    sliceY0 = imgRec[:, nY//2, :]
    sliceX0 = imgRec[:, :, nX//2]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))
    axes = ravel(axes)
    
    # z=0
    axes[0].imshow(abs(sliceZ0), origin="lower", cmap='gray')
    axes[0].set_title(f'z=0')
    axes[0].axis('off')
    
    # y=0
    axes[1].imshow(abs(sliceY0), origin="lower", cmap='gray')
    axes[1].set_title(f'y=0')
    axes[1].axis('off')
    
    # x=0
    axes[2].imshow(abs(sliceX0), origin="lower", cmap='gray')
    axes[2].set_title(f'x=0')
    axes[2].axis('off')
    
    # z=0
    axes[3].imshow(angle(sliceZ0), origin="lower", cmap='hsv', vmin=-pi, vmax=pi)
    axes[3].set_title(f'z=0')
    axes[3].axis('off')
    
    # y=0
    axes[4].imshow(angle(sliceY0), origin="lower", cmap='hsv', vmin=-pi, vmax=pi)
    axes[4].set_title(f'y=0')
    axes[4].axis('off')
    
    # x=0
    axes[5].imshow(angle(sliceX0), origin="lower", cmap='hsv', vmin=-pi, vmax=pi)
    axes[5].set_title(f'x=0')
    axes[5].axis('off')
    
    plt.show()