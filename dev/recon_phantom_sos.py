import h5py
import numpy as np
import matplotlib.pyplot as plt
# Import svds (Singular Value Decomposition for sparse/truncated calculation)
from scipy.sparse.linalg import svds
from sigpy.mri.app import EspiritCalib

# --- Configuration ---
pathRes = "/mnt/d/LProject/DcfBenchmark/resource/"
# dsid = "1b197efe-9865-43be-ac24-f237c380513e"
dsid = "2588bfa8-0c97-478c-aa5a-487cc88a590d"
FILE_PATH = f'{pathRes[:-1]}/{dsid}.h5'
DATASET_PATH = '/dataset/data'
TARGET_NX = 320 # Target readout size after cropping (to remove oversampling)

with h5py.File(FILE_PATH, 'r') as f:
    print(f"Loading data from {DATASET_PATH}...")
    
    # Load the structured array (includes 'head' and 'data')
    dataset = f[DATASET_PATH][:]
    
    # Extract the k-space data field (array of arrays) and header field
    data = dataset['data']
    head = dataset['head'] 
    print("kspace_data_field.dtype", data.dtype)
    
    if len(data) == 0:
        raise ValueError("No k-space data found.")
    
    # --- Determine 3D Dimensions from metadata ---
    
    print("kspace_data_field[0].shape:", data[0].shape)
    print("kspace_data_field[0].dtype:", data[0].dtype)
    
    nCh = head[0]['active_channels']
    nRO = data[0].shape[0]
    
    # Calculate N_samples_readout (Nx_raw)
    nX = nRO // nCh // 2
    
    nAcq = len(data)
    
    # Infer the phase encoding steps (Ny and Nz) using the kspace indices
    vecIdxPhEnc1 = head['idx']['kspace_encode_step_1']
    vecIdxPhEnc2 = head['idx']['kspace_encode_step_2'] # Z dimension candidate 1 (Kz)
    vecIdxSlice = head['idx']['slice'] # Z dimension candidate 2 (Slice index)
    
    # Debugging output for indices
    print(f"DEBUG: vecIdxPhEnc1 range: {vecIdxPhEnc1.min()} to {vecIdxPhEnc1.max()}")
    print(f"DEBUG: vecIdxPhEnc2 range: {vecIdxPhEnc2.min()} to {vecIdxPhEnc2.max()}")
    print(f"DEBUG: vecIdxSlice range: {vecIdxSlice.min()} to {vecIdxSlice.max()}")
    
    nY = vecIdxPhEnc1.max() + 1 # Ny
    
    # nZ = vecIdxPhEnc2.max() + 1
    nZ = vecIdxSlice.max() + 1
    
    print(f"Inferred Dimensions: Channels={nCh}, Readout Raw (Nx_raw)={nX}, Phase Y (Ny)={nY}, Phase Z (Nz)={nZ}")
    
    # --- Robust Reshaping of K-space Data (Mapping using 3D indices) ---
    
    # Initialize the final k-space grid: [N_channels, Nz, Ny, Nx_raw]
    kspMulCh = np.zeros((nCh, nZ, nY, nX), dtype=np.complex64)
    
    # Iterate through all acquisitions and place them at their correct 3D index
    for i in range(nAcq):
        iY = vecIdxPhEnc1[i]
        iZ = vecIdxSlice[i] # Use the determined Z index array
        arrS = data[i]
        
        # Place data into the 4D grid: [Channel, Nz, Ny, nX]
        kspMulCh[:, iZ, iY, :] = (arrS[0::2] + arrS[1::2]*1j).reshape(nCh, nX)
    kspMulCh = np.fft.fftn(kspMulCh, axes=(-3,))
    kspMulCh = np.fft.fftshift(kspMulCh, axes=(-3,))
    kspMulCh = np.fft.ifftn(kspMulCh, axes=(-3,))
    
    print(f"K-space array shape for reconstruction: {kspMulCh.shape}")
    
    # Ecalib
    arrCsm = EspiritCalib(kspMulCh)
    print(arrCsm.shape)
    exit()
    plt.figure()
    plt.imshow()
    

    # --- Step 1: 3D Inverse FFT (iFFT) ---
    
    tupAxes = (-2,-1)
    _ = np.fft.ifftshift(kspMulCh, axes=tupAxes)
    _ = np.fft.ifftn(_, axes=tupAxes)
    imgMulCh = np.fft.fftshift(_, axes=tupAxes)
    
    iY0 = nY//2-nZ//2; iY1 = iY0 + nZ
    iX0 = nX//2-nZ//2; iX1 = iX0 + nZ
    imgMulCh = imgMulCh[:,:,iY0:iY1,iX0:iX1] # crop
    
    print(f"Full Image domain array shape: {imgMulCh.shape}")

    # --- Step 1b: Crop Readout Oversampling (nX) ---
    
    nCh, nZ, nY, nX = imgMulCh.shape
    
    # --- Step 2: SVD Channel Combination using svds (3D) ---
    
    imgMulCh = imgMulCh.transpose(0,3,2,1)[:,::-1,:,:]
    
    
    # Reshape back to 3D image volume: [Nz, Ny, nX]
    imgRec = imgFlat.reshape(nZ, nY, nX)
    imgRec /= np.percentile(np.abs(imgRec), 95)
    np.save(f"{pathRes[:-1]}/{dsid}.npy", imgRec) # save
    
    print(f"Final combined image volume shape: {imgRec.shape}")
    
    # --- Step 3: Display Center Slices ---
    
    # Extract slices
    sliceZ0 = imgRec[nZ//2, :, :] # Z fixed, showing Y-X plane
    sliceY0 = imgRec[:, nY//2, :] # Y fixed, showing Z-X plane
    sliceX0 = imgRec[:, :, nX//2] # X fixed, showing Z-Y plane
    
    # Display setup
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))
    axes = np.ravel(axes)
    
    # 1. Axial (Z plane, typically the acquisition plane)
    axes[0].imshow(np.abs(sliceZ0), origin="lower", cmap='gray')
    axes[0].set_title(f'Axial Slice (Z={nZ//2}) | Y x X ({nY}x{nX})')
    axes[0].axis('off')
    
    # 2. Coronal (Y plane)
    axes[1].imshow(np.abs(sliceY0), origin="lower", cmap='gray')
    axes[1].set_title(f'Coronal Slice (Y={nY//2}) | Z x X ({nZ}x{nX})')
    axes[1].axis('off')
    
    # 3. Sagittal (X plane)
    axes[2].imshow(np.abs(sliceX0), origin="lower", cmap='gray')
    axes[2].set_title(f'Sagittal Slice (X={nX//2}) | Z x Y ({nZ}x{nY})')
    axes[2].axis('off')
    
    # 4. Axial (Z plane, typically the acquisition plane)
    axes[3].imshow(np.angle(sliceZ0), origin="lower", cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[3].set_title(f'Axial Slice (Z={nZ//2}) | Y x X ({nY}x{nX})')
    axes[3].axis('off')
    
    # 5. Coronal (Y plane)
    axes[4].imshow(np.angle(sliceY0), origin="lower", cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[4].set_title(f'Coronal Slice (Y={nY//2}) | Z x X ({nZ}x{nX})')
    axes[4].axis('off')
    
    # 6. Sagittal (X plane)
    axes[5].imshow(np.angle(sliceX0), origin="lower", cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[5].set_title(f'Sagittal Slice (X={nX//2}) | Z x Y ({nZ}x{nY})')
    axes[5].axis('off')
    
    fig.suptitle('3D Reconstructed MRI Volume (iFFT + SVD Combination)')
    plt.show()