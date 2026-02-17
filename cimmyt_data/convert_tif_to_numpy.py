import os
import rasterio
import numpy as np
from tqdm import tqdm

path = "../S2_clipped_with_cldprb/"

UIDs = os.listdir(path)

for uid in tqdm(UIDs):
    dates = os.listdir(os.path.join(path, uid))
    for date in dates:
        bands = os.listdir(os.path.join(path, uid, date))
        npz = {}
        for band in bands:
            with rasterio.open(os.path.join(path, uid, date, band)) as src:
                npz[band.split(".")[1]] = src.read(1).astype(np.float32)
        os.makedirs(os.path.join("../../data/S2/npy/", uid), exist_ok=True)
        with open(os.path.join("../../data/S2/npy/", uid, date+'.npz'), "wb") as fp:
            np.savez(fp, **npz)
