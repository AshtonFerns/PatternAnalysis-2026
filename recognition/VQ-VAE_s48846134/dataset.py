import os
import re
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

def get_data_path():
    if os.path.isdir("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"):
        return "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    else:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "keras_slices_data")

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c : c + 1][arr == c] = 1
    return res

def load_data_2D(imageNames, normImage=False, categorical=False, 
        dtype=np.float32, getAffines=False, early_stop=False):
    """
    Load medical image data from names; cases list provided into a list for each.
    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.
    normImage: bool (normalise the image 0.0-1.0)
    early_stop: Stop loading prematurely, leaves arrays mostly empty, for
        quick loading and testing scripts.
    """
    affines = []

    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching="unchanged")
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

def parse_filename(file):
    expression = re.compile(r"case_(\d+)_week_(\d+)_slice_(\d+)\.nii")
    result = expression.search(os.path.basename(file))
    return int(result.group(1)), int(result.group(2)), int(result.group(3))

def list_split(split):
    folder = os.path.join(get_data_path(), 'keras_slices_' + split)
    paths = [os.path.join(folder, name) for name in sorted(os.listdir(folder), key=parse_filename)]
    kept = []
    for p in paths:
        h, w = nib.load(p).shape[:2]
        if (h, w) == (256, 128):
            kept.append(p)
        else:
            print(f"{split}: {os.path.basename(p)} is {h}x{w} not {256}x{128}")
    return kept

def check_patient_leakage():
    splits = {"train": set(), "validate": set(), "test": set()}
    patient_splits = {}
    for split in splits:
        for file in list_split(split):
            patient_id, week, number = parse_filename(file)
            if patient_id in patient_splits and patient_splits[patient_id] != split:
                raise RuntimeError(f"patient {patient_id} is in {split} and {patient_splits[patient_id]}")
            patient_splits[patient_id] = split
            splits[split].add(patient_id)
    return splits

def normalise(img):
    hi = np.percentile(img, 99.5)
    return np.clip(img / hi, 0, 1).astype(np.float32)

def load_split(split):
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    os.makedirs(cache, exist_ok=True)
    cache_path = os.path.join(cache, f"{split}.pt")
    if os.path.isfile(cache_path):
        return torch.load(cache_path, weights_only=False)
    paths = list_split(split)
    raw = load_data_2D(paths)
    images = np.stack([normalise(im) for im in raw])
    data = {
        "images": torch.from_numpy(images)[:, None].float(),
        "keys": [parse_filename(p) for p in paths],
    }
    torch.save(data, cache_path)
    return data

class Slices(Dataset):
    def __init__(self, data, augment=False):
        self.images = data["images"]
        self.keys = data["keys"]
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = self.images[i]
        if self.augment:
            angle = (torch.rand(1) * 16 - 8) * torch.pi / 180
            scale = torch.empty(1).uniform_(0.92, 1.08)
            shift = torch.empty(2).uniform_(-0.04, 0.04)
            cos, sin = angle.cos(), angle.sin()
            theta = torch.tensor([[scale * cos, -scale * sin, shift[0]],
                                  [scale * sin, scale * cos, shift[1]]], dtype=x.dtype)
            grid = torch.nn.functional.affine_grid(theta[None], x[None].shape, align_corners=False)
            x = torch.nn.functional.grid_sample(x[None], grid, align_corners=False)[0]
            gamma = torch.empty(1).uniform_(np.log(0.8), np.log(1.25)).exp()
            x = x.clamp(0, 1) ** gamma
        return x, i

def get_loaders(batch_size=32):
    check_patient_leakage()
    data = {s: load_split(s) for s in ("train", "validate", "test")}
    return {
        "train": DataLoader(Slices(data["train"], augment=True), batch_size=batch_size, shuffle=True, drop_last=True),
        "validate": DataLoader(Slices(data["validate"]), batch_size=batch_size),
        "test": DataLoader(Slices(data["test"]), batch_size=batch_size),
    }


if __name__ == "__main__":
    check_patient_leakage()