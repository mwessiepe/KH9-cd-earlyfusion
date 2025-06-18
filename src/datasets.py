import torch
import matplotlib.pyplot as plt
from torchgeo.datasets import GeoDataset, RasterDataset, VectorDataset, IntersectionDataset
from rasterio.enums import Resampling

class AerialImages(RasterDataset):
    filename_glob = 'RGB*.tiff'
    is_image = True

class AerialImagesAverage(RasterDataset):
    filename_glob = 'RGB*.tiff'
    is_image = True
    resampling = Resampling.average

class AerialImagesCubic(RasterDataset):
    filename_glob = 'RGB*.tiff'
    is_image = True
    resampling = Resampling.cubic

class AerialImagesLanczos(RasterDataset):
    filename_glob = 'RGB*.tiff'
    is_image = True
    resampling = Resampling.lanczos


class KH9Images(RasterDataset):
    filename_glob = 'output_feature_*.tif'
    is_image = True
    # res = 0.25
    resampling = Resampling.bilinear


class KH9ImageFull(RasterDataset):
    filename_glob = 'D3C1217-100203A010_epsg28992_clip.tif'
    is_image = True
    # res = 0.25
    # resampling = Resampling.bilinear

class KH9ImageFullBilinear(RasterDataset):
    filename_glob = 'D3C1217-100203A010_epsg28992_clip.tif'
    is_image = True
    # res = 0.25
    resampling = Resampling.bilinear

class KH9ImageFullCubic(RasterDataset):
    filename_glob = 'D3C1217-100203A010_epsg28992_clip.tif'
    is_image = True
    # res = 0.25
    resampling = Resampling.cubic

class KH9ImageFullLanczos(RasterDataset):
    filename_glob = 'D3C1217-100203A010_epsg28992_clip.tif'
    is_image = True
    # res = 0.25
    resampling = Resampling.lanczos


class BagBuildings(VectorDataset):
    filename_glob = 'classified_buildings_binary.gpkg'
    is_image = False
    res = 0.25
    
class BitemporalIntersectionDataset(GeoDataset):
    def __init__(self, dataset_old, dataset_new, vector_dataset, transforms=None):
        # We don't call super().__init__ to avoid default index assignment.
        self.transforms = transforms
        # Create individual intersections for the old and new datasets.
        self.dataset_old = IntersectionDataset(dataset_old, vector_dataset)
        self.dataset_new = IntersectionDataset(dataset_new, vector_dataset)
        
        # Set the internal index (you probably already have this)
        self._index = IntersectionDataset(dataset_old, dataset_new).index
    
    def __len__(self):
        return len(self.dataset_new)
    
    def __getitem__(self, query):
        sample_old = self.dataset_old[query]
        sample_new = self.dataset_new[query]

        old_img = sample_old["image"]  # [1, H, W]
        new_img = sample_new["image"]  # [3, H, W]

        image = torch.cat([old_img, new_img], dim=0)  # [4, H, W]

        mask = sample_old["mask"]
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        mask = mask.unsqueeze(0)  # [1, H, W]

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return {
            "image": image,  # [4, H, W]
            "mask": mask,    # [1, H, W]
            "bounds": query,
            **({"crs": sample_old["crs"]} if "crs" in sample_old else {})
        }


    @property
    def bounds(self):
        return self._index.bounds

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def res(self):
        return self.dataset_old.res