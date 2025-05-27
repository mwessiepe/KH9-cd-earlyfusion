import os
import sys
import time
import yaml
import subprocess
import webbrowser
import geopandas as gpd
from torchgeo.datasets.utils import BoundingBox

from src.train import train
from src.test import test
from src.predict import predict


def read_rois_from_geopackage(path, layer=None):
    """
    Reads geometries from a GeoPackage and converts each geometry into a TorchGeo BoundingBox.
    """
    gdf = gpd.read_file(path, layer=layer)
    bboxes = []

    for geom in gdf.geometry:
        minx, miny, maxx, maxy = geom.bounds
        bbox = BoundingBox(minx, maxx, miny, maxy, 0, 1)  # Temporal bounds set to 0 and 1
        bboxes.append(bbox)

    return bboxes


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py path/to/config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required config fields
    required = [
        "mode", "task", "experiment_name", "patch_size",
        "results_root", "log_dir", "rois_path", "predictions_root"
    ]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config field: '{field}'")

    # Derived fields
    config["experiment_dir"] = os.path.join(config["results_root"], config["experiment_name"])
    config["rois"] = read_rois_from_geopackage(config["rois_path"])
    config.setdefault("checkpoint_name", None)
    config.setdefault("aoi", None)

    # Dispatch mode
    mode = config["mode"]

    # Launch TensorBoard
    tb_process = subprocess.Popen([
        "tensorboard",
        "--logdir", os.path.join(config["log_dir"], config["experiment_name"]),
        "--port", "6006"  # optional, choose port
    ])

    time.sleep(1)
    webbrowser.open("http://localhost:6006")

    if mode == "train":
        try:
            train(
                old_images_dir=config["old_images_dir"],
                new_images_dir=config["new_images_dir"],
                bag_buildings_dir=config["bag_buildings_dir"],
                experiment_name=config["experiment_name"],
                experiment_dir=config["experiment_dir"],
                log_dir=config["log_dir"],
                model=config["model"],
                backbone=config["backbone"],
                batch_size=config["batch_size"],
                patch_size=config["patch_size"],
                learning_rate=config["learning_rate"],
                num_dataloader_workers=config["num_dataloader_workers"],
                val_split_pct=config["val_split_pct"],
                test_split_pct=config["test_split_pct"],
                checkpoint_name=config["checkpoint_name"],
                rois=config["rois"],
                aoi=config["aoi"],
                task=config["task"]
            )

        finally: 
            tb_process.terminate()

    elif mode == "test":
        test(**config)
    elif mode == "predict":
        predictions_dir = os.path.join(config["predictions_root"], config["experiment_name"])
        predict(predictions_dir=predictions_dir, **config)
    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    main()
