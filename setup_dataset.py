# setup_dataset.py

from config import FIFTYONE_DATASET_NAME, BDD100K_DATA_DIR, LOCAL_DATASET_NAME
from fiftyone_utils import setup_dataset, preview_samples, launch_dataset_gui

if __name__ == "__main__":
    setup_dataset(fiftyone_dataset_zoo_name_or_url=FIFTYONE_DATASET_NAME,
                  source_local_dir=BDD100K_DATA_DIR,
                  save_local_dataset_name=LOCAL_DATASET_NAME)

    preview_samples(LOCAL_DATASET_NAME)
    launch_dataset_gui(LOCAL_DATASET_NAME)
