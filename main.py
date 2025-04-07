# main.py

import fiftyone as fo
import os

from fiftyone_utils import load_filtered_samples
from config import OUTPUT_DIR, LOCAL_DATASET_NAME, DIFFUSION_MODEL_NAME

from diffusers import StableDiffusionPipeline
import torch


def generate_prompt(sample):
    # 향후 CLIP 기반 확장 가능
    tags = sample.tags
    if "fog" in tags:
        return "a foggy road with low visibility"
    return "a blurry image of a street"


def generate_image(pipe, prompt):
    return pipe(prompt).images[0]


def save_sample(image, prompt, index, dataset, output_dir):
    filepath = os.path.join(output_dir, f"gen_{index}.png")
    image.save(filepath)

    sample = fo.Sample(filepath=filepath)
    sample["prompt"] = prompt
    sample.tags.append("generated")
    dataset.add_sample(sample)


def auto_data_refiner(local_dataset_name, save_dir):
    view = load_filtered_samples(local_dataset_name)
    pipe = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL_NAME, torch_dtype=torch.float16).to("cuda")

    gen_dataset_name = f"{local_dataset_name}_refined"
    if gen_dataset_name in fo.list_datasets():
        fo.delete_dataset(gen_dataset_name)

    gen_dataset = fo.Dataset(name=gen_dataset_name)

    for i, sample in enumerate(view[:10]):
        prompt = generate_prompt(sample)
        image = generate_image(pipe, prompt)
        save_sample(image, prompt, i, gen_dataset, save_dir)

    return gen_dataset


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gen_dataset = auto_data_refiner(local_dataset_name=LOCAL_DATASET_NAME, save_dir=OUTPUT_DIR)

    session = fo.launch_app(gen_dataset)
    session.wait()
