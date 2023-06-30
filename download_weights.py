"""Downloads GPT-2 weights from OpenAI and dumps the tensors in raw binary format.

Weight tensors are transposed so we can easily load them into PyTorch/zig. 

Based on https://github.com/openai/gpt-2/blob/master/download_model.py.
"""

import os

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm

model = "models/124M"

# Download the model weights from OpenAI if they don't already exist.
if not os.path.exists(model):
    os.makedirs(model)
    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        resp = requests.get(
            f"https://openaipublic.blob.core.windows.net/gpt-2/{model}/{filename}",
            stream=True,
        )

        with open("{model}/{filename}", "wb") as file_:
            file_size = int(resp.headers["content-length"])
            chunk_size = 1000
            with tqdm(
                ncols=100, desc=f"Fetching {filename}", total=file_size, unit_scale=True
            ) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes.
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    file_.write(chunk)
                    pbar.update(chunk_size)


# Dump the model weights in plain binary if they don't already exist.
weights_dir = f"{model}/raw"
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)
    checkpoint = tf.train.load_checkpoint(model)
    variables = sorted(list(checkpoint.get_variable_to_shape_map().keys()))
    with tqdm(
        ncols=100, desc=f"Dumping raw weights", total=len(variables), unit_scale=True
    ) as pbar:
        for name in variables:
            tensor = checkpoint.get_tensor(name).astype(np.float32).squeeze()
            # Store weight tensors in column major format.
            if name.endswith("/w"):
                tensor = tensor.T
            fname = name.replace("/", "-")
            with open(f"{weights_dir}/{fname}", "wb") as file_:
                file_.write(tensor.reshape(-1).tobytes())
            pbar.update(1)
