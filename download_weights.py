"""Downloads GPT-2 checkpoints from OpenAI. 

Weight tensors are transposed and dumped in raw binary so they can easily be loaded into 
Zig/PyTorch. The unicode->byte encoder is statically generated and dumped to json.

Based on https://github.com/openai/gpt-2.
"""

import json
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


# Dump the model weights in raw binary if they don't already exist.
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


# Statically create and dump the unicode->bytes encoder.
def bytes_to_unicode():
    """Returns list of utf-8 byte and a corresponding list of unicode strings."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    # !!NOTE!!: Unlike OpenAI's implementation, we dump out unicode->bytes so we don't
    # have to deal with non-string JSON keys.
    return dict(zip(cs, bs))


with open(f"{model}/bytes_encoder.json", "w") as file_:
    json.dump(bytes_to_unicode(), file_)
