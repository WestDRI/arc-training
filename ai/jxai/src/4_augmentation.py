import os
import polars as pl
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import grain.python as grain
import jax.numpy as jnp

import dm_pix as pix
from jax import random

with open("3_preprocessing.py") as file:
    exec(file.read())

key = random.key(0)

new_image = pix.random_crop(
    key=key,
    image=img,
    crop_sizes=(128,128,3))
