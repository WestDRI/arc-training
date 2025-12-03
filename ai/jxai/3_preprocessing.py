import os
import polars as pl
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import grain.python as grain
import jax.numpy as jnp

with open("2_dataset.py") as file:
    exec(file.read())

class NormAndCast(grain.MapTransform):
    """Transform class to normalize and cast images to float32."""
    def map(self, element):
        element['image'] = jnp.array(element['image'], dtype=jnp.float32) / 255.0
        return element

class BbCrop(grain.MapTransform):
    """Transform class to crop images to their bounding boxes."""
    def map(self, element):
        img = element['image']
        bbx = element['bbx']
        bby = element['bby']
        bbwidth = element['bbwidth']
        bbheight = element['bbheight']
        img_cropped = img[bby:bby+bbheight, bbx:bbx+bbwidth]
        element['image'] = img_cropped
        return element

transformations = [NormAndCast(), BbCrop()]
