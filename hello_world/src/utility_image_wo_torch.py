import collections.abc as abc
import io
import os
import tempfile

import cv2
import ipyplot
import IPython.display as display
import matplotlib
import matplotlib as mpl
import matplotlib.axes
import numpy as np
from IPython import get_ipython
from PIL import Image, ImageOps


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def seaborn_to_array(ax, dpi=200):
    buf = io.BytesIO()
    ax.figure.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def flatten_list(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list) or isinstance(item, tuple):
            flattened_list.extend(flatten_list(item))
        elif isinstance(item, np.ndarray):
            flattened_list.append(item)
        else:
            raise ValueError(f"Unsupported type: {type(item)}")
    return flattened_list


def png_to_jpg(image):
    # Open the PNG image
    img = Image.fromarray(image)
    # Ensure the image has an alpha channel (RGBA)
    if img.mode == "RGBA":
        # Split the image into its red, green, blue, and alpha channels
        r, g, b, a = img.split()

        # Create a grayscale version of the alpha channel
        gray_alpha = a.convert("L")

        # Merge the RGB channels with the grayscale version of the alpha channel
        img = Image.merge("RGB", (r, g, b))

        # Create a new image by pasting the grayscale onto the image
        img_with_gray_alpha = Image.composite(
            img, Image.new("RGB", img.size, (0, 0, 0)), gray_alpha
        )
        return np.array(img_with_gray_alpha)
    else:
        return image


@static_vars(disable=False)
def plot_images(images, img_width=None, max_images=5, parallel=False, parallel_size=5):
    if plot_images.disable:
        return
    if isinstance(images, matplotlib.axes.Axes):
        plot_images(seaborn_to_array(images), img_width=img_width, max_images=max_images)
        return
    if not isinstance(images, abc.Sequence):
        images = [images]
    images = images[:max_images]
    L = len(images)
    images = flatten_list(images)

    if not is_notebook():
        for image in images:
            with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
                if image.max() <= 1:
                    image = (image * 255).astype(np.uint8)
                if len(image.shape) == 3 and image.shape[2] == 4:
                    image = png_to_jpg(image)
                cv2.imwrite(f.name, image)
                # os.system(
                #     f"convert {f.name} -resize {img_width if img_width else 200} -alpha off sixel:-"
                # )
                print()
                os.system(f"img2sixel -w{img_width if img_width else 200} {f.name}")
                print()
        return

    cols = len(images) // L
    if len(images) == 1:
        images = images[0]
        height = max(images.shape) if img_width == -1 else img_width if img_width else 200
        if images.max() <= 1:
            images = (images * 255).astype(np.uint8)
        display(ImageOps.contain(Image.fromarray(images), (height, height)))
    else:
        if not parallel:
            for i in range(0, len(images), cols):
                height = images[i].shape[0] if img_width == -1 else img_width if img_width else 200
                ipyplot.plot_images(
                    images[i : i + cols],
                    img_width=height,
                )
        else:
            ipyplot.plot_images(
                images,
                max_images=parallel_size,
                img_width=img_width if img_width else 200,
            )
