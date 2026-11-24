"""Image discovery and loading for GSAFD inference."""

import os

import cv2


def list_images(directory, filter_keyword=None):
    """Return parallel lists of image paths and filename stems."""
    images = []
    names = []
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for root, _, files in os.walk(directory):
        if filter_keyword and filter_keyword not in root:
            continue

        for filename in sorted(files):
            if filename.lower().endswith(valid_extensions):
                images.append(os.path.join(root, filename))
                names.append(os.path.splitext(filename)[0])

    return images, names


def get_image(path, height=256, width=256, flag=False):
    """Read grayscale or, when flag is True, extract visible luminance."""
    read_color = flag is True
    mode = cv2.IMREAD_COLOR if read_color else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(os.fspath(path), mode)
    if image is None:
        raise FileNotFoundError(f"cv2 could not read image: {path}")

    if read_color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    if height is not None and width is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    return image
