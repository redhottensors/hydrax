"""Provides utilities for loading images with Pillow (PIL)."""

from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image, ImageCms

_srgb = ImageCms.createProfile(colorSpace='sRGB')

CONVERT_PERCEPTUAL = (ImageCms.Intent.PERCEPTUAL, 0x0400)
"""Perceptual colorspace conversion intent with ``HIGHRESPRECALC``."""

CONVERT_RELATIVE = (ImageCms.Intent.RELATIVE_COLORIMETRIC, 0x2400)
"""Relative colorimetric colorspace conversion intent with ``HIGHRESPRECALC`` and ``BLACKPOINTCOMPENSATION``."""

COEFS_FRACTIONAL = np.asarray([[255.0], [0.0]])
"""Coefficients to convert sRGB(A) data to a floating point representation in the range [0.0, 1.0]."""

COEFS_IMAGENET_SRGB = np.asarray([
    [58.395, 57.120, 57.375],
    [123.675/58.395, 116.280/57.120, 103.530/57.375]
])
"""ImageNet sRGB standardization coefficients, with alpha in the range [0.0, 1.0]."""

COEFS_IMAGENET_SRGB_ALPHA = np.asarray([
    [58.395, 57.120, 57.375, 255.0],
    [123.675/58.395, 116.280/57.120, 103.530/57.375, 0.0]
])
"""ImageNet sRGBA standardization coefficients, with alpha in the range [0.0, 1.0]."""

def make_coefs_balanced(scale: float = 1.0) -> np.ndarray:
    """Returns a set of coefficients for converting sRGB to a balanced floating point representation in the range
    [``-scale``, ``scale``].
    """

    return np.asarray([[127.5 / scale], [scale]])

def make_coefs_balanced_alpha(scale: float = 1.0) -> np.ndarray:
    """Returns a set of coefficients for converting sRGBA to a balanced floating point representation in the range
    [``-scale``, ``scale``], with alpha in the range [0.0, 1.0].
    """

    coef = 127.5 / scale
    return np.asarray([[coef, coef, coef, 255.0], [scale, scale, scale, 0.0]])

def open_sRGB(
    fp,
    mode: str = "RGB",
    convert: Tuple[ImageCms.Intent, int] = CONVERT_RELATIVE
) -> Image.Image:
    """Opens an image using PIL and ensures it is in the sRGB colorspace.

    :param fp: The image to open.
    :type fp: path-like object, file stream, or buffer:
    :param mode: The pixel format for the image. Defaults to "RGB", but could also be "RGBA" or "RGBa".
    :param convert: The colorspace conversion intent and flags for converting to sRGB. The default, `CONVERT_RELATIVE`
        is usually correct. If the specified intent cannot be used with the image's embedded color profile, the
        profile's default intent is used instead.
    :type convert: (ImageCMS.Intent, ImageCMS.Flags)
    """

    img = Image.open(fp)

    # ensure image is in sRGB color space
    icc_raw = img.info.get("icc_profile")
    if icc_raw:
        profile = ImageCms.ImageCmsProfile(BytesIO(icc_raw))

        if not ImageCms.isIntentSupported(profile, convert[0], ImageCms.Direction.INPUT):
            intent = ImageCms.getDefaultIntent(profile)
            flags = convert[1] if intent == ImageCms.Intent.RELATIVE_COLORIMETRIC else convert[1] & ~0x2000
            convert = (intent, flags)

        if img.mode == mode:
            ImageCms.profileToProfile(
                img,
                profile,
                _srgb,
                renderingIntent=convert[0],
                inPlace=True,
                flags=convert[1]
            )
        else:
            img = ImageCms.profileToProfile(
                img,
                profile,
                _srgb,
                renderingIntent=convert[0],
                outputMode=mode,
                inPlace=False,
                flags=convert[1]
            )
    elif img.mode != mode:
        img = img.convert(mode)

    return img

def write_HWC(image: Image.Image | np.ndarray, array: np.ndarray, coefs: np.ndarray | None = None) -> None:
    """Writes a PIL image or image data array to the specified numpy array with shape (Height, Width, Channels).

    :param image: Image data to write.
    :param array: Destination array.
    :param coefs: Representation conversion coefficients. If not specified, the destination array will have the range
        [0, 255].

    .. tip::
        HWC or "channels-last" is the usual format for images in JAX.
    """

    np.copyto(array, np.asarray(image), casting="safe")
    if coefs is not None:
        array /= coefs[0]
        array -= coefs[1]

def write_CHW(image: Image.Image | np.ndarray, array: np.ndarray, coefs: np.ndarray | None = None) -> None:
    """Writes a PIL image or image data array to the specified numpy array with shape (Channels, Height, Width).

    :param image: Image data to write.
    :param array: Destination array.
    :param coefs: Representation conversion coefficients. If not specified, the destination array will have the range
        [0, 255].

    .. tip::
        CHW or "channels-first" is the usual format for images in PyTorch.
    """

    np.copyto(array, np.asarray(image).transpose((2, 0, 1)), casting="safe")
    if coefs is not None:
        array /= coefs[0, :, None, None]
        array -= coefs[1, :, None, None]
