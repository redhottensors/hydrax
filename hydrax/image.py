"""Provides colorspace-aware utilities for loading and processing images with
`Pillow (PIL) <https://python-pillow.org/>`_.

The primary entry-point to this module is :func:`open_srgb`.

.. tip::
    You can ensure this module has its dependencies by installing hydrax with the "image" extra via
    ``pip install hydrax[image,...]``.
"""

from io import BytesIO
from functools import lru_cache
from typing import Any, TypeAlias, TYPE_CHECKING, final

import numpy as np

from PIL import Image, ImageCms

try:
    from numpy.typing import ArrayLike, DTypeLike
except ImportError:
    if not TYPE_CHECKING:
        ArrayLike: TypeAlias = Any
        DTypeLike: TypeAlias = Any

_SRGB = ImageCms.createProfile(colorSpace='sRGB')

@final
class Coefs:
    """Dtype-agnostic coefficients for image representation and colorspace conversion.

    :param coefs: Coefficient array.

    .. tip::
        Many useful predefined coefficients are provided: :attr:`FRACTIONAL`, :attr:`IMAGENET_SRGB`,
        :attr:`IMAGENET_SRGB_ALPHA`, :func:`balanced`
    """

    __slots__ = ("_coefs", "_cache")

    FRACTIONAL: "Coefs"
    IMAGENET_SRGB: "Coefs"
    IMAGENET_SRGB_ALPHA: "Coefs"

    def __init__(self, coefs: ArrayLike):
        self._coefs = coefs
        self._cache: dict[np.dtype, np.ndarray] = {}

    def __getitem__(self, dtype: DTypeLike) -> np.ndarray:
        dtype = np.dtype(dtype)

        coefs = self._cache.get(dtype)
        if coefs is None:
            coefs = np.asarray(self._coefs, dtype=dtype)
            coefs.setflags(write=False)
            self._cache[dtype] = coefs

        return coefs

    @lru_cache(maxsize=8)
    @staticmethod
    def balanced(scale: float = 1.0, *, alpha: bool = False) -> "Coefs":
        """Returns a set of coefficients for converting sRGB in the range ``[0, 255]`` to a balanced floating point
        representation in the range ``[-scale, scale]``.

        :param scale: The scale of the balanced representation.
        :param alpha: If ``True``, alpha channel coefficients for output in the range ``[0.0, 1.0]`` are included.
        """

        coef = 127.5 / scale

        if alpha:
            return Coefs([[coef, coef, coef, 255.0], [scale, scale, scale, 0.0]])
        else:
            return Coefs([[coef, coef, coef], [scale, scale, scale]])

Coefs.FRACTIONAL = Coefs([[255.0], [0.0]])
"""Coefficients to convert sRGB(A) data in the range ``[0, 255]`` to a floating point representation in the range
``[0.0, 1.0]``."""

Coefs.IMAGENET_SRGB = Coefs([
    [58.395, 57.120, 57.375],
    [123.675/58.395, 116.280/57.120, 103.530/57.375]
])
"""ImageNet sRGB standardization coefficients for the input range of ``[0, 255]`` and no alpha channel."""

Coefs.IMAGENET_SRGB_ALPHA = Coefs([
    [58.395, 57.120, 57.375, 255.0],
    [123.675/58.395, 116.280/57.120, 103.530/57.375, 0.0]
])
"""ImageNet sRGBA standardization coefficients for the input range of ``[0, 255]`` and alpha output in the range
``[0.0, 1.0]``.
"""

_LINEAR_SRGB_TO_LMS = Coefs([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005]
])

_LMS_TO_OKLAB = Coefs([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660]
])

_LMS_TO_LINEAR_SRGB = Coefs([
    [4.0767416621, -3.3077115913, 0.2309699292],
    [-1.2684380046, 2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, 1.7076147010]
])

_OKLAB_TO_LMS = Coefs([
    [1.0, 0.3963377774, 0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480]
])

CONVERT_PERCEPTUAL = (ImageCms.Intent.PERCEPTUAL, 0x0400)
"""Perceptual colorspace conversion intent with ``HIGHRESPRECALC``."""

CONVERT_RELATIVE = (ImageCms.Intent.RELATIVE_COLORIMETRIC, 0x2400)
"""Relative colorimetric colorspace conversion intent with ``HIGHRESPRECALC`` and ``BLACKPOINTCOMPENSATION``."""

def set_max_image_pixels(threshold: int | None) -> None:
    """Sets the PIL decompression bomb threshold.

    :param threshold: The error threshold to set, or ``None`` to disable the check entirely. A warning will be raised at
        half the specified threshold.

    .. caution::
        This is a global setting which affects all PIL ``open`` operations, including :func:`open_srgb`. See
        `PIL.Image.open <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open>`_ for more
        details.
    """

    Image.MAX_IMAGE_PIXELS = None if threshold is None else (threshold + 1) // 2

def open_srgb(
    fp,
    *,
    mode: str = "RGB",
    convert: tuple[ImageCms.Intent, int] = CONVERT_RELATIVE,
    formats: list[str] | tuple[str, ...] | None = None
) -> Image.Image:
    """Opens an image using PIL and ensures it is in the `sRGB <https://en.wikipedia.org/wiki/SRGB>`_ colorspace.

    :param fp: The image to open.
    :type fp: path-like object, file stream, or buffer
    :param mode: The pixel format for the image. Defaults to ``"RGB"``, but could also be ``"RGBA"`` or ``"RGBa"``.
    :param convert: The colorspace conversion intent and flags for converting to sRGB. The default,
        :attr:`CONVERT_RELATIVE` is usually correct. If the specified intent
        `cannot be used <https://pillow.readthedocs.io/en/stable/reference/ImageCms.html#PIL.ImageCms.isIntentSupported>`_
        with the image's embedded color profile, the profile's
        `default intent <https://pillow.readthedocs.io/en/stable/reference/ImageCms.html#PIL.ImageCms.getDefaultIntent>`_
        is used instead.
    :type convert:
        `PIL.ImageCMS.Intent, PIL.ImageCMS.Flags <https://pillow.readthedocs.io/en/stable/reference/ImageCms.html>`_
    :param formats: A list or tuple of formats to attempt to load the file in. See
        `PIL.Image.open <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open>`_ for more
        details.

    .. tip::
        After opening an image, you can write it to a :class:`hydrax.Dataloader` ``loader_func`` array using
        :func:`write_hwc` or :func:`write_chw`.

    .. seealso::
        :attr:`CONVERT_RELATIVE`, :attr:`CONVERT_PERCEPTUAL`, :func:`set_max_image_pixels`
    """

    img = Image.open(fp, formats=formats)

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
                _SRGB,
                renderingIntent=convert[0],
                inPlace=True,
                flags=convert[1]
            )
        else:
            img = ImageCms.profileToProfile(
                img,
                profile,
                _SRGB,
                renderingIntent=convert[0],
                outputMode=mode,
                inPlace=False,
                flags=convert[1]
            )
    elif img.mode != mode:
        img = img.convert(mode)

    return img

def _convert_hwc(array: np.ndarray, coefs: np.ndarray) -> None:
    array /= coefs[0]
    array -= coefs[1]

def write_hwc(image: ArrayLike, array: np.ndarray, coefs: Coefs | None = None) -> None:
    """Writes a PIL image or HWC image data array to the specified NumPy array with shape HWC.

    :param image: Image data to write.
    :type image: array_like
    :param array: Destination array.
    :param coefs: Representation conversion coefficients. If not specified, the destination array will have the same
        representation range as the source image. In the case of a PIL image, that will be ``[0, 255]``, which is
        generally not useful.

    .. tip::
        HWC or "channels-last" is the usual format for images in JAX and PIL.
    """

    np.copyto(array, np.asarray(image), casting="safe")

    if coefs is not None:
        _convert_hwc(array, coefs[array.dtype])

def _convert_chw(array: np.ndarray, coefs: np.ndarray) -> None:
    array /= coefs[0, :, None, None]
    array -= coefs[1, :, None, None]

def write_chw(image: ArrayLike, array: np.ndarray, coefs: Coefs | None = None) -> None:
    """Writes a PIL image or HWC image data array to the specified NumPy array with shape CHW.

    :param image: Image data to write.
    :type image: array_like
    :param array: Destination array.
    :param coefs: Representation conversion coefficients. If not specified, the destination array will have the same
        representation range as the source image. In the case of a PIL image, that will be ``[0, 255]``, which is
        generally not useful.

    .. tip::
        CHW or "channels-first" is the usual format for images in PyTorch.
    """

    np.copyto(array, np.asarray(image).transpose((2, 0, 1)), casting="safe")

    if coefs is not None:
        _convert_chw(array, coefs[array.dtype])

def _convert_srgb_hwc(image: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    image += coefs[1]
    image *= coefs[0]
    np.round(image, out=image)
    np.clip(image, 0.0, 255.0, out=image)
    return image.astype("uint8")

def from_srgb_hwc(image: ArrayLike, coefs: Coefs, *, mode: str | None = None, copy: bool = False) -> Image.Image:
    """Creates a PIL image from sRGB image data in the HWC format, using the specified coefficients for byte conversion.

    :param image: Image data.
    :type image: array_like
    :param coefs: Inverse representation conversion coefficients. Inverse coefficients are expected so that additional
        sets of coefficients don't need to be provided in this module.
    :param mode: PIL image mode, such as ``"RGB"``, ``"RGBA"``, or ``"RGBa"``. If not specified, the mode is inferred
        from the shape of the image data.
    :param copy: If ``False``, the default, ``image`` is modified in-place prior to the datatype conversion. Otherwise,
        a copy is first made, which has a performance impact. In some cases, a copy must be made.

    Pixel values outside the range ``[0, 255]`` will be clipped and fractional values will be appropriately rounded.
    NumPy will produce a warning if any ``NaN`` values are encountered and substitute ``0``.

    .. caution::
        Specify ``copy = True`` if you intend to use ``image`` in subsequent operations.
    """

    image = np.array(image, copy=copy, order='C')
    return Image.fromarray(_convert_srgb_hwc(image, coefs[image.dtype]), mode=mode)

def _convert_srgb_to_oklab_frac_hwc(
    image: np.ndarray,
    linear_srgb_to_lms: np.ndarray,
    lms_to_oklab: np.ndarray
) -> None:
    view = image[..., :, :, :3]

    mask = view > 0.04045
    np.multiply(view, 1 / 12.92, where=~mask, out=view)
    np.add(view, 0.055, where=mask, out=view)
    np.multiply(view, 1 / 1.055, where=mask, out=view)
    np.power(view, 2.4, where=mask, out=view)

    np.dot(view, linear_srgb_to_lms, out=view)
    np.cbrt(view, out=view)
    np.dot(view, lms_to_oklab, out=view)

def convert_srgb_to_oklab_frac_hwc(image: np.ndarray) -> None:
    """Converts an image data array in fractional ``[0.0, 1.0]`` HWC format from sRGB to
    `Oklab <https://bottosson.github.io/posts/oklab/>`_ in-place.

    :param image: Image data to convert. Any number of leading batch dimensions may be present. The alpha channel, if
        present, is not modified.

    .. seealso:
        :func:`write_hwc`
    """

    _convert_srgb_to_oklab_frac_hwc(
        image,
        _LINEAR_SRGB_TO_LMS[image.dtype],
        _LMS_TO_OKLAB[image.dtype]
    )

def convert_srgb_to_oklab_frac_chw(image: np.ndarray) -> None:
    """Converts an image data array in fractional ``[0.0, 1.0]`` CHW format from sRGB to
    `Oklab <https://bottosson.github.io/posts/oklab/>`_ in-place.

    :param image: Image data to convert. Any number of leading batch dimensions may be present. The alpha channel, if
        present, is not modified.

    .. seealso:
        :func:`write_chw`
    """

    _convert_srgb_to_oklab_frac_hwc(
        np.moveaxis(image, -3, -1),
        _LINEAR_SRGB_TO_LMS[image.dtype],
        _LMS_TO_OKLAB[image.dtype]
    )

def _convert_oklab_to_srgb_frac_hwc(
    image: np.ndarray,
    oklab_to_lms: np.ndarray,
    lms_to_linear_srgb: np.ndarray
) -> None:
    view = image[..., :, :, :3]

    np.dot(view, oklab_to_lms, out=view)
    np.power(view, 3, out=view)
    np.dot(view, lms_to_linear_srgb, out=view)

    mask = view > 0.0031308
    np.multiply(view, 12.92, where=~mask, out=view)
    np.power(view, 1 / 2.4, where=mask, out=view)
    np.multiply(view, 1.055, where=mask, out=view)
    np.subtract(view, 0.055, where=mask, out=view)

def convert_oklab_to_srgb_frac_hwc(image: np.ndarray) -> None:
    """Converts an image data array in fractional ``[0.0, 1.0]`` HWC format from
    `Oklab <https://bottosson.github.io/posts/oklab/>`_ to sRGB in-place.

    :param image: Image data to convert. Any number of leading batch dimensions may be present. The alpha channel, if
        present, is not modified.
    """

    _convert_oklab_to_srgb_frac_hwc(
        image,
        _OKLAB_TO_LMS[image.dtype],
        _LMS_TO_LINEAR_SRGB[image.dtype]
    )

def convert_oklab_to_srgb_frac_chw(image: np.ndarray) -> None:
    """Converts an image data array in fractional ``[0.0, 1.0]`` CHW format from
    `Oklab <https://bottosson.github.io/posts/oklab/>`_ to sRGB in-place.

    :param image: Image data to convert. Any number of leading batch dimensions may be present. The alpha channel, if
        present, is not modified.
    """

    _convert_oklab_to_srgb_frac_hwc(
        np.moveaxis(image, -3, -1),
        _OKLAB_TO_LMS[image.dtype],
        _LMS_TO_LINEAR_SRGB[image.dtype]
    )

def convert_frac_to_balanced_hwc(image: np.ndarray, scale: float = 1.0) -> None:
    """Converts an image data array in fractional ``[0.0, 1.0]`` HWC format to a balanced ``[-scale, scale]``
    representation in-place.

    :param image: Image data to convert. Any number of leading batch dimensions may be present. The alpha channel, if
        present, is not modified.
    :param scale: Scale of the balanced representation. The default is ``1.0``.

    .. seealso:
        :func:`write_hwc`
    """

    view = image[..., :, :, :3]
    view -= 0.5
    view *= scale * 2.0

def convert_frac_to_balanced_chw(image: np.ndarray, scale: float = 1.0) -> None:
    """Converts an image data array in fractional ``[0.0, 1.0]`` CHW format to a balanced ``[-scale, scale]``
    representation in-place.

    :param image: Image data to convert. Any number of leading batch dimensions may be present. The alpha channel, if
        present, is not modified.
    :param scale: Scale of the balanced representation. The default is ``1.0``.

    .. seealso:
        :func:`write_chw`
    """

    view = image[..., :3, :, :]
    view -= 0.5
    view *= scale * 2.0

def convert_balanced_to_frac_hwc(image: np.ndarray, scale: float = 1.0) -> None:
    """Converts an image data array in balanced ``[-scale, scale]`` HWC format to a fractional ``[0.0, 1.0]``
    representation in-place.

    :param image: Image data to convert. Any number of leading batch dimensions may be present. The alpha channel, if
        present, is not modified.
    :param scale: Scale of the balanced representation. The default is ``1.0``.

    .. seealso:
        :func:`write_hwc`
    """

    view = image[..., :, :, :3]
    view /= scale * 2.0
    view += 0.5

def convert_balanced_to_frac_chw(image: np.ndarray, scale: float = 1.0) -> None:
    """Converts an image data array in balanced ``[-scale, scale]`` HWC format to a fractional ``[0.0, 1.0]``
    representation in-place.

    :param image: Image data to convert. Any number of leading batch dimensions may be present. The alpha channel, if
        present, is not modified.
    :param scale: Scale of the balanced representation. The default is ``1.0``.

    .. seealso:
        :func:`write_chw`
    """

    view = image[..., :3, :, :]
    view /= scale * 2.0
    view += 0.5
