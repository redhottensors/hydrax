import numpy as np

from hydrax.image import *

def test_oklab_round_trip():
    img = open_srgb("test-image.webp")

    img_data = np.empty((img.height, img.width, 3), dtype="float32")
    write_hwc(img, img_data, Coefs.FRACTIONAL)

    orig_data = img_data.copy()

    convert_srgb_to_oklab_frac_hwc(img_data)
    convert_oklab_to_srgb_frac_hwc(img_data)

    assert np.allclose(img_data, orig_data, atol=0.25/255.0)

    new_img = from_srgb_hwc(img_data, Coefs.FRACTIONAL)
    new_img_data = np.asarray(new_img)

    assert np.allclose(img_data, new_img_data, atol=0)

if __name__ == "__main__":
    test_oklab_round_trip()
    print("All tests passed.")

