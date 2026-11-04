import numpy as np
import pytest

from ml.datasets.preprocessing import (Preprocessor, apply_clahe, boxes_to_mask, compute_resize_geometry,
                                       load_dicom, load_image_any, resize_image, transform_boxes,
                                       window_to_uint8, xywh_to_xyxy)
from ml.utils.synthetic import write_synthetic_dicom


def test_dicom_loading_rescale_and_metadata(tmp_path):
    px = np.arange(64 * 80, dtype=np.uint16).reshape(64, 80) % 4096
    p = tmp_path / "a.dcm"
    write_synthetic_dicom(p, px, view="AP", slope=2.0, intercept=-10.0)
    arr, meta = load_dicom(p)
    assert arr.shape == (64, 80) and arr.dtype == np.float32
    np.testing.assert_allclose(arr, px.astype(np.float32) * 2 - 10)
    assert meta["view_position"] == "AP" and meta["modality"] == "CR" and meta["rows"] == 64


def test_dicom_monochrome1_is_inverted(tmp_path):
    px = np.zeros((32, 32), np.uint16)
    px[:, :16] = 1000
    p = tmp_path / "m1.dcm"
    write_synthetic_dicom(p, px, photometric="MONOCHROME1")
    arr, _ = load_dicom(p)
    assert arr[:, :16].mean() < arr[:, 16:].mean()  # bright stored values become dark


def test_load_image_any_png(tmp_path):
    import cv2
    img = (np.random.default_rng(0).random((40, 50)) * 255).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "x.png"), img)
    arr, meta = load_image_any(tmp_path / "x.png")
    assert arr.shape == (40, 50) and meta["source"] == "raster"


def test_window_and_clahe():
    arr = np.random.default_rng(0).normal(1000, 100, (64, 64)).astype(np.float32)
    u8 = window_to_uint8(arr)
    assert u8.dtype == np.uint8 and u8.min() == 0 and u8.max() == 255
    assert apply_clahe(u8).shape == u8.shape
    assert window_to_uint8(np.ones((8, 8))).max() == 0  # constant image


@pytest.mark.parametrize("mode", ["stretch", "letterbox"])
def test_box_transform_matches_image_geometry(mode):
    h, w = 200, 100
    img = np.zeros((h, w), np.uint8)
    box = np.array([[20, 50, 60, 120]], np.float32)  # x1,y1,x2,y2
    img[50:120, 20:60] = 255
    geom = compute_resize_geometry(h, w, 64, mode)
    out = resize_image(img, geom)
    tb = transform_boxes(box, geom)[0]
    ys, xs = np.where(out > 127)
    # bright region of the resized image coincides with the transformed box (within 1 px)
    assert abs(xs.min() - tb[0]) <= 1.0 and abs(xs.max() + 1 - tb[2]) <= 1.0
    assert abs(ys.min() - tb[1]) <= 1.0 and abs(ys.max() + 1 - tb[3]) <= 1.0


def test_boxes_to_mask_exact_and_union():
    m = boxes_to_mask(np.array([[2, 3, 6, 5], [5, 4, 8, 8]]), 10, 10)
    assert m[3:5, 2:6].all() and m[4:8, 5:8].all()
    assert m.sum() == 4 * 2 + 3 * 4 - 1 * 1  # union with 1 overlapping pixel
    assert boxes_to_mask(np.zeros((0, 4)), 5, 5).sum() == 0


def test_xywh_to_xyxy():
    np.testing.assert_allclose(xywh_to_xyxy([[1, 2, 3, 4]]), [[1, 2, 4, 6]])


def test_preprocessor_tensor_roundtrip():
    pre = Preprocessor({"image_size": 32, "clahe": True})
    img, geom = pre(np.random.default_rng(0).random((100, 80)).astype(np.float32) * 4000)
    assert img.shape == (32, 32) and geom.orig_h == 100
    t = pre.to_tensor(img)
    assert t.shape == (3, 32, 32)
    np.testing.assert_allclose(pre.denormalize(t), img / 255.0, atol=1e-5)
