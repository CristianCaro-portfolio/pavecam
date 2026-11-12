import numpy as np
from PIL import Image

from pavecam.vision import PHYSICS_FEATURES, HashEncoder, is_usable_frame, physics_features


def test_physics_features_on_real_frames(sample_dir):
    frames = sorted((sample_dir / "frames").glob("*.jpg"))
    assert frames, "sample frames missing"
    for p in frames:
        f = physics_features(Image.open(p))
        assert set(f) == set(PHYSICS_FEATURES)
        assert all(0.0 <= v <= 1.0 for v in f.values())
        assert is_usable_frame(f)[0]


def test_frame_gate_rejects_black_and_flat_frames():
    black = physics_features(Image.new("RGB", (320, 240), (0, 0, 0)))
    grey = physics_features(Image.new("RGB", (320, 240), (128, 128, 128)))
    assert is_usable_frame(black) == (False, "too dark")
    assert is_usable_frame(grey) == (False, "flat frame")


def test_snow_scene_is_whiter_than_bare_scene():
    snow = Image.new("RGB", (320, 240), (235, 238, 240))
    road = Image.new("RGB", (320, 240), (90, 92, 95))
    for im in (snow, road):  # some texture so it is not a flat frame
        a = np.asarray(im).copy()
        a[::7, :, :] = 40
        im.paste(Image.fromarray(a))
    assert physics_features(snow)["white_frac"] > physics_features(road)["white_frac"]


def test_hash_encoder_is_deterministic_and_normalised(sample_dir):
    imgs = [Image.open(p) for p in sorted((sample_dir / "frames").glob("*.jpg"))]
    a, b = HashEncoder().embed_images(imgs), HashEncoder().embed_images(imgs)
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert HashEncoder().zero_shot(a).sum(1) == __import__("pytest").approx(np.ones(len(imgs)))
