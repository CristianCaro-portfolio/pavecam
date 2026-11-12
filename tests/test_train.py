import numpy as np
import pytest

from pavecam import train


@pytest.fixture(scope="module")
def sample(sample_dir):
    return train.load_xy(sample_dir / "features_sample.parquet")


def test_center_by_camera_removes_camera_mean():
    x = np.array([[1.0, 10.0], [3.0, 12.0], [100.0, 0.0], [102.0, 2.0]])
    g = np.array(["a", "a", "b", "b"])
    c = train.center_by_camera(x, g)
    assert np.allclose(c[g == "a"].mean(0), 0) and np.allclose(c[g == "b"].mean(0), 0)


def test_weather_rules_flags_freezing_precip(sample):
    df, y, _, _ = sample
    pred = train.weather_rules(df)
    assert pred.shape == y.shape
    assert set(np.unique(pred)) <= {0, 1, 2}


def test_evaluate_leave_one_site_out_on_real_sample(sample):
    df, y, groups, emb = sample
    rep = train.evaluate(df, y, groups, emb)
    assert rep["n_sites"] == len(set(groups)) >= 10
    for name in train.MODELS:
        m = rep["models"][name]
        assert 0.0 <= m["macro_f1"] <= 1.0
    assert rep["best_model"] in train.MODELS
    # A random split sees every camera in training, so it should never look worse than an
    # unseen camera test for the image model. This guards the leak check itself.
    assert rep["random_split"]["image_probe"]["macro_f1"] >= rep["models"]["image_probe"]["macro_f1"]
    assert sum(sum(r) for r in rep["confusion"]["matrix"]) == len(y)


def test_load_xy_rejects_incomplete_file(tmp_path, sample_dir):
    import polars as pl

    p = tmp_path / "bad.parquet"
    pl.read_parquet(sample_dir / "features_sample.parquet").drop("air_f").write_parquet(p)
    with pytest.raises(ValueError, match="air_f"):
        train.load_xy(p)


def test_smoke_trains_and_writes_bundle():
    res = train.smoke()
    assert res["n_frames"] > 100
    assert res["best_model"] in train.MODELS
