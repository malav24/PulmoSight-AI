"""API tests. Data and checkpoints here are SYNTHETIC (random noise) - they exercise the software path only;
predictions produced in these tests have no medical meaning."""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    base = tmp_path_factory.mktemp("api")
    from ml.utils.synthetic import make_synthetic_nih, make_synthetic_rsna
    make_synthetic_rsna(base / "rsna", n=24, size=128)
    make_synthetic_nih(base / "nih", n=8, size=128)
    for k, v in {"RFRR_CHECKPOINT_ROOT": base / "ck", "RFRR_OUTPUT_ROOT": base / "out",
                 "RFRR_RSNA_ROOT": base / "rsna", "RFRR_NIH_ROOT": base / "nih", "RFRR_DEVICE": "cpu"}.items():
        mp.setenv(k, str(v))
    yield base
    mp.undo()


@pytest.fixture(scope="module")
def client(env):
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.services import datasets
    from backend.app.services.model_manager import manager
    manager.invalidate()
    manager.active_key = None
    datasets._CACHE["value"] = None
    return TestClient(app, raise_server_exceptions=False)


def png_bytes(size=300, seed=0) -> bytes:
    import cv2
    img = (np.random.default_rng(seed).random((size, size)) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def train_tiny(env, cfg_name="model_b.yaml"):
    from ml.training.trainer import run_training
    from ml.utils.config import load_config
    cfg = load_config(cfg_name, {
        "data": {"rsna": {"splits_csv": str(env / "splits.csv")}, "preprocessing": {"image_size": 64},
                 "split": {"train": 0.6, "val": 0.2, "test": 0.2}},
        "loader": {"batch_size": 4, "num_workers": 0}, "model": {"pretrained_backbone": False},
        "training": {"epochs": 1, "amp": False, "warmup_epochs": 0}})
    run_training(cfg)


# ------------------------------------------------------------- no trained models
def test_health_and_models_without_checkpoints(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["ml"]["available"] and h["models"]["trained_count"] == 0
    m = client.get("/api/models").json()
    assert m["any_trained"] is False and m["message"] == "Models not trained yet."
    keys = {x["key"] for x in m["models"]}
    assert {"resnet50", "densenet121", "efficientnet_b0", "vit_small", "model_a", "model_b"} <= keys
    for x in m["models"]:
        assert x["trained"] is False and x["checkpoint"]["status"] == "not_trained"
        assert not x["checkpoint"]["path"].startswith("/")  # no absolute server paths


def test_missing_checkpoint_scenario(client):
    r = client.post("/api/analyze", files={"file": ("x.png", png_bytes(), "image/png")})
    assert r.status_code == 409 and r.json()["error"]["code"] == "model_not_trained"
    r = client.post("/api/models/load", json={"model_key": "model_a"})
    assert r.status_code == 409
    r = client.post("/api/models/load", json={"model_key": "nonexistent"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_model"


@pytest.mark.parametrize("name,content,code,status", [
    ("a.txt", b"hello", "unsupported_file_type", 415),
    ("a.png", b"not really a png at all", "content_mismatch", 415),
    ("a.png", b"\x89PNG\r\n\x1a\n" + b"0" * 64, "corrupt_image", 415),
    ("a.dcm", b"\x00" * 200, "invalid_dicom", 415),
    ("a.png", b"", "empty_file", 422),
])
def test_invalid_uploads(client, name, content, code, status):
    r = client.post("/api/analyze", files={"file": (name, content, "application/octet-stream")})
    assert r.status_code == status, r.text
    assert r.json()["error"]["code"] == code


def test_upload_size_and_dimension_limits(client, monkeypatch):
    from backend.app import settings as s
    small = s.Settings(max_upload_mb=0.001)
    monkeypatch.setattr("backend.app.routers.api.get_settings", lambda: small)
    r = client.post("/api/analyze", files={"file": ("x.png", png_bytes(300), "image/png")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"
    tiny_pixels = s.Settings(max_image_pixels=1000)
    monkeypatch.setattr("backend.app.routers.api.get_settings", lambda: tiny_pixels)
    r = client.post("/api/analyze", files={"file": ("x.png", png_bytes(300), "image/png")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "image_too_large"


def test_missing_file_field_is_validation_error(client):
    r = client.post("/api/analyze")
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_datasets_status_reports_real_files(client):
    d = client.get("/api/datasets/status?refresh=true").json()
    assert d["rsna"]["annotation_csv"] and d["rsna"]["image_count"] == 24
    assert d["rsna"]["preparation_status"] == "ready to prepare"
    assert d["nih"]["metadata_csv"] and d["nih"]["image_count"] == 8
    assert d["nih"]["rsna_overlap_mapping"] is False and d["nih"]["external_validity_warning"]
    assert "custom location" in d["rsna"]["directory"]  # absolute temp path hidden


def test_experiments_empty(client):
    r = client.get("/api/experiments/results").json()
    assert r["available"] is False and r["message"] == "No experiment results available."
    r = client.post("/api/experiments/run", json={"kind": "train", "model_key": "model_b"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "confirmation_required"
    r = client.post("/api/experiments/run", json={"kind": "bogus"})
    assert r.status_code == 422


def test_research_report_without_models(client):
    r = client.get("/api/research/report")
    assert r.status_code == 200 and "Models not trained yet" in r.text and "not a medical device" in r.text
    assert client.get("/api/research/report?analysis_id=xyz").status_code == 400
    assert client.get("/api/research/report?analysis_id=" + "0" * 32).status_code == 404


# ------------------------------------------------------------- with a (synthetic) trained checkpoint
def test_trained_checkpoint_flow(client, env):
    train_tiny(env, "model_b.yaml")
    from backend.app.services.model_manager import manager
    manager.invalidate()
    m = {x["key"]: x for x in client.get("/api/models").json()["models"]}
    assert m["model_b"]["trained"] and m["model_b"]["checkpoint"]["checkpoint_id"].startswith("model_b@epoch0#")
    assert "val_auc" in m["model_b"]["checkpoint"]["validation_metrics"]
    assert not m["model_a"]["trained"]

    r = client.post("/api/models/load", json={"model_key": "model_b"})
    assert r.status_code == 200 and r.json()["model"]["active"]

    r = client.post("/api/analyze", files={"file": ("scan.png", png_bytes(300), "image/png")})
    assert r.status_code == 200, r.text
    a = r.json()
    assert len(a["analysis_id"]) == 32 and a["model"]["key"] == "model_b"
    assert 0 <= a["prediction"]["probability"] <= 1
    assert a["prediction"]["outcome"] in ("suspected", "not_detected", "abstained")
    assert "diagnos" not in a["prediction"]["label"].lower()
    assert set(a["images"]) == {"original", "localization", "attention", "gradcam"}
    assert all(v.startswith("data:image/png;base64,") for v in a["images"].values())
    assert a["localization"]["available"] and a["uncertainty"]["n_samples"] > 0
    assert a["input"]["file_name"] == "scan.png" and a["input_reference"].startswith("sha256:")

    rec = client.get(f"/api/analyses/{a['analysis_id']}").json()
    assert "file_name" not in json.dumps(rec)  # original file name not persisted
    recent = client.get("/api/analyses").json()["analyses"]
    assert recent[0]["analysis_id"] == a["analysis_id"]

    rep = client.get(f"/api/research/report?analysis_id={a['analysis_id']}")
    assert rep.status_code == 200
    for needle in (a["analysis_id"], "Checkpoint identifier", "Grad-CAM++", "Limitations", "not a medical device"):
        assert needle in rep.text
    assert client.get(f"/api/research/report?analysis_id={a['analysis_id']}&format=json").json()["analysis_id"]


def test_explain_with_rsna_reference_and_provided_boxes(client, env):
    dcm = next((env / "rsna" / "stage_2_train_images").glob("*.dcm"))
    import pandas as pd
    labels = pd.read_csv(env / "rsna" / "stage_2_train_labels.csv")
    pos = labels[labels.Target == 1].patientId.iloc[0]
    dcm = env / "rsna" / "stage_2_train_images" / f"{pos}.dcm"
    r = client.post("/api/explain", files={"file": (dcm.name, dcm.read_bytes(), "application/dicom")},
                    data={"model_key": "model_b"})
    assert r.status_code == 200, r.text
    e = r.json()
    assert e["reference"]["available"] and e["reference"]["source"] == "rsna_annotations"
    assert {"localization", "gradcam", "attention_alignment"} <= set(e["metrics"])
    assert 0 <= e["metrics"]["localization"]["iou"] <= 1 and "pointing_game_hit" in e["metrics"]["gradcam"]
    assert e["influence"]["gradcam_zones"] and e["limitations"]
    r = client.post("/api/explain", files={"file": ("x.png", png_bytes(300), "image/png")},
                    data={"reference_boxes": json.dumps([[50, 60, 80, 90]])})
    assert r.json()["reference"]["source"] == "provided" and len(r.json()["reference"]["boxes"]) == 1
    r = client.post("/api/explain", files={"file": ("x.png", png_bytes(300), "image/png")},
                    data={"reference_boxes": "[[1,2]]"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_boxes"
    r = client.post("/api/explain", files={"file": ("y.png", png_bytes(300, 3), "image/png")})
    assert r.json()["reference"]["available"] is False and r.json()["metrics"] == {}


def test_experiment_job_runs_in_background_and_results_load(client, env):
    r = client.post("/api/experiments/run", json={"kind": "evaluate", "model_key": "model_b",
                                                  "params": {"explain_n": 2}})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    # while the job runs, inference/health still respond
    assert client.get("/api/health").status_code == 200
    assert client.post("/api/experiments/run", json={"kind": "ablation_summary"}).status_code in (202, 409)
    for _ in range(240):
        j = client.get(f"/api/experiments/jobs/{job_id}").json()
        if j["status"] != "running":
            break
        time.sleep(1)
    assert j["status"] == "succeeded", "\n".join(j["log_tail"])
    res = client.get("/api/experiments/results").json()
    assert res["available"] and "model_b" in res["models"]
    mb = res["models"]["model_b"]
    assert mb["metrics"]["rsna_test"]["classification"]["n"] > 0 and mb["training_log"]
    assert "rsna_test" in mb["curves"]
    files = [f["name"] for f in res["files"]]
    assert "predictions/model_b_rsna_test.csv" in files
    d = client.get("/api/experiments/files/predictions/model_b_rsna_test.csv")
    assert d.status_code == 200 and d.text.startswith("id,")
    assert client.get("/api/experiments/files/predictions/..%2F..%2Fsecret.csv").status_code in (400, 404)
    assert client.get("/api/experiments/files/other/x.csv").status_code == 400
    m = {x["key"]: x for x in client.get("/api/models").json()["models"]}
    assert m["model_b"]["evaluation"]["rsna_test_auc"] is not None or m["model_b"]["evaluation"] is not None
    assert m["model_b"]["calibration"]["temperature"] > 0
