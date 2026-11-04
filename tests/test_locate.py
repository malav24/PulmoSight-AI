"""Dataset discovery / extraction / .env persistence (synthetic folders only; no network)."""
import os
import zipfile
from pathlib import Path

import pytest

from ml.datasets import locate


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "Downloads").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    return home


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    orig = locate.save_env_var
    monkeypatch.setattr(locate, "save_env_var", lambda k, v, env_path=p: orig(k, v, env_path))
    for k in ("RFRR_RSNA_ROOT", "RFRR_NIH_ROOT"):
        monkeypatch.delenv(k, raising=False)
    yield p
    for k in ("RFRR_RSNA_ROOT", "RFRR_NIH_ROOT"):
        os.environ.pop(k, None)


def make_rsna(folder: Path):
    (folder / "stage_2_train_images").mkdir(parents=True)
    (folder / "stage_2_train_labels.csv").write_text("patientId,x,y,width,height,Target\n")


def test_configured_root_used_directly(tmp_path, env_file):
    root = tmp_path / "data" / "rsna"
    make_rsna(root)
    assert locate.ensure_rsna(root, search=False) == root
    assert not env_file.exists()


def test_nested_subfolder_found(tmp_path, env_file):
    root = tmp_path / "data" / "rsna"
    make_rsna(root / "rsna-pneumonia-detection-challenge")
    found = locate.ensure_rsna(root, search=False)
    assert found == root / "rsna-pneumonia-detection-challenge"
    assert "RFRR_RSNA_ROOT=" in env_file.read_text()


def test_kaggle_zip_in_root_is_extracted(tmp_path, env_file):
    root = tmp_path / "data" / "rsna"
    root.mkdir(parents=True)
    with zipfile.ZipFile(root / "rsna-pneumonia-detection-challenge.zip", "w") as z:
        z.writestr("stage_2_train_labels.csv", "patientId,x,y,width,height,Target\n")
        z.writestr("stage_2_train_images/abc.dcm", b"")
    assert locate.ensure_rsna(root, search=False) == root
    assert (root / "stage_2_train_labels.csv").exists()


def test_found_in_downloads_archive_folder_and_saved_to_env(tmp_path, fake_home, env_file):
    elsewhere = fake_home / "Downloads" / "archive (7)"
    make_rsna(elsewhere)
    root = tmp_path / "data" / "rsna"
    root.mkdir(parents=True)
    found = locate.ensure_rsna(root, search=True)
    assert found == elsewhere
    assert os.environ["RFRR_RSNA_ROOT"] == elsewhere.as_posix()
    assert env_file.read_text().strip() == f"RFRR_RSNA_ROOT={elsewhere.as_posix()}"
    # updating keeps other lines and replaces the old value
    env_file.write_text("RFRR_DEVICE=cuda\nRFRR_RSNA_ROOT=old\n")
    locate.save_env_var("RFRR_RSNA_ROOT", "new/place", env_file)
    assert env_file.read_text() == "RFRR_DEVICE=cuda\nRFRR_RSNA_ROOT=new/place\n"


def test_kaggle_zip_in_downloads_is_extracted_into_project(tmp_path, fake_home, env_file):
    with zipfile.ZipFile(fake_home / "Downloads" / "rsna-pneumonia-detection-challenge.zip", "w") as z:
        z.writestr("stage_2_train_labels.csv", "x\n")
        z.writestr("stage_2_train_images/a.dcm", b"")
    root = tmp_path / "data" / "rsna"
    assert locate.ensure_rsna(root, search=True) == root


def test_not_found_returns_none_and_download_errors_are_clear(tmp_path, fake_home, env_file, monkeypatch):
    root = tmp_path / "data" / "rsna"
    root.mkdir(parents=True)
    assert locate.ensure_rsna(root, search=True) is None

    import builtins
    real_import = builtins.__import__

    def no_kaggle(name, *a, **k):
        if name.startswith("kaggle"):
            raise ImportError("no kaggle")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_kaggle)
    with pytest.raises(RuntimeError, match="pip install kaggle"):
        locate.ensure_rsna(root, download=True, search=False)


def test_nih_located_and_tars_extracted(tmp_path, fake_home, env_file, monkeypatch):
    import io
    import tarfile
    monkeypatch.setattr(locate, "download_rsna_nih_mapping", lambda *a, **k: False)  # no network in tests
    nih = fake_home / "Downloads" / "ChestXray-NIHCC"
    nih.mkdir()
    (nih / "Data_Entry_2017_v2020.csv").write_text("Image Index\n")
    data = b"png"
    with tarfile.open(nih / "images_001.tar.gz", "w:gz") as tf:
        info = tarfile.TarInfo("images/00000001_000.png")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    found = locate.ensure_nih(tmp_path / "data" / "nih")
    assert found == nih and (nih / "images" / "00000001_000.png").exists()
    assert locate.extract_nih_tars(nih) == 0  # not extracted twice
    assert os.environ["RFRR_NIH_ROOT"] == nih.as_posix()


# ----------------------------------------------------------------------------- downloads (Kaggle mocked)
class FakeKaggle:
    """Stands in for the Kaggle API; writes files shaped like the real downloads."""

    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []

    def competition_download_files(self, competition, path, quiet=False):
        self.calls.append(("competition", competition))
        if self.fail:
            raise Exception(self.fail)
        with zipfile.ZipFile(Path(path) / f"{competition}.zip", "w") as z:
            z.writestr("stage_2_train_labels.csv", "patientId,x,y,width,height,Target\n")
            z.writestr("stage_2_detailed_class_info.csv", "patientId,class\n")
            z.writestr("stage_2_train_images/a.dcm", b"")

    def dataset_download_files(self, dataset, path, unzip=True, quiet=False):
        self.calls.append(("dataset", dataset))
        d = Path(path)
        (d / "sample" / "images").mkdir(parents=True, exist_ok=True)
        (d / "sample" / "images" / "00000013_005.png").write_bytes(b"png")
        (d / "sample_labels.csv").write_text(
            "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,"
            "OriginalImageWidth,OriginalImageHeight,OriginalImagePixelSpacing_x,OriginalImagePixelSpacing_y\n"
            "00000013_005.png,Pneumonia|Effusion,5,13,060Y,M,AP,3056,2544,0.139,0.139\n")


def test_download_rsna_extracts_and_deletes_zip(tmp_path, env_file, fake_home, monkeypatch):
    fake = FakeKaggle()
    monkeypatch.setattr(locate, "kaggle_api", lambda: fake)
    monkeypatch.setattr(locate, "free_gb", lambda p: 100.0)
    root = tmp_path / "data" / "rsna"
    assert locate.ensure_rsna(root, download=True, search=True) == root
    assert fake.calls == [("competition", "rsna-pneumonia-detection-challenge")]
    assert (root / "stage_2_train_labels.csv").exists() and not list(root.glob("*.zip"))
    # second run: already present -> no new download
    locate.ensure_rsna(root, download=True)
    assert len(fake.calls) == 1


def test_download_refused_without_disk_space_or_rules(tmp_path, env_file, fake_home, monkeypatch):
    monkeypatch.setattr(locate, "kaggle_api", lambda: FakeKaggle(fail="403 Client Error: Forbidden"))
    monkeypatch.setattr(locate, "free_gb", lambda p: 2.0)
    with pytest.raises(RuntimeError, match="Not enough disk space"):
        locate.download_rsna(tmp_path / "r")
    monkeypatch.setattr(locate, "free_gb", lambda p: 100.0)
    with pytest.raises(RuntimeError, match="I Understand and Accept"):
        locate.download_rsna(tmp_path / "r")


def test_nih_sample_download_and_parsing(tmp_path, env_file, fake_home, monkeypatch):
    import warnings
    from ml.datasets.nih import find_entries_csv, index_nih_images, is_nih_sample, parse_nih_entries, select_nih_external_set
    fake = FakeKaggle()
    monkeypatch.setattr(locate, "kaggle_api", lambda: fake)
    monkeypatch.setattr(locate, "free_gb", lambda p: 100.0)
    monkeypatch.setattr(locate, "download_rsna_nih_mapping", lambda *a, **k: False)
    root = tmp_path / "data" / "nih"
    assert locate.ensure_nih(root, search=True, download_sample=True) == root
    assert fake.calls == [("dataset", "nih-chest-xrays/sample")]
    entries = find_entries_csv(root, "Data_Entry_2017_v2020.csv")
    assert entries.name == "sample_labels.csv" and is_nih_sample(entries)
    df = parse_nih_entries(entries)
    assert df.pneumonia.tolist() == [1] and df.orig_w.tolist() == [3056]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = select_nih_external_set(df, {"subset": "official_test", "exclude_rsna_overlap": False,
                                           "view_positions": ["PA", "AP"]}, root)
    assert len(sel) == 1 and "test_list.txt not found" in sel.attrs["selection_notes"][0]
    assert "00000013_005.png" in index_nih_images(root)
