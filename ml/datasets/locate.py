"""Find, unpack and (optionally) download the datasets so users don't have to arrange folders by hand.

RSNA is recognised by `stage_2_train_labels.csv` next to a `stage_2_train_images/` folder; NIH by
`Data_Entry_2017*.csv`. Search order: the configured root (and sub-folders), then the user's Downloads,
Desktop and Documents folders (a few levels deep). A dataset found elsewhere is used IN PLACE (nothing is
copied) and its location is saved to the project's `.env` so training, evaluation and the API agree.
"""
from __future__ import annotations

import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from ml.utils.config import PROJECT_ROOT

RSNA_MARKER = "stage_2_train_labels.csv"
RSNA_IMAGES = "stage_2_train_images"
RSNA_COMPETITION = "rsna-pneumonia-detection-challenge"
NIH_MARKERS = ("Data_Entry_2017_v2020.csv", "Data_Entry_2017.csv", "sample_labels.csv")
NIH_SAMPLE_DATASET = "nih-chest-xrays/sample"   # official 5 % random sample of NIH ChestX-ray14 on Kaggle
RSNA_NIH_MAPPING_URL = ("https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/"
                        "pneumonia-challenge-dataset-mappings_2018.json")
SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "__pycache__", "site-packages", "AppData", "$RECYCLE.BIN"}


# ------------------------------------------------------------------ generic search
def _walk(base: Path, max_depth: int) -> Iterable[Path]:
    """Breadth-first directories under base up to max_depth (errors ignored, big irrelevant dirs skipped)."""
    frontier: List[Path] = [base]
    for _ in range(max_depth + 1):
        nxt: List[Path] = []
        for d in frontier:
            yield d
            try:
                nxt.extend(c for c in d.iterdir() if c.is_dir() and c.name not in SKIP_DIRS
                           and not c.name.startswith(("stage_2_", "images_")))
            except (PermissionError, OSError):
                continue
        frontier = nxt


def default_search_locations() -> List[Path]:
    home = Path.home()
    return [p for p in (home / "Downloads", home / "Desktop", home / "Documents") if p.exists()]


def _find(predicate: Callable[[Path], bool], roots: Iterable[Path], max_depth: int) -> Optional[Path]:
    for root in roots:
        if root.exists():
            for d in _walk(root, max_depth):
                if predicate(d):
                    return d
    return None


def is_rsna_root(d: Path) -> bool:
    return (d / RSNA_MARKER).is_file() and (d / RSNA_IMAGES).is_dir()


def is_nih_root(d: Path) -> bool:
    return any((d / m).is_file() for m in NIH_MARKERS)


# ------------------------------------------------------------------ archives
def extract_rsna_zips(folder: Path, log: Callable[[str], None] = print, delete_after: bool = False) -> bool:
    """Extract any Kaggle RSNA zip found in `folder` (non-recursive). Returns True if something was extracted."""
    done = False
    for z in sorted(folder.glob("*.zip")) if folder.exists() else []:
        try:
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
                if not any(n.endswith(RSNA_MARKER) for n in names):
                    continue
                log(f"Extracting {z.name} ({len(names):,} files) ...")
                zf.extractall(folder)
                done = True
            if delete_after:
                z.unlink()
                log(f"Deleted {z.name} to save disk space")
        except zipfile.BadZipFile:
            log(f"WARNING: {z.name} is not a valid zip (incomplete download?) - skipped")
    return done


def extract_nih_tars(folder: Path, log: Callable[[str], None] = print) -> int:
    """Extract NIH images_*.tar.gz archives whose contents are not yet present. Returns number extracted."""
    n = 0
    for t in sorted(folder.glob("images_*.tar.gz")) if folder.exists() else []:
        marker = folder / f".{t.name}.extracted"
        if marker.exists():
            continue
        log(f"Extracting {t.name} ...")
        with tarfile.open(t) as tf:
            try:
                tf.extractall(folder, filter="data")      # Python >= 3.10.12 / 3.11.4 safe extraction
            except TypeError:
                tf.extractall(folder)                      # older Python
        marker.write_text("ok")
        n += 1
    return n


# ------------------------------------------------------------------ downloads
def free_gb(path: Path) -> float:
    import shutil
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def check_free_space(path: Path, need_gb: float, what: str) -> None:
    have = free_gb(path)
    if have < need_gb:
        raise RuntimeError(f"Not enough disk space for {what}: about {need_gb:.0f} GB needed, {have:.1f} GB free on the "
                           f"drive holding {path}. Free up space or set RFRR_RSNA_ROOT / RFRR_NIH_ROOT in .env to a "
                           f"folder on another drive.")


KAGGLE_CREDS_MSG = ("Kaggle login needed. Run:  kaggle auth login   (opens a browser sign-in), then run this command "
                    "again. Alternative: create a token at https://www.kaggle.com/settings/api and set the "
                    "KAGGLE_API_TOKEN environment variable or save it to ~/.kaggle/access_token.")


def _try_kaggle_api():
    """Return an authenticated KaggleApi or raise. Kaggle's own multi-line help text is suppressed."""
    import contextlib
    import io
    import sys as _sys
    buf = io.StringIO()
    # Kaggle calls the site `exit()` when credentials are missing, which CLOSES sys.stdin before raising
    # SystemExit; give it a throw-away stdin so the real terminal input stays usable (for the sign-in prompt).
    real_stdin, _sys.stdin = _sys.stdin, io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            for name in [m for m in _sys.modules if m == "kaggle" or m.startswith("kaggle.")]:
                del _sys.modules[name]  # a failed earlier import may have left a half-initialised module
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
        return api
    except ImportError as exc:
        raise RuntimeError(f"The 'kaggle' package could not be imported ({exc}). Run:  pip install kaggle six") from None
    except (Exception, SystemExit):  # kaggle raises or exits in various ways without credentials
        raise RuntimeError(KAGGLE_CREDS_MSG) from None
    finally:
        _sys.stdin = real_stdin


def kaggle_login_interactive(log: Callable[[str], None] = print) -> bool:
    """Run `kaggle auth login` in this terminal (browser sign-in). Returns True if it exited successfully."""
    import subprocess
    import sys as _sys
    log("Starting Kaggle sign-in: open the link below in your browser, sign in, then paste the verification code here.")
    # Same entry point as the `kaggle` console command. (`python -m kaggle` would first run kaggle's own
    # credential check at import time and fail before the login sub-command starts.)
    code = "import sys; sys.argv = ['kaggle', 'auth', 'login']; from kaggle.cli import main; sys.exit(main())"
    try:
        return subprocess.call([_sys.executable, "-c", code]) == 0
    except OSError as exc:
        log(f"Could not start the Kaggle sign-in ({exc}).")
        return False


def kaggle_api(interactive: bool = False, log: Callable[[str], None] = print):
    """Authenticated Kaggle API client. With interactive=True (and a real terminal) offers to sign in first."""
    import sys as _sys
    try:
        return _try_kaggle_api()
    except RuntimeError as exc:
        tty = bool(_sys.stdin) and not _sys.stdin.closed and _sys.stdin.isatty()
        if "could not be imported" in str(exc) or not (interactive and tty):
            raise
        log("You are not signed in to Kaggle yet.")
        answer = input("Sign in now with  kaggle auth login ? [Y/n] ").strip().lower()
        if answer not in ("", "y", "yes"):
            raise
        if not kaggle_login_interactive(log):
            raise RuntimeError("Kaggle sign-in did not complete. " + KAGGLE_CREDS_MSG) from None
        return _try_kaggle_api()


def _kaggle_error(exc: BaseException, rules_url: Optional[str]) -> RuntimeError:
    msg = str(exc)
    if ("403" in msg or "Forbidden" in msg) and rules_url:
        return RuntimeError("Kaggle refused the download (403 Forbidden). Usually the competition rules have not been "
                            f"accepted yet: open {rules_url} while logged in, click 'I Understand and Accept', then run "
                            "this again. If you already did, check that your token belongs to the same Kaggle account "
                            "and that your network allows kaggle.com.")
    if "401" in msg or "Unauthorized" in msg:
        return RuntimeError("Kaggle rejected the credentials (401). Run  kaggle auth login  again.")
    return RuntimeError(f"Kaggle download failed: {msg}")


RSNA_NEED_GB = 9.0        # zip + extracted DICOMs (zip deleted afterwards) + room for the 384-px cache
NIH_SAMPLE_NEED_GB = 5.0  # zip + extracted sample images


def download_rsna(dest: Path, log: Callable[[str], None] = print, keep_zip: bool = False) -> None:
    """Download + extract the RSNA competition files with the Kaggle API (needs login and accepted rules)."""
    dest.mkdir(parents=True, exist_ok=True)
    check_free_space(dest, RSNA_NEED_GB, "the RSNA dataset")
    api = kaggle_api()
    log(f"Downloading '{RSNA_COMPETITION}' from Kaggle into {dest} (several GB, this takes a while) ...")
    try:
        api.competition_download_files(RSNA_COMPETITION, path=str(dest), quiet=False)
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        raise _kaggle_error(exc, "https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules") from exc
    extract_rsna_zips(dest, log, delete_after=not keep_zip)


def download_nih_sample(dest: Path, log: Callable[[str], None] = print) -> None:
    """Download the official 5 % random sample of NIH ChestX-ray14 (Kaggle dataset nih-chest-xrays/sample)."""
    dest.mkdir(parents=True, exist_ok=True)
    check_free_space(dest, NIH_SAMPLE_NEED_GB, "the NIH ChestX-ray14 sample")
    api = kaggle_api()
    log(f"Downloading Kaggle dataset '{NIH_SAMPLE_DATASET}' (NIH random sample) into {dest} ...")
    try:
        api.dataset_download_files(NIH_SAMPLE_DATASET, path=str(dest), unzip=True, quiet=False)
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        raise _kaggle_error(exc, None) from exc
    for z in dest.glob("*.zip"):  # some kaggle versions leave the archive behind
        z.unlink()


def download_rsna_nih_mapping(dest_file: Path, log: Callable[[str], None] = print) -> bool:
    """Fetch RSNA's public RSNA->NIH image mapping (used to exclude overlap from the NIH external test)."""
    if dest_file.exists():
        return True
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        log(f"Downloading RSNA->NIH mapping file to {dest_file} ...")
        tmp = dest_file.with_suffix(".part")
        with urllib.request.urlopen(RSNA_NIH_MAPPING_URL, timeout=60) as r, open(tmp, "wb") as fh:
            fh.write(r.read())
        tmp.replace(dest_file)
        return True
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: could not download the mapping file ({exc}). Download it manually from\n  {RSNA_NIH_MAPPING_URL}")
        return False


# ------------------------------------------------------------------ .env persistence
def save_env_var(key: str, value: str, env_path: Path = PROJECT_ROOT / ".env") -> None:
    """Create/update KEY=value in the project .env (other lines untouched) and set it for this process."""
    value = Path(value).as_posix()
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    out, replaced = [], False
    for line in lines:
        if line.split("=", 1)[0].strip() == key:
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.environ[key] = value


def _project_relative(p: Path) -> str:
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.resolve().as_posix()


# ------------------------------------------------------------------ high-level helpers
def ensure_rsna(configured_root: Path, download: bool = False, search: bool = True,
                log: Callable[[str], None] = print) -> Optional[Path]:
    """Return a folder containing the RSNA files, extracting / downloading / locating it as needed."""
    configured_root = Path(configured_root)
    if is_rsna_root(configured_root):
        return configured_root
    # 1. inside the configured folder (nested sub-folder, or still zipped)
    found = _find(is_rsna_root, [configured_root], max_depth=3)
    if not found and extract_rsna_zips(configured_root, log):
        found = _find(is_rsna_root, [configured_root], max_depth=3)
    # 2. elsewhere on this computer (Downloads, Desktop, Documents)
    if not found and search:
        log("RSNA not in the configured folder - searching Downloads, Desktop and Documents ...")
        found = _find(is_rsna_root, default_search_locations(), max_depth=3)
        if not found:  # a Kaggle zip lying in Downloads?
            for loc in default_search_locations():
                for z in loc.glob("rsna-pneumonia-detection-challenge*.zip"):
                    log(f"Found {z} - extracting into {configured_root}")
                    configured_root.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(configured_root)
                    found = _find(is_rsna_root, [configured_root], max_depth=3)
                    break
                if found:
                    break
    # 3. download
    if not found and download:
        download_rsna(configured_root, log)
        found = _find(is_rsna_root, [configured_root], max_depth=3)
    if found and found.resolve() != configured_root.resolve():
        save_env_var("RFRR_RSNA_ROOT", _project_relative(found))
        log(f"Using RSNA dataset at: {found}\n  (saved as RFRR_RSNA_ROOT in .env - training, evaluation and the web app "
            f"will use it)")
    return found


def ensure_nih(configured_root: Path, search: bool = True, fetch_mapping: bool = True,
               download_sample: bool = False, log: Callable[[str], None] = print) -> Optional[Path]:
    configured_root = Path(configured_root)
    found = configured_root if is_nih_root(configured_root) else _find(is_nih_root, [configured_root], max_depth=3)
    if not found and search:
        log("NIH not in the configured folder - searching Downloads, Desktop and Documents ...")
        found = _find(is_nih_root, default_search_locations(), max_depth=3)
    if not found and download_sample:
        download_nih_sample(configured_root, log)
        found = _find(is_nih_root, [configured_root], max_depth=3)
    if found:
        extract_nih_tars(found, log)
        if found.resolve() != configured_root.resolve():
            save_env_var("RFRR_NIH_ROOT", _project_relative(found))
            log(f"Using NIH dataset at: {found}\n  (saved as RFRR_NIH_ROOT in .env)")
        if fetch_mapping:
            download_rsna_nih_mapping(found / "pneumonia-challenge-dataset-mappings_2018.json", log)
    return found
