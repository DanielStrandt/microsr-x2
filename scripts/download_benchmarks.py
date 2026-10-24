#!/usr/bin/env python3
"""Download and prepare the four classic SR sets linked by EDSR.

Dataset image files stay in the ignored local data/ folder. Manga109 is gated
and intentionally skipped; obtain it from its official site after access is
granted, then place authorized HR pages under data/benchmarks/Manga109/HR/.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("https://cv.snu.ac.kr/research/EDSR/benchmark.tar", "EDSR reference archive"),
    ("https://figshare.com/ndownloader/articles/21586188/versions/1", "Figshare benchmark archive"),
)
EXPECTED = {"Set5": 5, "Set14": 14, "BSD100": 100, "Urban100": 100}
ALIASES = {"Set5": {"Set5"}, "Set14": {"Set14"}, "BSD100": {"BSD100", "B100"},
           "Urban100": {"Urban100"}}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "MicroSR-X2-benchmark-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    """Extract regular files and directories only; reject path traversal/links."""
    root = destination.resolve()
    for member in tar.getmembers():
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Unsafe path in dataset archive: {member.name}")
        if not (member.isdir() or member.isfile()):
            raise ValueError(f"Unsupported special file in dataset archive: {member.name}")
        output = (destination / Path(*name.parts)).resolve()
        if root not in output.parents and output != root:
            raise ValueError(f"Archive path escapes extraction folder: {member.name}")
        if member.isfile():
            output.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Unable to read archive member: {member.name}")
            with source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
        else:
            output.mkdir(parents=True, exist_ok=True)


def safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        name = PurePosixPath(member.filename)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
        mode = member.external_attr >> 16
        if mode and not (mode & 0o170000 in (0, 0o100000, 0o040000)):
            raise ValueError(f"Unsupported special file in dataset archive: {member.filename}")
        output = (destination / Path(*name.parts)).resolve()
        if root not in output.parents and output != root:
            raise ValueError(f"Archive path escapes extraction folder: {member.filename}")
        if member.is_dir():
            output.mkdir(parents=True, exist_ok=True)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)


def safe_extract_archive(path: Path, destination: Path) -> None:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            safe_extract_zip(archive, destination)
        return
    with tarfile.open(path, "r:*") as archive:
        safe_extract(archive, destination)


def candidate_hr_images(extracted: Path, aliases: set[str], expected: int) -> list[Path] | None:
    candidates: list[list[Path]] = []
    for folder in extracted.rglob("*"):
        if not folder.is_dir() or folder.name.lower() not in {x.lower() for x in aliases}:
            continue
        # Prefer explicit HR folders. Ignore LR and pre-generated x2/x3/x4 trees.
        hr_dirs = [p for p in folder.rglob("*") if p.is_dir() and p.name.lower() == "hr"]
        search_dirs = hr_dirs or [folder]
        for search_dir in search_dirs:
            images = sorted(p for p in search_dir.rglob("*") if p.is_file()
                            and p.suffix.lower() in IMAGE_SUFFIXES
                            and not any(part.lower() in {"lr", "lr_bicubic", "x2", "x3", "x4"}
                                        for part in p.relative_to(folder).parts[:-1]))
            if len(images) == expected:
                candidates.append(images)
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "benchmarks")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "downloads" / "benchmark.tar")
    parser.add_argument("--refresh", action="store_true", help="Download the archive again")
    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)

    available = []
    for name, count in EXPECTED.items():
        folder = args.data_root / name / "HR"
        images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES] if folder.is_dir() else []
        if len(images) == count:
            print(f"READY {name}: {count} HR images already present")
            available.append(name)
    needed = [name for name in EXPECTED if name not in available]
    if needed:
        if args.refresh or not args.cache.is_file():
            temp_download = args.cache.with_suffix(args.cache.suffix + ".part")
            try:
                last_error = None
                for url, label in SOURCES:
                    try:
                        print(f"Downloading {label} from {url}")
                        download(url, temp_download)
                        # Reject HTML/error pages and partial responses before caching.
                        if not (tarfile.is_tarfile(temp_download) or zipfile.is_zipfile(temp_download)):
                            raise ValueError("response is not a readable TAR or ZIP archive")
                        os.replace(temp_download, args.cache)
                        last_error = None
                        break
                    except (OSError, urllib.error.URLError, ValueError, tarfile.TarError) as exc:
                        temp_download.unlink(missing_ok=True)
                        print(f"Unavailable: {label} ({exc})")
                        last_error = exc
                if last_error is not None:
                    print(f"SKIP downloadable sets: no archive source was obtainable")
                    needed = []
            except (OSError, urllib.error.URLError) as exc:
                temp_download.unlink(missing_ok=True)
                print(f"SKIP downloadable sets: archive unavailable ({exc})")
                needed = []
        if needed:
            with tempfile.TemporaryDirectory(prefix="microsr-benchmark-") as temp:
                extract_root = Path(temp) / "extracted"
                extract_root.mkdir()
                try:
                    safe_extract_archive(args.cache, extract_root)
                except (OSError, tarfile.TarError, ValueError) as exc:
                    print(f"SKIP downloadable sets: archive could not be safely read ({exc})")
                    needed = []
                for name in needed:
                    images = candidate_hr_images(extract_root, ALIASES[name], EXPECTED[name])
                    if images is None:
                        print(f"SKIP {name}: archive did not contain exactly {EXPECTED[name]} HR images")
                        continue
                    target = args.data_root / name / "HR"
                    target.mkdir(parents=True, exist_ok=True)
                    for image in images:
                        shutil.copy2(image, target / image.name)
                    print(f"READY {name}: downloaded and verified {len(images)} HR images")
                    available.append(name)

    # Manga109 is gated; never try alternate mirrors or bypass its access flow.
    manga_dir = args.data_root / "Manga109" / "HR"
    manga_images = [p for p in manga_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES] if manga_dir.is_dir() else []
    if len(manga_images) == 109:
        print("READY Manga109: 109 authorized HR images already present")
        available.append("Manga109")
    else:
        print("SKIP Manga109: gated; request access and place the authorized 109 HR images locally")
    print("Available datasets:", ", ".join(available) if available else "none")
    return 0 if available else 2


if __name__ == "__main__":
    raise SystemExit(main())

