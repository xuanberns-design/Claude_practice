"""Create a verified code-only ZIP, excluding checkpoints and historical results."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


SOURCES = {"src": {".py"}, "configs": {".yaml"}, "scripts": {".py"},
           "tests": {".py"}, "docs": {".md"}}
ROOT_FILES = ("README_中文.md", "requirements.txt", ".gitignore")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(project: Path, output: Path):
    project, output = project.resolve(), output.resolve()
    if output.is_relative_to(project):
        raise ValueError("ZIP 输出必须位于项目目录之外")
    entries = []
    for filename in ROOT_FILES:
        path = project / filename
        if path.is_file():
            entries.append(path)
    for folder, suffixes in SOURCES.items():
        entries.extend(path for path in (project / folder).rglob("*")
                       if path.is_file() and path.suffix.lower() in suffixes
                       and "__pycache__" not in path.parts)
    if not any(path.name == "cli.py" for path in entries):
        raise FileNotFoundError("缺少 src/cli.py")
    entries.sort(key=lambda path: path.relative_to(project).as_posix())
    manifest = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "description": "32 views 分割改进代码包；不含患者数据、旧结果或模型权重",
                "files": [{"path": path.relative_to(project).as_posix(),
                           "size": path.stat().st_size, "sha256": sha256(path)} for path in entries]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for path in entries:
            archive.write(path, "ircadb_multitask_v4/" + path.relative_to(project).as_posix())
        archive.writestr("ircadb_multitask_v4/CODE_PACKAGE_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        damaged = archive.testzip()
        if damaged:
            raise IOError(f"ZIP CRC 校验失败: {damaged}")
        contents = set(archive.namelist())
        if any(name.lower().endswith((".pt", ".pth", ".ckpt", ".png", ".npy")) for name in contents):
            raise ValueError("代码包意外包含权重、图像或缓存")
    temporary.replace(output)
    return {"path": str(output), "bytes": output.stat().st_size,
            "sha256": sha256(output), "files": len(entries) + 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
