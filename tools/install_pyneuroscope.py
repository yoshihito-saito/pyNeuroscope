from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


APP_NAME = "pyNeuroscope"
ZIP_NAME = "pyNeuroscope-Windows.zip"


def _bundle_dir() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def _message(title: str, text: str) -> None:
    ctypes.windll.user32.MessageBoxW(None, text, title, 0x40)


def _desktop_dir() -> Path:
    return Path(os.path.join(os.environ["USERPROFILE"], "Desktop"))


def _quote_ps(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _create_shortcut(target: Path, shortcut: Path) -> None:
    script = "\n".join(
        [
            "$shell = New-Object -ComObject WScript.Shell",
            f"$shortcut = $shell.CreateShortcut({_quote_ps(shortcut)})",
            f"$shortcut.TargetPath = {_quote_ps(target)}",
            f"$shortcut.WorkingDirectory = {_quote_ps(target.parent)}",
            f"$shortcut.IconLocation = {_quote_ps(target)}",
            "$shortcut.Save()",
        ]
    )
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def main() -> int:
    quiet = "--quiet" in sys.argv
    zip_path = _bundle_dir() / ZIP_NAME
    if not zip_path.exists():
        if not quiet:
            _message(APP_NAME, f"Installer payload was not found:\n{zip_path}")
        return 1

    install_parent = Path(os.environ["LOCALAPPDATA"]) / "Programs"
    install_root = install_parent / APP_NAME
    exe_path = install_root / "pyNeuroscope.exe"

    install_parent.mkdir(parents=True, exist_ok=True)
    if install_root.exists():
        shutil.rmtree(install_root)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(install_parent)

    if not exe_path.exists():
        if not quiet:
            _message(APP_NAME, f"Installed executable was not found:\n{exe_path}")
        return 1

    shortcut_path = _desktop_dir() / "pyNeuroscope.lnk"
    _create_shortcut(exe_path, shortcut_path)
    if not quiet:
        _message(
            APP_NAME,
            f"pyNeuroscope was installed to:\n{install_root}\n\nA desktop shortcut was created.",
        )
        subprocess.Popen([str(exe_path)], cwd=str(install_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
