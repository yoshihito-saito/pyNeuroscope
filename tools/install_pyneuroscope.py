from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import textwrap
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


def _error_message(title: str, text: str) -> None:
    ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)


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


def _default_install_parent() -> Path:
    return Path(os.environ["LOCALAPPDATA"]) / "Programs"


def _parse_install_parent(args: list[str]) -> Path | None:
    for index, arg in enumerate(args):
        if arg == "--install-dir" and index + 1 < len(args):
            return Path(args[index + 1]).expanduser()
        if arg.startswith("--install-dir="):
            return Path(arg.split("=", 1)[1]).expanduser()
    return None


def _choose_install_parent(default_parent: Path) -> Path | None:
    default_parent.mkdir(parents=True, exist_ok=True)
    script = textwrap.dedent(
        f"""
        Add-Type -AssemblyName System.Windows.Forms
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
        $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
        $dialog.Description = "Select the folder where pyNeuroscope will be installed"
        $dialog.SelectedPath = {_quote_ps(default_parent)}
        $dialog.ShowNewFolderButton = $true
        if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
            Write-Output $dialog.SelectedPath
            exit 0
        }}
        exit 2
        """
    ).strip()
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode == 2:
        return None
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Could not open the install folder picker.")
    selected = result.stdout.strip()
    if not selected:
        return None
    return Path(selected).expanduser()


def _install_paths(parent_or_root: Path) -> tuple[Path, Path]:
    selected = parent_or_root.resolve()
    if selected.name.lower() == APP_NAME.lower():
        return selected.parent, selected
    return selected, selected / APP_NAME


def _replace_install_tree(zip_path: Path, install_parent: Path, install_root: Path) -> None:
    install_parent.mkdir(parents=True, exist_ok=True)
    if install_root.exists():
        try:
            shutil.rmtree(install_root)
        except PermissionError as exc:
            raise PermissionError(
                f"Could not replace the existing install folder:\n{install_root}\n\n"
                "Close pyNeuroscope and any Explorer windows showing that folder, "
                "then run the installer again or choose a different install location."
            ) from exc
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(install_parent)


def main() -> int:
    quiet = "--quiet" in sys.argv
    zip_path = _bundle_dir() / ZIP_NAME
    if not zip_path.exists():
        if not quiet:
            _message(APP_NAME, f"Installer payload was not found:\n{zip_path}")
        return 1

    try:
        selected_install = _parse_install_parent(sys.argv[1:])
        if selected_install is None:
            selected_install = _default_install_parent() if quiet else _choose_install_parent(_default_install_parent())
        if selected_install is None:
            return 0
        install_parent, install_root = _install_paths(selected_install)
    except Exception as exc:
        if not quiet:
            _error_message(APP_NAME, str(exc))
        return 1
    exe_path = install_root / "pyNeuroscope.exe"

    try:
        _replace_install_tree(zip_path, install_parent, install_root)
    except Exception as exc:
        if not quiet:
            _error_message(APP_NAME, str(exc))
        return 1

    if not exe_path.exists():
        if not quiet:
            _error_message(APP_NAME, f"Installed executable was not found:\n{exe_path}")
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
