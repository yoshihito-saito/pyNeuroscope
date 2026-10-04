# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

datas = [
    ("src/pyneuroscope/resources/probe_templates.json", "pyneuroscope/resources"),
    ("src/pyneuroscope/resources/neuropixels_geometries.json", "pyneuroscope/resources"),
    ("logo/logo.ico", "pyneuroscope/resources"),
]

a = Analysis(
    ["pyneuroscope_launcher.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jedi",
        "numba",
        "pandas",
        "pytest",
        "setuptools",
        "sympy",
        "torch",
        "torchvision",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
# Qt on Windows uses the OS ICU API. Conda/Poppler DLLs found on PATH can
# expose a different ABI and prevent QtCore from loading in the frozen app.
a.binaries = [entry for entry in a.binaries
              if entry[0].rsplit("\\", 1)[-1].rsplit("/", 1)[-1].lower()
              not in {"icuuc.dll", "icuin.dll", "icudt.dll"}]
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="pyNeuroscope",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="logo/logo.ico",
    version="tools/windows_version_info.txt",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="pyNeuroscope",
)
