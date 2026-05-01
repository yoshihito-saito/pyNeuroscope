# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

datas = [
    ("src/pyneuroscope/resources/probe_templates.json", "pyneuroscope/resources"),
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
        "matplotlib",
        "numba",
        "pandas",
        "PIL",
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
