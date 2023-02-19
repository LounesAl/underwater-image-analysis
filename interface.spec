# -*- mode: python ; coding: utf-8 -*-

import torch
import inspect
block_cipher = None
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
excluded_modules = ['torch.distributions', 'torch.cuda'] # add this

a = Analysis(
    ['interface.py'],
    pathex=[],
    binaries=[],
    datas=[*collect_data_files("torch", include_py_files=True), *collect_data_files("fvcore", include_py_files=True), *collect_data_files("detectron2", include_py_files=True)],
    hiddenimports=[], # add this
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules, # add this
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
    name='interface',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='data/logo/logo.ico'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='interface',
)
