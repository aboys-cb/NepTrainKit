#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Redirect C-level stdout/stderr (printf) within Python.

Temporarily redirects process fds 1/2 so native printf 等输出可被静默或重定向。
"""
from __future__ import annotations

from contextlib import ContextDecorator
from pathlib import Path
import os
import sys
from typing import Optional, Union

_IS_WINDOWS = os.name == "nt"
if _IS_WINDOWS:
    import ctypes
    from ctypes import wintypes
    import msvcrt
    STD_OUTPUT_HANDLE = -11
    STD_ERROR_HANDLE = -12
    GENERIC_WRITE = 0x40000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    CREATE_ALWAYS = 2
    OPEN_ALWAYS = 4
    FILE_ATTRIBUTE_NORMAL = 0x00000080
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    FILE_BEGIN = 0
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    GetStdHandle = kernel32.GetStdHandle
    GetStdHandle.argtypes = [wintypes.DWORD]
    GetStdHandle.restype = wintypes.HANDLE
    SetStdHandle = kernel32.SetStdHandle
    SetStdHandle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
    SetStdHandle.restype = wintypes.BOOL
    CreateFileW = kernel32.CreateFileW
    CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                            wintypes.HANDLE]
    CreateFileW.restype = wintypes.HANDLE
    CloseHandle = kernel32.CloseHandle
    CloseHandle.argtypes = [wintypes.HANDLE]
    CloseHandle.restype = wintypes.BOOL
    SetFilePointerEx = kernel32.SetFilePointerEx
    SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong,
                                 ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
    SetFilePointerEx.restype = wintypes.BOOL

PathLike = Union[str, os.PathLike[str], Path]


class redirect_c_stdout_stderr(ContextDecorator):
    """Temporarily redirect C-level stdout/stderr to file or devnull.

    Parameters
    ----------
    to : PathLike | None, default=None
        None/silent -> devnull；str/Path -> 文件路径
    append : bool
        仅对文件路径有效，控制追加/覆盖。
    """
    def __init__(self, to: Optional[PathLike] = None, *, append: bool = False) -> None:
        self._to = to
        self._append = append
        self._saved_out_fd: Optional[int] = None
        self._saved_err_fd: Optional[int] = None
        self._target_fd: Optional[int] = None
        self._saved_out_os: Optional[int] = None
        self._saved_err_os: Optional[int] = None
        self._py_stdout_old = None
        self._py_stderr_old = None
        self._py_stdout_new = None
        self._py_stderr_new = None

    def __enter__(self):
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass

        self._py_stdout_old = sys.stdout
        self._py_stderr_old = sys.stderr
        self._saved_out_fd = os.dup(1)
        self._saved_err_fd = os.dup(2)

        if _IS_WINDOWS:
            self._saved_out_os = GetStdHandle(STD_OUTPUT_HANDLE)
            self._saved_err_os = GetStdHandle(STD_ERROR_HANDLE)
            if self._to is None:
                target_path = 'NUL'
                create_disp = OPEN_ALWAYS
            else:
                p = Path(self._to)
                p.parent.mkdir(parents=True, exist_ok=True)
                target_path = str(p)
                create_disp = OPEN_ALWAYS if self._append else CREATE_ALWAYS
            handle = CreateFileW(target_path,
                                 GENERIC_WRITE,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE,
                                 None,
                                 create_disp,
                                 FILE_ATTRIBUTE_NORMAL,
                                 None)
            if handle == INVALID_HANDLE_VALUE or handle is None:
                self._target_fd = os.open(os.devnull if self._to is None else os.fspath(target_path),
                                          os.O_WRONLY | os.O_CREAT | (os.O_APPEND if self._append else os.O_TRUNC),
                                          0o666)
                os.dup2(self._target_fd, 1)
                os.dup2(self._target_fd, 2)
                return self
            if self._append and self._to is not None:
                pos = ctypes.c_longlong(0)
                SetFilePointerEx(handle, 0, ctypes.byref(pos), 2)  # FILE_END
            SetStdHandle(STD_OUTPUT_HANDLE, handle)
            SetStdHandle(STD_ERROR_HANDLE, handle)
            fd = msvcrt.open_osfhandle(int(handle), 0)
            self._target_fd = fd
            os.dup2(fd, 1)
            os.dup2(fd, 2)
        else:
            if self._to is None:
                target_path = os.devnull
                flags = os.O_WRONLY
            else:
                p = Path(self._to)
                p.parent.mkdir(parents=True, exist_ok=True)
                target_path = os.fspath(p)
                flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if self._append else os.O_TRUNC)
            self._target_fd = os.open(target_path, flags, 0o666)
            os.dup2(self._target_fd, 1)
            os.dup2(self._target_fd, 2)

        try:
            enc = getattr(self._py_stdout_old, 'encoding', None) or 'utf-8'
            errh = getattr(self._py_stdout_old, 'errors', None) or 'replace'
            self._py_stdout_new = os.fdopen(self._saved_out_fd, 'w', buffering=1, encoding=enc, errors=errh, closefd=False)
        except Exception:
            self._py_stdout_new = os.fdopen(self._saved_out_fd, 'w', buffering=1, closefd=False)
        try:
            enc = getattr(self._py_stderr_old, 'encoding', None) or 'utf-8'
            errh = getattr(self._py_stderr_old, 'errors', None) or 'replace'
            self._py_stderr_new = os.fdopen(self._saved_err_fd, 'w', buffering=1, encoding=enc, errors=errh, closefd=False)
        except Exception:
            self._py_stderr_new = os.fdopen(self._saved_err_fd, 'w', buffering=1, closefd=False)

        sys.stdout = self._py_stdout_new
        sys.stderr = self._py_stderr_new
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            try:
                if self._target_fd is not None:
                    os.fsync(self._target_fd)
            except Exception:
                pass
            try:
                if self._py_stdout_new is not None:
                    self._py_stdout_new.flush()
                if self._py_stderr_new is not None:
                    self._py_stderr_new.flush()
            except Exception:
                pass
            if self._saved_out_fd is not None:
                os.dup2(self._saved_out_fd, 1)
            if self._saved_err_fd is not None:
                os.dup2(self._saved_err_fd, 2)
            if _IS_WINDOWS:
                if self._saved_out_os is not None:
                    SetStdHandle(STD_OUTPUT_HANDLE, self._saved_out_os)
                if self._saved_err_os is not None:
                    SetStdHandle(STD_ERROR_HANDLE, self._saved_err_os)
            if self._py_stdout_old is not None:
                sys.stdout = self._py_stdout_old
            if self._py_stderr_old is not None:
                sys.stderr = self._py_stderr_old
        finally:
            if self._target_fd is not None:
                try:
                    os.close(self._target_fd)
                except Exception:
                    pass
                self._target_fd = None
            if self._saved_out_fd is not None:
                try:
                    os.close(self._saved_out_fd)
                except Exception:
                    pass
                self._saved_out_fd = None
            if self._saved_err_fd is not None:
                try:
                    os.close(self._saved_err_fd)
                except Exception:
                    pass
                self._saved_err_fd = None
            self._py_stdout_new = None
            self._py_stderr_new = None
        return False
