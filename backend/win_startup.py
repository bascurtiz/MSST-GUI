"""Windows startup hygiene.

* hidden_run — subprocess without a console flash (nvidia-smi, etc.).
* thread-local WH_CALLWNDPROC — intercepts WM_WINDOWPOSCHANGING on this
  thread *before* the window is mapped, so stray Qt/combo/drop-shadow
  HWNDs never become visible (an after-the-fact hide still flashes).
"""
from __future__ import annotations

import os
import subprocess
import sys

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def hidden_run(args, **kwargs):
    """subprocess.run with no console window on Windows."""
    kwargs.setdefault("capture_output", True)
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            int(kwargs.get("creationflags") or 0) | _CREATE_NO_WINDOW
        )
        si = kwargs.get("startupinfo") or subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        kwargs["startupinfo"] = si
    return subprocess.run(args, **kwargs)


_keep = set()
_block_shows = False
_cwp_hook = None
_cbt_hook = None
_cwp_cb = None
_cbt_cb = None


def keep_hwnd(hwnd) -> None:
    try:
        h = int(hwnd)
    except Exception:
        return
    if h:
        _keep.add(h)


def strip_native_chrome(hwnd) -> None:
    """Remove the Win32 caption so a frameless Qt window cannot peek an
    'MSST' title bar out from behind a translucent splash card."""
    if sys.platform != "win32" or not hwnd:
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    hwnd = int(hwnd)
    GWL_STYLE = -16
    GWL_EXSTYLE = -20
    WS_POPUP = 0x80000000
    WS_VISIBLE = 0x10000000
    WS_CAPTION = 0x00C00000
    WS_THICKFRAME = 0x00040000
    WS_MINIMIZE = 0x20000000
    WS_MAXIMIZE = 0x01000000
    WS_SYSMENU = 0x00080000
    WS_EX_APPWINDOW = 0x00040000
    WS_EX_DLGMODALFRAME = 0x00000001
    WS_EX_WINDOWEDGE = 0x00000100
    WS_EX_CLIENTEDGE = 0x00000200
    WS_EX_TOOLWINDOW = 0x00000080
    WS_EX_LAYERED = 0x00080000
    WS_EX_TOPMOST = 0x00000008
    SWP_NOSIZE = 0x0001
    SWP_NOMOVE = 0x0002
    SWP_NOZORDER = 0x0004
    SWP_FRAMECHANGED = 0x0020
    SWP_NOACTIVATE = 0x0010

    user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.SetWindowLongPtrW.argtypes = [
        wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t,
    ]

    style = int(user32.GetWindowLongPtrW(hwnd, GWL_STYLE) or 0)
    style &= ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZE | WS_MAXIMIZE | WS_SYSMENU)
    style |= WS_POPUP | WS_VISIBLE
    user32.SetWindowLongPtrW(hwnd, GWL_STYLE, style)

    ex = int(user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE) or 0)
    ex &= ~(WS_EX_APPWINDOW | WS_EX_DLGMODALFRAME | WS_EX_WINDOWEDGE | WS_EX_CLIENTEDGE)
    ex |= WS_EX_TOOLWINDOW | WS_EX_LAYERED | WS_EX_TOPMOST
    user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, ex)

    user32.SetWindowPos(
        hwnd, None, 0, 0, 0, 0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOACTIVATE,
    )

    try:
        dwm = ctypes.windll.dwmapi
        DWMWA_NCRENDERING_POLICY = 2
        DWMNCRP_DISABLED = 1
        val = ctypes.c_int(DWMNCRP_DISABLED)
        dwm.DwmSetWindowAttribute(
            wintypes.HWND(hwnd),
            ctypes.c_uint(DWMWA_NCRENDERING_POLICY),
            ctypes.byref(val),
            ctypes.sizeof(val),
        )
    except Exception:
        pass


def install_show_hook() -> None:
    """Block ShowWindow on this thread except hwnds passed to keep_hwnd()."""
    global _cwp_hook, _cbt_hook, _cwp_cb, _cbt_cb, _block_shows
    if sys.platform != "win32" or _cwp_hook:
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    try:
        console = int(kernel32.GetConsoleWindow() or 0)
        if console:
            _keep.add(console)
    except Exception:
        pass

    WH_CALLWNDPROC = 4
    WH_CBT = 5
    HCBT_CREATEWND = 3
    WM_WINDOWPOSCHANGING = 0x0046
    SWP_SHOWWINDOW = 0x0040
    SWP_HIDEWINDOW = 0x0080
    SWP_NOACTIVATE = 0x0010
    WS_VISIBLE = 0x10000000

    class WINDOWPOS(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("hwndInsertAfter", wintypes.HWND),
            ("x", ctypes.c_int),
            ("y", ctypes.c_int),
            ("cx", ctypes.c_int),
            ("cy", ctypes.c_int),
            ("flags", wintypes.UINT),
        ]

    class CWPSTRUCT(ctypes.Structure):
        _fields_ = [
            ("lParam", wintypes.LPARAM),
            ("wParam", wintypes.WPARAM),
            ("message", wintypes.UINT),
            ("hwnd", wintypes.HWND),
        ]

    class CREATESTRUCTW(ctypes.Structure):
        _fields_ = [
            ("lpCreateParams", ctypes.c_void_p),
            ("hInstance", wintypes.HINSTANCE),
            ("hMenu", wintypes.HMENU),
            ("hwndParent", wintypes.HWND),
            ("cy", ctypes.c_int),
            ("cx", ctypes.c_int),
            ("y", ctypes.c_int),
            ("x", ctypes.c_int),
            ("style", wintypes.LONG),
            ("lpszName", wintypes.LPCWSTR),
            ("lpszClass", wintypes.LPCWSTR),
            ("dwExStyle", wintypes.DWORD),
        ]

    class CBT_CREATEWNDW(ctypes.Structure):
        _fields_ = [
            ("lpcs", ctypes.POINTER(CREATESTRUCTW)),
            ("hwndInsertAfter", wintypes.HWND),
        ]

    HOOKPROC = ctypes.WINFUNCTYPE(
        ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM,
    )

    user32.SetWindowsHookExW.restype = ctypes.c_void_p
    user32.SetWindowsHookExW.argtypes = [
        ctypes.c_int, HOOKPROC, ctypes.c_void_p, wintypes.DWORD,
    ]
    user32.CallNextHookEx.restype = ctypes.c_ssize_t
    user32.CallNextHookEx.argtypes = [
        ctypes.c_void_p, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM,
    ]
    user32.UnhookWindowsHookEx.restype = wintypes.BOOL
    user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]

    def _blocked(hwnd) -> bool:
        if not _block_shows or not hwnd:
            return False
        try:
            return int(hwnd) not in _keep
        except Exception:
            return True

    @HOOKPROC
    def _cwp(nCode, wParam, lParam):
        try:
            if nCode >= 0 and lParam and _block_shows:
                cwp = ctypes.cast(lParam, ctypes.POINTER(CWPSTRUCT)).contents
                if cwp.message == WM_WINDOWPOSCHANGING and cwp.lParam:
                    if _blocked(cwp.hwnd):
                        wp = ctypes.cast(
                            cwp.lParam, ctypes.POINTER(WINDOWPOS),
                        ).contents
                        if wp.flags & SWP_SHOWWINDOW:
                            wp.flags = (
                                (wp.flags & ~SWP_SHOWWINDOW)
                                | SWP_HIDEWINDOW | SWP_NOACTIVATE
                            )
        except Exception:
            pass
        return user32.CallNextHookEx(None, nCode, wParam, lParam)

    @HOOKPROC
    def _cbt(nCode, wParam, lParam):
        try:
            if nCode == HCBT_CREATEWND and lParam and _block_shows:
                cbt = ctypes.cast(
                    lParam, ctypes.POINTER(CBT_CREATEWNDW),
                ).contents
                lpcs = cbt.lpcs.contents
                lpcs.style = int(lpcs.style) & ~WS_VISIBLE
        except Exception:
            pass
        return user32.CallNextHookEx(None, nCode, wParam, lParam)

    _cwp_cb = _cwp
    _cbt_cb = _cbt
    tid = kernel32.GetCurrentThreadId()
    _cwp_hook = user32.SetWindowsHookExW(WH_CALLWNDPROC, _cwp_cb, None, tid)
    _cbt_hook = user32.SetWindowsHookExW(WH_CBT, _cbt_cb, None, tid)
    _block_shows = True
    if not _cwp_hook:
        _cwp_cb = None
    if not _cbt_hook:
        _cbt_cb = None


def uninstall_show_hook() -> None:
    global _cwp_hook, _cbt_hook, _cwp_cb, _cbt_cb, _block_shows
    _block_shows = False
    if sys.platform != "win32":
        return
    import ctypes
    user32 = ctypes.windll.user32
    for handle in (_cwp_hook, _cbt_hook):
        if handle:
            try:
                user32.UnhookWindowsHookEx(handle)
            except Exception:
                pass
    _cwp_hook = None
    _cbt_hook = None
    _cwp_cb = None
    _cbt_cb = None
    _keep.clear()
