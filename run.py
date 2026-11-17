"""Start FaceWatch from any working directory: python run.py."""
import importlib.util
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'
NODE_DIR = ROOT / '.tools' / 'node-v22.14.0-win-x64'
URL = 'http://127.0.0.1:8000'


def open_interface(path="/"):
    """Open an explicit foreground app window without starting another backend."""
    if not path.startswith('/') or path.startswith('//') or '\\' in path:
        raise ValueError('Đường dẫn giao diện phải nằm trong ứng dụng localhost.')
    destination = URL + path
    if os.name == 'nt':
        locations = [os.environ.get('PROGRAMFILES'), os.environ.get('PROGRAMFILES(X86)'),
                     os.environ.get('LOCALAPPDATA')]
        browsers = ('Google/Chrome/Application/chrome.exe', 'Microsoft/Edge/Application/msedge.exe')
        for browser in browsers:
            for location in locations:
                executable = Path(location) / browser if location else None
                if executable is None or not executable.is_file():
                    continue
                try:
                    subprocess.Popen([str(executable), '--new-window', '--start-maximized', destination],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
                except OSError:
                    continue
    try:
        if webbrowser.open(destination, new=1, autoraise=True):
            return True
    except Exception as exc:
        print(f'Browser launch failed: {exc}', flush=True)
    print(f'Open {destination} in your browser.', flush=True)
    return False


def command(args, *, cwd=ROOT):
    child = subprocess.Popen([str(arg) for arg in args], cwd=cwd)
    try:
        code = child.wait()
    except KeyboardInterrupt:
        # The child shares the console and receives Ctrl+C as well. Allow its
        # FastAPI shutdown handler to release cameras, jobs and temporary files.
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.terminate()
            child.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, args)


def setup():
    print('Preparing local dependencies and official models...', flush=True)
    command(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
             '-File', ROOT / 'setup.ps1'])


def existing_server():
    with socket.socket() as connection:
        connection.settimeout(.5)
        if connection.connect_ex(('127.0.0.1', 8000)) != 0:
            return False
    try:
        with urllib.request.urlopen(URL + '/api/health', timeout=3) as response:
            health = json.load(response)
        if str(health.get('model', '')).startswith('hybrid-lvface-antelopev2-petalface-'):
            return True
    except (OSError, ValueError, urllib.error.URLError):
        pass
    raise OSError('Port 8000 is occupied by another application. Close it before starting FaceWatch.')


def build_frontend():
    frontend = ROOT / 'frontend'
    sources = [frontend / name for name in ('index.html', 'package.json', 'package-lock.json', 'vite.config.js')]
    for directory in ('src', 'public'):
        sources.extend(p for p in (frontend / directory).rglob('*') if p.is_file())
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(path.relative_to(frontend).as_posix().encode())
        digest.update(path.read_bytes())
    signature = digest.hexdigest()
    stamp = ROOT / '.tools/frontend-build.sha256'
    index = frontend / 'dist/index.html'
    usable = index.is_file()
    if usable:
        assets = re.findall(r'''(?:src|href)=["'](/assets/[^"']+)["']''', index.read_text(encoding='utf8'))
        usable = bool(assets) and all((frontend / 'dist' / asset.lstrip('/')).is_file() for asset in assets)
    cached = stamp.is_file() and stamp.read_text(encoding='ascii').strip() == signature
    fresh = not stamp.exists() and usable and all(p.stat().st_mtime_ns <= index.stat().st_mtime_ns for p in sources)
    if usable and (cached or fresh):
        print('[1/3] Frontend ready (no rebuild needed).', flush=True)
    else:
        print('[1/3] Building frontend...', flush=True)
        command([NODE_DIR / 'node.exe', frontend / 'node_modules/vite/bin/vite.js', 'build'], cwd=frontend)
    stamp.parent.mkdir(exist_ok=True)
    stamp.write_text(signature, encoding='ascii')


def _install_console_close_handler(server):
    """Ask Uvicorn to shut down cleanly when its owning console closes."""
    if os.name != 'nt':
        return None
    import ctypes
    from ctypes import wintypes

    handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

    @handler_type
    def handle_console_event(event):
        # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT and CTRL_SHUTDOWN_EVENT.
        if event in (2, 5, 6):
            server.should_exit = True
            return True
        return False

    set_handler = ctypes.windll.kernel32.SetConsoleCtrlHandler
    set_handler.argtypes = (handler_type, wintypes.BOOL)
    set_handler.restype = wintypes.BOOL
    if not set_handler(handle_console_event, True):
        raise ctypes.WinError()
    return handle_console_event  # Keep the callback alive while the server runs.


def serve(open_browser, interface_path="/"):
    import uvicorn

    print('[2/3] Starting backend and loading AI models...', flush=True)
    server = uvicorn.Server(uvicorn.Config('backend.main:app', host='127.0.0.1', port=8000,
                                         ws_max_size=22_000_000, use_colors=False,
                                         timeout_graceful_shutdown=8))
    stopped = threading.Event()
    close_handler = _install_console_close_handler(server)

    def notify_ready():
        while not server.started:
            if stopped.wait(.1):
                return
        print(f'[3/3] Interface ready: {URL}', flush=True)
        print('Loading AI... The camera button becomes available when the models are ready.', flush=True)
        print('Stop: Ctrl+C, Q + Enter, or close this terminal window.', flush=True)
        if open_browser:
            open_interface(interface_path)
        from backend.ai import engine
        while not engine.ready and not engine.error:
            if stopped.wait(.2):
                return
        if engine.error:
            print(f'AI failed to load: {engine.error}', file=sys.stderr, flush=True)
        else:
            print(f'AI READY ({engine.device.upper()}). FaceWatch is ready to use.', flush=True)

    def console_stop():
        def request_stop():
            server.should_exit = True

        # A blocking TextIO read on a Windows pipe can hold a CRT stream lock
        # while NumPy/PyTorch DLLs initialize. Poll without blocking that lock.
        try:
            if sys.stdin is None:
                request_stop()
                return
            buffer = ''
            if os.name == 'nt':
                import ctypes
                from ctypes import wintypes
                import msvcrt
                terminal = sys.stdin.isatty()
                peek = ctypes.windll.kernel32.PeekNamedPipe
                peek.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                 ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                                 ctypes.POINTER(wintypes.DWORD)]
                peek.restype = wintypes.BOOL
                handle = msvcrt.get_osfhandle(sys.stdin.fileno())
            while not stopped.wait(.1):
                if os.name == 'nt':
                    if terminal:
                        if not msvcrt.kbhit():
                            continue
                        chunk = msvcrt.getwch()
                    else:
                        available = wintypes.DWORD()
                        if not peek(handle, None, 0, None, ctypes.byref(available), None):
                            request_stop()
                            return
                        if not available.value:
                            continue
                        chunk = os.read(sys.stdin.fileno(), min(available.value, 4096)).decode('utf8', errors='replace')
                else:
                    import select
                    if not select.select([sys.stdin], [], [], 0)[0]:
                        continue
                    chunk = os.read(sys.stdin.fileno(), 4096).decode('utf8', errors='replace')
                if not chunk:
                    request_stop()
                    return
                for char in chunk:
                    if char == '\x03':
                        server.should_exit = True
                        return
                    if char in '\r\n':
                        if buffer.strip().casefold() in ('q', 'quit', 'exit'):
                            server.should_exit = True
                            return
                        buffer = ''
                    elif char in ('\b', '\x7f'):
                        buffer = buffer[:-1]
                    else:
                        buffer += char
        except (EOFError, OSError, ValueError):
            request_stop()

    threading.Thread(target=notify_ready, daemon=True).start()
    threading.Thread(target=console_stop, daemon=True).start()
    try:
        server.run()
    finally:
        stopped.set()
        if close_handler is not None:
            import ctypes
            ctypes.windll.kernel32.SetConsoleCtrlHandler(close_handler, False)


def main():
    parser = argparse.ArgumentParser(description='Start FaceWatch AI locally.')
    parser.add_argument('--no-browser', action='store_true', help='Do not open the browser automatically.')
    parser.add_argument('--open-path', default='/', help='Open this local application path in a new browser window.')
    args = parser.parse_args()
    if not args.open_path.startswith('/') or args.open_path.startswith('//') or '\\' in args.open_path:
        parser.error('--open-path must be a path within localhost.')
    os.chdir(ROOT)
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONUTF8'] = '1'
    os.environ['NO_COLOR'] = '1'
    os.environ.pop('FORCE_COLOR', None)
    required = [PYTHON, NODE_DIR / 'node.exe', NODE_DIR / 'node_modules/npm/bin/npm-cli.js',
                ROOT / 'frontend/node_modules/vite/bin/vite.js', ROOT / 'models/hybrid/manifest.json']
    if any(not path.exists() for path in required):
        setup()

    if Path(sys.executable).resolve() != PYTHON.resolve():
        command([PYTHON, '-u', ROOT / 'run.py', *sys.argv[1:]])
        return

    modules = ('fastapi', 'uvicorn', 'torch', 'torchvision', 'cv2', 'numpy',
               'onnxruntime', 'onnx', 'timm', 'skimage', 'imageio_ffmpeg', 'multipart')
    if any(importlib.util.find_spec(name) is None for name in modules):
        setup()

    if existing_server():
        print(f'FaceWatch is already running: {URL}', flush=True)
        if not args.no_browser:
            open_interface(args.open_path)
        return

    os.environ['PATH'] = str(NODE_DIR) + os.pathsep + os.environ.get('PATH', '')
    build_frontend()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    serve(open_browser=not args.no_browser, interface_path=args.open_path)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f'Cannot start FaceWatch: {exc}', file=sys.stderr)
        sys.exit(1)
