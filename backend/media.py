from pathlib import Path
import subprocess
import cv2


def normalize_video(source: Path, target: Path):
    """Local H.264 stream for consistent browser/OpenCV orientation and timestamps."""
    import imageio_ffmpeg
    cap = cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():
            raise ValueError('Không mở được video.')
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if width <= 0 or height <= 0 or width*height > 16_000_000:
            raise ValueError('Video không hợp lệ hoặc lớn hơn 16 megapixel mỗi frame.')
    finally:
        cap.release()
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-nostdin',
               '-protocol_whitelist', 'file,pipe', '-i', str(source), '-map', '0:v:0', '-map', '0:a?',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
               '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
               '-fps_mode', 'passthrough', '-c:a', 'aac', '-movflags', '+faststart',
               '-metadata:s:v:0', 'rotate=0', '-y', str(target)]
    flags = subprocess.CREATE_NO_WINDOW if __import__('os').name == 'nt' else 0
    completed = subprocess.run(command, capture_output=True, timeout=3600, creationflags=flags)
    if completed.returncode:
        target.unlink(missing_ok=True)
        raise ValueError('Không chuẩn hóa được video: '+completed.stderr.decode('utf8', errors='replace')[-500:])
