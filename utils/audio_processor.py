import os
import sys
import glob
import shutil
import yt_dlp
from pydub import AudioSegment
from pydub.utils import which

def ensure_ffmpeg():
    """Detect and configure ffmpeg executable on system PATH and for pydub/yt-dlp."""
    ffmpeg_exe = which("ffmpeg")
    if not ffmpeg_exe:
        # Search common Windows locations
        candidates = []
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            winget_pattern = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "*ffmpeg*", "**", "ffmpeg.exe")
            candidates.extend(glob.glob(winget_pattern, recursive=True))

        venv_scripts = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Scripts"))
        candidates.append(os.path.join(venv_scripts, "ffmpeg.exe"))
        candidates.extend([
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        ])

        for c in candidates:
            if os.path.isfile(c):
                ffmpeg_dir = os.path.dirname(c)
                if ffmpeg_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
                AudioSegment.converter = c
                break

ensure_ffmpeg()

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url: str) -> str:
    url = url.strip().strip('"').strip("'")
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": False,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        raw_filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(raw_filename)
        expected_wav = base + ".wav"

        if os.path.exists(expected_wav):
            return expected_wav

        # Fallback to the most recently generated WAV file in DOWNLOAD_DIR
        wav_files = glob.glob(os.path.join(DOWNLOAD_DIR, "*.wav"))
        if wav_files:
            return max(wav_files, key=os.path.getmtime)

        return expected_wav

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    input_path = input_path.strip().strip('"').strip("'")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Local media file not found: {input_path}")

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16khz mono for Whisper/Sarvam
    audio.export(output_path, format="wav")
    return output_path

def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found for chunking: {wav_path}")

    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    if len(audio) <= chunk_ms:
        chunk_path = f"{wav_path}_chunk_0.wav"
        audio.export(chunk_path, format="wav")
        return [chunk_path]

    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks

def process_input(source: str) -> list:
    source = source.strip().strip('"').strip("'")
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...", flush=True)
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...", flush=True)
        wav_path = convert_to_wav(source)

    print("Chunking audio...", flush=True)
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.", flush=True)
    return chunks



