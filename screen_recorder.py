"""
Screen Recorder — Desktop Application
Records your screen with optional microphone audio.

Requirements:
    pip install mss opencv-python numpy pyaudio moviepy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import os
import shutil
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import mss

try:
    import sounddevice as sd
    import soundfile as sf
    _AUDIO_OK = True
except ImportError:
    _AUDIO_OK = False

# ── Colours / fonts ───────────────────────────────────────────────────────────
BG_DARK     = "#1a1a2e"
BG_CARD     = "#16213e"
BG_INPUT    = "#0f3460"
ACCENT      = "#e94560"
ACCENT_H    = "#c73652"
GREEN       = "#4caf50"
GREEN_H     = "#388e3c"
YELLOW      = "#ff9800"
YELLOW_H    = "#e65100"
TEXT        = "#eaeaea"
MUTED       = "#a0a0b0"
FONT_TITLE  = ("Segoe UI", 22, "bold")
FONT_SUB    = ("Segoe UI", 11)
FONT        = ("Segoe UI", 10)
FONT_SM     = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 14, "bold")


# ── Hover Button ──────────────────────────────────────────────────────────────
class HoverButton(tk.Button):
    def __init__(self, master, normal, hover, **kw):
        super().__init__(master, bg=normal, activebackground=hover, **kw)
        self.bind("<Enter>", lambda _: self.config(bg=hover))
        self.bind("<Leave>", lambda _: self.config(bg=normal))


# ── Recorder Engine ───────────────────────────────────────────────────────────
class Recorder:
    """Captures screen frames and optional audio in background threads."""

    def __init__(self):
        self.is_recording  = False
        self.is_paused     = False
        self.output_path   = ""
        self._fps          = 30
        self._monitor_idx  = 1
        self._record_audio = False
        self._audio_frames: list = []
        self._audio_queue: queue.Queue = queue.Queue()
        self._start_time: float | None = None
        self._vid_thread: threading.Thread | None = None
        self._aud_thread: threading.Thread | None = None
        self._temp_vid     = ""
        self._temp_aud     = ""

    # ── public API ────────────────────────────────────────────────────────────
    def start(self, output_path: str, fps: int, record_audio: bool, monitor_idx: int):
        self.output_path   = output_path
        self._fps          = fps
        self._record_audio = record_audio and _AUDIO_OK
        self._monitor_idx  = monitor_idx
        self.is_recording  = True
        self.is_paused     = False
        self._audio_frames = []
        self._audio_queue  = queue.Queue()
        self._start_time   = time.time()

        base             = os.path.splitext(output_path)[0]
        self._temp_vid   = base + "__vid_tmp.avi"
        self._temp_aud   = base + "__aud_tmp.wav"

        self._vid_thread = threading.Thread(target=self._capture_video, daemon=True)
        self._vid_thread.start()

        if self._record_audio:
            self._aud_thread = threading.Thread(target=self._capture_audio, daemon=True)
            self._aud_thread.start()

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.is_recording = False
        if self._vid_thread:
            self._vid_thread.join(timeout=10)
        if self._aud_thread:
            self._aud_thread.join(timeout=10)

        audio_ready = (
            self._record_audio
            and os.path.exists(self._temp_aud)
        )
        print(f"[stop] audio_ready={audio_ready}, temp_vid={os.path.exists(self._temp_vid)}, temp_aud={os.path.exists(self._temp_aud)}")
        if audio_ready:
            self._merge(self._temp_vid, self._temp_aud, self.output_path)
        elif os.path.exists(self._temp_vid):
            shutil.move(self._temp_vid, self.output_path)

    def elapsed(self) -> str:
        if not self._start_time:
            return "00:00:00"
        t = int(time.time() - self._start_time)
        return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

    # ── internal ──────────────────────────────────────────────────────────────
    def _capture_video(self):
        with mss.MSS() as sct:
            mon    = sct.monitors[self._monitor_idx]
            w, h   = mon["width"], mon["height"]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out    = cv2.VideoWriter(self._temp_vid, fourcc, float(self._fps), (w, h))
            interval = 1.0 / self._fps
            try:
                while self.is_recording:
                    if self.is_paused:
                        time.sleep(0.05)
                        continue
                    t0    = time.perf_counter()
                    frame = cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2BGR)
                    out.write(frame)
                    lag = interval - (time.perf_counter() - t0)
                    if lag > 0:
                        time.sleep(lag)
            finally:
                out.release()

    def _capture_audio(self):
        CH, RATE = 1, 44100

        def callback(indata, frames, time_info, status):
            if not self.is_paused and self.is_recording:
                self._audio_queue.put(indata.copy())

        try:
            with sd.InputStream(samplerate=RATE, channels=CH,
                                dtype="int16", callback=callback):
                while self.is_recording:
                    time.sleep(0.05)
                # drain remaining frames after stop signal
                time.sleep(0.1)
        except Exception as e:
            print(f"[audio error] {e}")
            self._record_audio = False
            return

        frames = []
        while not self._audio_queue.empty():
            try:
                frames.append(self._audio_queue.get_nowait())
            except queue.Empty:
                break

        print(f"[audio] captured {len(frames)} chunks")
        if frames:
            audio_np = np.concatenate(frames, axis=0)
            sf.write(self._temp_aud, audio_np, RATE)
            print(f"[audio] saved to {self._temp_aud}")

    @staticmethod
    def _merge(vid: str, aud: str, out: str):
        try:
            import imageio_ffmpeg
            import subprocess
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg, "-y",
                "-i", vid,
                "-i", aud,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                out
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr[-500:])
            print(f"[merge] success → {out}")
        except Exception as e:
            print(f"[merge error] {e}")
            if os.path.exists(vid):
                shutil.move(vid, out)
        finally:
            for f in (vid, aud):
                if f and os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass


# ── GUI ───────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Tk):
        self.root    = root
        root.title("Screen Recorder")
        root.geometry("580x710")
        root.resizable(False, False)
        root.configure(bg=BG_DARK)

        self._rec     = Recorder()
        self._ticking = False

        self._fps_v = tk.StringVar(value="30")
        self._fmt_v = tk.StringVar(value="MP4")
        self._aud_v = tk.BooleanVar(value=_AUDIO_OK)
        self._dir_v = tk.StringVar(value=str(Path.home() / "Videos"))

        self._build()
        self._load_monitors()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build(self):
        # Header
        hdr = tk.Frame(self.root, bg=BG_DARK)
        hdr.pack(fill="x", padx=24, pady=(24, 0))
        tk.Label(hdr, text="🎥", font=("Segoe UI", 28),
                 bg=BG_DARK, fg=ACCENT).pack(side="left")
        ti = tk.Frame(hdr, bg=BG_DARK)
        ti.pack(side="left", padx=(12, 0))
        tk.Label(ti, text="Screen Recorder", font=FONT_TITLE,
                 bg=BG_DARK, fg=TEXT).pack(anchor="w")
        tk.Label(ti, text="Record your screen with optional microphone audio",
                 font=FONT_SUB, bg=BG_DARK, fg=MUTED).pack(anchor="w")

        # Status bar
        sb = tk.Frame(self.root, bg=BG_CARD)
        sb.pack(fill="x", padx=24, pady=(20, 0))
        si = tk.Frame(sb, bg=BG_CARD)
        si.pack(fill="x", padx=16, pady=12)
        self._dot    = tk.Label(si, text="⬤", font=("Segoe UI", 14),
                                bg=BG_CARD, fg=MUTED)
        self._dot.pack(side="left")
        self._status = tk.Label(si, text="Ready to record",
                                font=FONT, bg=BG_CARD, fg=MUTED)
        self._status.pack(side="left", padx=(8, 0))
        self._clock  = tk.Label(si, text="00:00:00",
                                font=FONT_MONO, bg=BG_CARD, fg=TEXT)
        self._clock.pack(side="right")

        # Settings card
        sc = tk.Frame(self.root, bg=BG_CARD)
        sc.pack(fill="x", padx=24, pady=(16, 0))
        tk.Label(sc, text="SETTINGS", font=("Segoe UI", 8, "bold"),
                 bg=BG_CARD, fg=MUTED).pack(anchor="w", padx=16, pady=(12, 8))

        # Monitor
        r = self._row(sc, "Monitor")
        self._mon = ttk.Combobox(r, state="readonly", width=26, font=FONT)
        self._mon.pack(side="left")

        # FPS
        r = self._row(sc, "Frame Rate")
        for v in ("15", "24", "30", "60"):
            tk.Radiobutton(r, text=f"{v} FPS", variable=self._fps_v, value=v,
                           bg=BG_CARD, fg=TEXT, selectcolor=BG_INPUT,
                           activebackground=BG_CARD, activeforeground=TEXT,
                           font=FONT).pack(side="left", padx=(0, 6))

        # Format
        r = self._row(sc, "Format")
        for v in ("MP4", "AVI"):
            tk.Radiobutton(r, text=v, variable=self._fmt_v, value=v,
                           bg=BG_CARD, fg=TEXT, selectcolor=BG_INPUT,
                           activebackground=BG_CARD, activeforeground=TEXT,
                           font=FONT).pack(side="left", padx=(0, 6))

        # Audio
        r = self._row(sc, "Audio", pad_b=12)
        cb = tk.Checkbutton(r, text="Record microphone audio",
                            variable=self._aud_v,
                            bg=BG_CARD, fg=TEXT, selectcolor=BG_INPUT,
                            activebackground=BG_CARD, activeforeground=TEXT,
                            font=FONT)
        cb.pack(side="left")
        if not _AUDIO_OK:
            cb.config(state="disabled")
            tk.Label(r, text="  (install pyaudio to enable)",
                     font=FONT_SM, bg=BG_CARD, fg=MUTED).pack(side="left")

        # Save location
        fc = tk.Frame(self.root, bg=BG_CARD)
        fc.pack(fill="x", padx=24, pady=(16, 0))
        tk.Label(fc, text="SAVE LOCATION", font=("Segoe UI", 8, "bold"),
                 bg=BG_CARD, fg=MUTED).pack(anchor="w", padx=16, pady=(12, 8))
        fr = tk.Frame(fc, bg=BG_CARD)
        fr.pack(fill="x", padx=16, pady=(0, 12))
        tk.Entry(fr, textvariable=self._dir_v, bg=BG_INPUT, fg=TEXT,
                 font=FONT, relief="flat",
                 insertbackground=TEXT).pack(side="left", fill="x",
                                             expand=True, ipady=6)
        HoverButton(fr, BG_INPUT, ACCENT, text=" Browse ", font=FONT,
                    fg=TEXT, relief="flat", cursor="hand2",
                    command=self._browse).pack(side="left", padx=(8, 0), ipady=6)

        # Control buttons
        self._btn_frame = tk.Frame(self.root, bg=BG_DARK)
        self._btn_frame.pack(fill="x", padx=24, pady=(24, 0))

        # Start button (visible when idle)
        self._start_btn = HoverButton(self._btn_frame, GREEN, GREEN_H,
                                      text="⬤  Start Recording",
                                      font=("Segoe UI", 12, "bold"),
                                      fg="white", relief="flat",
                                      cursor="hand2", command=self._start)
        self._start_btn.pack(fill="x", ipady=12)

        # Recording controls (hidden until recording starts)
        self._rec_frame = tk.Frame(self.root, bg=BG_DARK)
        self._rec_frame.pack(fill="x", padx=24, pady=(10, 0))
        self._pause_btn = HoverButton(self._rec_frame, YELLOW, YELLOW_H,
                                      text="⏸  Pause",
                                      font=("Segoe UI", 12, "bold"),
                                      fg="white", relief="flat",
                                      cursor="hand2",
                                      command=self._toggle_pause)
        self._pause_btn.pack(side="left", fill="x", expand=True, ipady=12)
        self._stop_btn  = HoverButton(self._rec_frame, ACCENT, ACCENT_H,
                                      text="⏹  Stop & Save",
                                      font=("Segoe UI", 12, "bold"),
                                      fg="white", relief="flat",
                                      cursor="hand2", command=self._stop)
        self._stop_btn.pack(side="left", fill="x", expand=True,
                            ipady=12, padx=(10, 0))
        self._rec_frame.pack_forget()   # hidden at start

        self._saved = tk.Label(self.root, text="", font=FONT_SM,
                               bg=BG_DARK, fg=GREEN, wraplength=530)
        self._saved.pack(pady=(14, 0))

        tk.Label(self.root,
                 text="Tip: Recordings are saved to your chosen folder when you click Stop & Save.",
                 font=FONT_SM, bg=BG_DARK, fg=MUTED).pack(side="bottom", pady=16)

    def _row(self, parent: tk.Frame, label: str, pad_b: int = 8) -> tk.Frame:
        f = tk.Frame(parent, bg=BG_CARD)
        f.pack(fill="x", padx=16, pady=(0, pad_b))
        tk.Label(f, text=label, font=FONT, bg=BG_CARD, fg=MUTED,
                 width=12, anchor="w").pack(side="left")
        return f

    def _load_monitors(self):
        with mss.MSS() as sct:
            mons = sct.monitors[1:]   # index 0 = "all monitors" in mss
        vals = [f"Monitor {i + 1}  ({m['width']} × {m['height']})"
                for i, m in enumerate(mons)]
        self._mon["values"] = vals
        if vals:
            self._mon.current(0)

    # ── actions ───────────────────────────────────────────────────────────────
    def _browse(self):
        d = filedialog.askdirectory(initialdir=self._dir_v.get())
        if d:
            self._dir_v.set(d)

    def _out_path(self) -> str:
        ext = self._fmt_v.get().lower()
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self._dir_v.get(), f"Recording_{ts}.{ext}")

    def _start(self):
        d = self._dir_v.get()
        if not os.path.isdir(d):
            messagebox.showerror("Error", f"Save folder not found:\n{d}")
            return
        self._rec.start(
            output_path  = self._out_path(),
            fps          = int(self._fps_v.get()),
            record_audio = self._aud_v.get(),
            monitor_idx  = self._mon.current() + 1,
        )
        # Hide Start, show Pause + Stop
        self._btn_frame.pack_forget()
        self._rec_frame.pack(fill="x", padx=24, pady=(24, 0))
        self._dot.config(fg=ACCENT)
        self._status.config(text="Recording…", fg=ACCENT)
        self._saved.config(text="")
        self._ticking = True
        self._tick()

    def _toggle_pause(self):
        if self._rec.is_paused:
            self._rec.resume()
            self._pause_btn.config(text="⏸  Pause")
            self._dot.config(fg=ACCENT)
            self._status.config(text="Recording…", fg=ACCENT)
            self._ticking = True
            self._tick()
        else:
            self._rec.pause()
            self._pause_btn.config(text="▶  Resume")
            self._dot.config(fg=YELLOW)
            self._status.config(text="Paused", fg=YELLOW)
            self._ticking = False

    def _stop(self):
        self._ticking = False
        self._pause_btn.config(state="disabled")
        self._stop_btn.config(state="disabled")
        self._dot.config(fg=YELLOW)
        self._status.config(text="Saving file… please wait", fg=YELLOW)
        path = self._rec.output_path
        threading.Thread(target=self._finish, args=(path,), daemon=True).start()

    def _finish(self, path: str):
        self._rec.stop()
        self.root.after(0, self._done, path)

    def _done(self, path: str):
        # Hide Pause+Stop, show Start again
        self._rec_frame.pack_forget()
        self._btn_frame.pack(fill="x", padx=24, pady=(24, 0))
        self._pause_btn.config(text="⏸  Pause")
        self._dot.config(fg=GREEN)
        self._status.config(text="Saved successfully!", fg=GREEN)
        self._clock.config(text="00:00:00")
        self._saved.config(text=f"✔  Saved: {path}")

    def _tick(self):
        if self._ticking and self._rec.is_recording and not self._rec.is_paused:
            self._clock.config(text=self._rec.elapsed())
            self.root.after(1000, self._tick)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
