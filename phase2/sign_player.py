# phase2/sign_player.py

"""
Sign Player Module
------------------
Multimedia UI and video playback layer for Phase 2 of SignConnect Live.
Takes output from text_to_sign.py and plays corresponding sign language videos
from phase2/assets/.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

# Import text_to_sign function from text_to_sign module
try:
    from text_to_sign import text_to_sign
except ModuleNotFoundError:
    from phase2.text_to_sign import text_to_sign


class SignPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SIGNCONNECT LIVE - Text → Sign")
        self.root.geometry("650x650")
        self.root.resizable(True, True)

        # Base path for video assets (phase2/assets/)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_dir = os.path.join(self.base_dir, "assets")

        # Playback State
        self.recognized_signs = []  # List of tuples: [("I", "i.mp4"), ...]
        self.unsupported_words = []
        self.current_index = -1
        self.cap = None
        self.is_playing = False
        self.after_id = None

        self._build_ui()

    def _build_ui(self):
        # Title Header
        header_frame = ttk.Frame(self.root, padding=10)
        header_frame.pack(fill=tk.X)
        title_label = ttk.Label(
            header_frame,
            text="SIGNCONNECT LIVE\nText → Sign",
            font=("Arial", 16, "bold"),
            justify=tk.CENTER
        )
        title_label.pack()

        # Input Section
        input_frame = ttk.LabelFrame(self.root, text="Input Sentence", padding=10)
        input_frame.pack(fill=tk.X, padx=15, pady=5)

        self.entry_var = tk.StringVar(value="I NEED WATER")
        self.entry_box = ttk.Entry(input_frame, textvariable=self.entry_var, font=("Arial", 11))
        self.entry_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.entry_box.bind("<Return>", lambda event: self.process_input())

        convert_btn = ttk.Button(input_frame, text="Convert to Signs", command=self.process_input)
        convert_btn.pack(side=tk.RIGHT)

        # Output Summary Section
        summary_frame = ttk.Frame(self.root, padding=10)
        summary_frame.pack(fill=tk.X, padx=15, pady=5)

        self.rec_label = ttk.Label(summary_frame, text="Recognized signs:\nNone", font=("Arial", 10), justify=tk.LEFT)
        self.rec_label.pack(anchor=tk.W)

        self.unsupp_label = ttk.Label(summary_frame, text="Unsupported words:\nNone", font=("Arial", 10), justify=tk.LEFT, foreground="gray")
        self.unsupp_label.pack(anchor=tk.W, pady=(5, 0))

        # Sign Selection Buttons Section
        self.selection_frame = ttk.LabelFrame(self.root, text="Select a sign", padding=10)
        self.selection_frame.pack(fill=tk.X, padx=15, pady=5)
        self.buttons_container = ttk.Frame(self.selection_frame)
        self.buttons_container.pack(fill=tk.X)

        # Video Display Area
        player_frame = ttk.LabelFrame(self.root, text="Video Player", padding=10)
        player_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.video_label = ttk.Label(
            player_frame,
            text="[ Video Area ]",
            anchor=tk.CENTER,
            font=("Arial", 12),
            background="#1e1e1e",
            foreground="#ffffff"
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Playback Controls Section
        controls_frame = ttk.Frame(self.root, padding=10)
        controls_frame.pack(fill=tk.X, padx=15, pady=5)

        self.prev_btn = ttk.Button(controls_frame, text="Previous", command=self.play_previous, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.play_btn = ttk.Button(controls_frame, text="Play/Pause", command=self.toggle_play_pause, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = ttk.Button(controls_frame, text="Next", command=self.play_next, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(controls_frame, text="Ready", font=("Arial", 9))
        self.status_label.pack(side=tk.RIGHT, padx=5)

    def process_input(self):
        text = self.entry_var.get().strip()
        if not text:
            return

        # Separate text_to_sign function call
        self.recognized_signs, self.unsupported_words = text_to_sign(text)

        # Display Recognized Signs
        if self.recognized_signs:
            rec_lines = [f"{word:<10} → {filename}" for word, filename in self.recognized_signs]
            rec_str = "Recognized signs:\n" + "\n".join(rec_lines)
            self.rec_label.config(text=rec_str)
        else:
            self.rec_label.config(text="Recognized signs:\nNone")

        # Display Unsupported Words
        if self.unsupported_words:
            unsupp_str = "Unsupported words:\n" + "\n".join(self.unsupported_words)
            self.unsupp_label.config(text=unsupp_str, foreground="#d9534f")
        else:
            self.unsupp_label.config(text="Unsupported words:\nNone", foreground="gray")

        # Rebuild Sign Selection Buttons
        for widget in self.buttons_container.winfo_children():
            widget.destroy()

        if self.recognized_signs:
            for idx, (word, filename) in enumerate(self.recognized_signs):
                btn = ttk.Button(
                    self.buttons_container,
                    text=f"[ {word} ]",
                    command=lambda i=idx: self.select_sign(i)
                )
                btn.pack(side=tk.LEFT, padx=3, pady=2)

            # Auto-select the first recognized sign
            self.select_sign(0)
        else:
            self.stop_video()
            self.current_index = -1
            self.video_label.config(
                image="",
                text="[ No recognized signs found ]",
                background="#1e1e1e",
                foreground="#ffffff"
            )
            self.update_control_states()

    def select_sign(self, index):
        if 0 <= index < len(self.recognized_signs):
            self.current_index = index
            word, filename = self.recognized_signs[index]
            self.load_and_play_video(filename)
            self.update_control_states()

    def load_and_play_video(self, filename):
        self.stop_video()

        # Resolve video path relative to phase2/assets/
        video_path = os.path.join(self.assets_dir, filename)
        rel_path = os.path.join("phase2", "assets", filename)

        # Check if video exists
        if not os.path.exists(video_path):
            error_msg = f"Video not found:\n{rel_path}"
            self.video_label.config(
                image="",
                text=error_msg,
                background="#2a1010",
                foreground="#ff6b6b",
                font=("Arial", 12, "bold")
            )
            self.status_label.config(text=f"Missing: {filename}")
            return

        # Open video using OpenCV
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            error_msg = f"Failed to load video:\n{rel_path}"
            self.video_label.config(
                image="",
                text=error_msg,
                background="#2a1010",
                foreground="#ff6b6b"
            )
            return

        self.is_playing = True
        self.status_label.config(text=f"Playing: {filename}")
        self._update_frame()

    def _update_frame(self):
        if not self.is_playing or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((480, 320), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img)

            self.video_label.img_tk = img_tk
            self.video_label.config(image=img_tk, text="")

            # Frame rate timing (~30 fps)
            self.after_id = self.root.after(33, self._update_frame)
        else:
            # Loop video on end of stream
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.after_id = self.root.after(33, self._update_frame)

    def toggle_play_pause(self):
        if self.cap is None:
            return

        self.is_playing = not self.is_playing
        if self.is_playing:
            self._update_frame()
            self.status_label.config(text="Playing")
        else:
            if self.after_id:
                self.root.after_cancel(self.after_id)
            self.status_label.config(text="Paused")

    def play_previous(self):
        if self.current_index > 0:
            self.select_sign(self.current_index - 1)

    def play_next(self):
        if self.current_index < len(self.recognized_signs) - 1:
            self.select_sign(self.current_index + 1)

    def stop_video(self):
        self.is_playing = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_control_states(self):
        has_signs = len(self.recognized_signs) > 0
        if has_signs and self.current_index >= 0:
            word, filename = self.recognized_signs[self.current_index]
            video_path = os.path.join(self.assets_dir, filename)
            video_exists = os.path.exists(video_path)

            self.prev_btn.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
            self.next_btn.config(state=tk.NORMAL if self.current_index < len(self.recognized_signs) - 1 else tk.DISABLED)
            self.play_btn.config(state=tk.NORMAL if video_exists else tk.DISABLED)
        else:
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.play_btn.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = SignPlayerApp(root)
    # Automatically run initial conversion for default text
    app.process_input()
    root.mainloop()


if __name__ == "__main__":
    main()
