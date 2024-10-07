import tkinter as tk
from tkinter import filedialog, messagebox, Label, Entry, Button, Frame
from tkinterdnd2 import TkinterDnD, DND_FILES
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
import threading


# Convert data to binary
def to_bin(data):
    if isinstance(data, str):
        return ''.join([format(ord(i), "08b") for i in data])
    elif isinstance(data, bytes) or isinstance(data, np.ndarray):
        return [format(i, "08b") for i in data]
    elif isinstance(data, int) or isinstance(data, np.uint8):
        return format(data, "08b")
    else:
        raise TypeError("Unsupported data type.")


# Embedding/Extraction unified function
def modify_bits(frame, bits=None, lsb_count=1, mode='hide'):
    flat_frame = frame.reshape(-1, 3)
    if mode == 'hide' and bits is not None:
        for idx, pixel in enumerate(flat_frame):
            if idx < len(bits):
                bit_segment = bits[idx: idx + lsb_count].ljust(lsb_count, '0')
                pixel[0] = (pixel[0] & (255 - ((1 << lsb_count) - 1))) | int(bit_segment, 2)
        return flat_frame.reshape(frame.shape)
    elif mode == 'extract':
        extracted_bits = [(pixel[0] & ((1 << lsb_count) - 1)) for pixel in flat_frame]
        return [str(bit) for bit in extracted_bits]
    else:
        raise ValueError("Invalid mode. Use 'hide' or 'extract'.")


# Hide data inside video
def hide_data_in_video(video_file, secret_message, output_path, lsb_count):
    video = VideoFileClip(video_file)
    frames = list(video.iter_frames())

    secret_message += "#####\0"  # Delimiter to mark the end of the message
    bits = to_bin(secret_message)

    bit_idx = 0
    for frame_idx, frame in enumerate(frames):
        frame = frame.astype(np.uint8)
        if bit_idx < len(bits):
            # Prepare the number of bits to be modified
            num_pixels = frame.size // 3  # Since frame is H x W x 3 (RGB)
            frame_bits = bits[bit_idx:bit_idx + num_pixels * lsb_count]

            # Modify bits of the current frame
            modified_frame = modify_bits(frame, bits=frame_bits, lsb_count=lsb_count, mode='hide')
            frames[frame_idx] = modified_frame

            # Update bit index for next frame
            bit_idx += len(frame_bits)
        if bit_idx >= len(bits):
            break

    new_video = ImageSequenceClip(frames, fps=video.fps)
    new_video.write_videofile(output_path, codec='libx264', audio_codec="aac", bitrate="2000k", preset="medium")

    return bit_idx


# Extract data from video
def extract_data_from_video_optimized(video_file, lsb_count, progress_callback=None):
    video = VideoFileClip(video_file)
    total_frames = int(video.fps * video.duration)
    binary_delimiter = to_bin("#####\0")
    extracted_bits = []
    found_delimiter = False

    for frame_idx, frame in enumerate(video.iter_frames()):
        frame = frame.astype(np.uint8)
        frame_bits = modify_bits(frame, lsb_count=lsb_count, mode='extract')
        extracted_bits.extend(frame_bits)

        if progress_callback:
            progress_callback((frame_idx + 1) / total_frames)

        # Debugging output (optional)
        if frame_idx % 10 == 0:  # Limit to every 10 frames for performance
            extracted_string = ''.join(extracted_bits)
            print(f"Extracted so far (frame {frame_idx}): {extracted_string[:50]}...")  # Print first 50 chars

        # Check for delimiter in the extracted bits
        if ''.join(extracted_bits).find(binary_delimiter) != -1:
            found_delimiter = True
            break

    if found_delimiter:
        # Convert extracted bits to bytes and decode message
        message_bytes = [''.join(extracted_bits[i:i + 8]) for i in range(0, len(extracted_bits), 8)]
        decoded_message = ''.join([chr(int(byte, 2)) for byte in message_bytes if byte])
        hidden_message = decoded_message.split("#####")[0]
        return hidden_message
    else:
        return None


# GUI Implementation
class SteganographyApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Steganography Tool")
        self.geometry("700x600")
        self.configure(bg='#f0f0f0')

        # Header
        self.header_label = Label(self, text="Steganography Tool", font=("Helvetica", 18, "bold"), bg='#f0f0f0')
        self.header_label.pack(pady=20)

        # Labels
        self.label_drag = Label(self, text="Select your media file (Video):", font=("Helvetica", 12), bg='#f0f0f0')
        self.label_drag.pack(pady=10)

        # Video file selection entry and button
        self.video_entry = Entry(self, width=50, font=("Helvetica", 12), relief="sunken", borderwidth=2)
        self.video_entry.pack(pady=5)
        self.select_video_button = Button(self, text="Select Video", command=self.select_video, font=("Helvetica", 12),
                                          bg='#007acc', fg='white', relief="raised")
        self.select_video_button.pack(pady=5)

        # Payload file selection entry and button
        self.payload_entry = Entry(self, width=50, font=("Helvetica", 12), relief="sunken", borderwidth=2)
        self.payload_entry.pack(pady=5)
        self.select_payload_button = Button(self, text="Select Payload", command=self.select_payload,
                                            font=("Helvetica", 12), bg='#007acc', fg='white', relief="raised")
        self.select_payload_button.pack(pady=5)

        # Frame for user input (message and LSBs)
        input_frame = Frame(self, bg='#f0f0f0')
        input_frame.pack(pady=20)

        # LSB slider
        self.lsb_label = Label(input_frame, text="Select number of LSBs to use:", font=("Helvetica", 12), bg='#f0f0f0')
        self.lsb_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.lsb_slider = tk.Scale(input_frame, from_=1, to=8, orient=tk.HORIZONTAL, length=200, font=("Helvetica", 12),
                                   bg='#f0f0f0', troughcolor='#dc3545', relief="flat")
        self.lsb_slider.set(3)
        self.lsb_slider.grid(row=0, column=1, padx=10, pady=10)

        # Progress label for extraction
        self.progress_label = Label(self, text="Progress: 0%", font=("Helvetica", 12), bg='#f0f0f0')
        self.progress_label.pack(pady=10)

        # Frame for buttons
        button_frame = Frame(self, bg='#f0f0f0')
        button_frame.pack(pady=20)

        # Buttons for hiding and extracting
        self.hide_button = Button(button_frame, text="Hide Message", command=self.threaded_hide_message,
                                  font=("Helvetica", 12), width=15, bg='#007acc', fg='white', relief="raised")
        self.hide_button.grid(row=0, column=0, padx=15, pady=10)
        self.extract_button = Button(button_frame, text="Extract Message", command=self.threaded_extract_message,
                                     font=("Helvetica", 12), width=15, bg='#007acc', fg='white', relief="raised")
        self.extract_button.grid(row=0, column=1, padx=15, pady=10)

        # Exit button
        self.exit_button = Button(button_frame, text="Quit", command=self.quit, font=("Helvetica", 12), width=15,
                                  bg='#dc3545', fg='white', relief="raised")
        self.exit_button.grid(row=0, column=2, padx=15, pady=10)

        # Variables for file paths
        self.media_path = None
        self.text_file_path = None

    def select_video(self):
        """Select video file."""
        video_file = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if video_file:
            self.media_path = video_file
            self.video_entry.delete(0, tk.END)
            self.video_entry.insert(0, video_file)

    def select_payload(self):
        """Select payload file (text file)."""
        payload_file = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if payload_file:
            self.text_file_path = payload_file
            self.payload_entry.delete(0, tk.END)
            self.payload_entry.insert(0, payload_file)

    def hide_message(self):
        if not self.media_path:
            messagebox.showerror("Error", "Please select a video file first")
            return

        if not self.text_file_path:
            messagebox.showerror("Error", "Please select a payload file first")
            return

        # Read the message from the payload file
        try:
            with open(self.text_file_path, 'r') as f:
                secret_message = f.read()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read payload file: {e}")
            return

        lsb_count = self.lsb_slider.get()

        try:
            output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
            if output_path:
                bits_embedded = hide_data_in_video(self.media_path, secret_message, output_path, lsb_count)
                messagebox.showinfo("Success", f"Successfully hidden {bits_embedded} bits.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not hide message: {e}")

    def extract_message(self):
        if not self.media_path:
            messagebox.showerror("Error", "Please select a video file first")
            return

        print(f"Media Path: {self.media_path}")  # Check the path

        lsb_count = self.lsb_slider.get()

        try:
            hidden_message = extract_data_from_video_optimized(self.media_path, lsb_count,
                                                               progress_callback=self.update_progress)
            if hidden_message:
                messagebox.showinfo("Extracted Message", f"Hidden message: {hidden_message}")
            else:
                messagebox.showwarning("No Message", "No hidden message found.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def threaded_hide_message(self):
        """Run hide message in a separate thread."""
        threading.Thread(target=self.hide_message).start()

    def threaded_extract_message(self):
        """Run extract message in a separate thread."""
        threading.Thread(target=self.extract_message).start()

    def update_progress(self, progress):
        """Update progress bar in GUI."""
        self.progress_label.config(text=f"Progress: {progress * 100:.2f}%")
        self.update_idletasks()


if __name__ == "__main__":
    app = SteganographyApp()
    app.mainloop()
