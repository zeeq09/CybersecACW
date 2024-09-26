import tkinter as tk
from tkinter import filedialog, messagebox, Label, Entry, Button
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
import numpy as np
import cv2
import wave
import struct
from moviepy.editor import VideoFileClip, ImageSequenceClip
import os

# Convert data to binary
def to_bin(data):
    """Converts data to binary format as a string."""
    if isinstance(data, str):
        return ''.join([format(ord(i), "08b") for i in data])
    elif isinstance(data, bytes) or isinstance(data, np.ndarray):
        return [format(i, "08b") for i in data]
    elif isinstance(data, int) or isinstance(data, np.uint8):
        return format(data, "08b")
    else:
        raise TypeError("Unsupported data type.")

### IMAGE STEGANOGRAPHY ###
def hide_data_in_image(image, secret_message, lsb_count):
    """Hides secret data in an image using LSB."""
    n_bytes = image.size * 3 * lsb_count // 8  # Calculate maximum bytes
    if len(secret_message) > n_bytes:
        raise ValueError("Payload too large for the selected image or LSBs.")
    
    secret_message += "#####"  # delimiter to mark end of message
    binary_secret_message = to_bin(secret_message)
    data_len = len(binary_secret_message)
    
    data_index = 0
    for row in image:
        for pixel in row:
            for i in range(3):  # Modify R, G, and B channels
                if data_index < data_len:
                    pixel[i] = (pixel[i] & (255 - (1 << lsb_count) + 1)) | int(
                        binary_secret_message[data_index:data_index + lsb_count], 2)
                    data_index += lsb_count
                if data_index >= data_len:
                    break
    return image

def extract_data_from_image(image, lsb_count):
    """Extracts hidden data from the image."""
    binary_data = ""
    for row in image:
        for pixel in row:
            for i in range(3):
                binary_data += format(pixel[i] & ((1 << lsb_count) - 1), f'0{lsb_count}b')

    # Split by 8 bits and convert from binary to ASCII
    message_bytes = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]
    decoded_data = ""
    for byte in message_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data[-5:] == "#####":
            break
    return decoded_data[:-5]

### AUDIO STEGANOGRAPHY ###
def hide_data_in_audio(audio_path, secret_message, output_path, lsb_count):
    """Hides secret data inside an audio file using LSB."""
    song = wave.open(audio_path, mode='rb')
    n_frames = song.getnframes()
    n_channels = song.getnchannels()
    sample_width = song.getsampwidth()
    params = song.getparams()

    frames = song.readframes(n_frames)
    song.close()

    total_samples = n_frames * n_channels

    # Convert frames to samples
    if sample_width == 1:  # 8-bit samples are unsigned
        fmt = '<' + str(total_samples) + 'B'
    elif sample_width == 2:  # 16-bit samples are signed
        fmt = '<' + str(total_samples) + 'h'
    else:
        raise ValueError("Only supports 8-bit and 16-bit audio formats.")

    samples = list(struct.unpack(fmt, frames))

    # Prepare the message
    secret_message += "#####"  # delimiter
    bits = ''.join([format(ord(c), '08b') for c in secret_message])

    max_bits_to_hide = total_samples * lsb_count
    if len(bits) > max_bits_to_hide:
        raise ValueError("Payload too large for the selected audio file.")

    # Modify the LSBs of audio samples
    data_index = 0
    for i in range(len(samples)):
        if data_index + lsb_count <= len(bits):
            # Modify the LSBs
            sample = samples[i]
            # Clear the least significant bits
            sample &= ~((1 << lsb_count) - 1)
            # Set the new least significant bits
            bits_to_hide = bits[data_index:data_index + lsb_count]
            sample |= int(bits_to_hide, 2)
            samples[i] = sample
            data_index += lsb_count
        else:
            break

    # Pack samples back to bytes
    frames_modified = struct.pack(fmt, *samples)

    # Write modified frames to new audio file
    with wave.open(output_path, 'wb') as fd:
        fd.setparams(params)
        fd.writeframes(frames_modified)

def extract_data_from_audio(audio_path, lsb_count):
    """Extracts hidden data from an audio file."""
    song = wave.open(audio_path, mode='rb')
    n_frames = song.getnframes()
    n_channels = song.getnchannels()
    sample_width = song.getsampwidth()
    params = song.getparams()

    frames = song.readframes(n_frames)
    song.close()

    total_samples = n_frames * n_channels

    # Convert frames to samples
    if sample_width == 1:
        fmt = '<' + str(total_samples) + 'B'
    elif sample_width == 2:
        fmt = '<' + str(total_samples) + 'h'
    else:
        raise ValueError("Only supports 8-bit and 16-bit audio formats.")

    samples = list(struct.unpack(fmt, frames))

    # Extract the LSBs
    extracted_bits = ""
    for sample in samples:
        # Get the least significant bits
        bits = format(sample & ((1 << lsb_count) - 1), f'0{lsb_count}b')
        extracted_bits += bits

    # Convert bits to bytes
    message_bytes = [extracted_bits[i:i + 8] for i in range(0, len(extracted_bits), 8)]
    decoded_message = ""
    for byte in message_bytes:
        decoded_message += chr(int(byte, 2))
        if decoded_message[-5:] == "#####":
            break
    return decoded_message[:-5]

### VIDEO STEGANOGRAPHY ###
def hide_data_in_video(video_path, secret_message, output_path, lsb_count):
    """Hides secret data inside a video file using LSB."""
    video = VideoFileClip(video_path)
    frames = list(video.iter_frames())  # Get all frames

    secret_message += "#####"  # delimiter
    bits = to_bin(secret_message)

    new_frames = []
    bit_idx = 0

    for frame in frames:
        frame = frame.astype(np.uint8)
        if bit_idx < len(bits):
            for row in frame:
                for pixel in row:
                    for i in range(3):  # Modify R, G, and B channels
                        if bit_idx < len(bits):
                            pixel[i] = (pixel[i] & (255 - (1 << lsb_count) + 1)) | int(
                                bits[bit_idx:bit_idx + lsb_count], 2)
                            bit_idx += lsb_count
                        else:
                            break
        new_frames.append(frame)

    # Create a new video from the modified frames
    new_video = ImageSequenceClip(new_frames, fps=video.fps)
    new_video.write_videofile(output_path, codec='libx264')

def extract_data_from_video(video_path, lsb_count):
    """Extracts hidden data from the frames of a video file."""
    video = VideoFileClip(video_path)
    frames = video.iter_frames()

    extracted_bits = ""
    for frame in frames:
        for row in frame:
            for pixel in row:
                for i in range(3):  # Extract from R, G, and B channels
                    extracted_bits += format(pixel[i] & ((1 << lsb_count) - 1), f'0{lsb_count}b')

    message_bytes = [extracted_bits[i:i + 8] for i in range(0, len(extracted_bits), 8)]
    decoded_message = ''.join([chr(int(byte, 2)) for byte in message_bytes])
    return decoded_message.split("#####")[0]

### GUI Implementation ###
class SteganographyApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Steganography - Image, Audio, and Video")
        self.geometry("700x600")
        self.configure(bg='lightgrey')

        # Labels
        self.label_drag = Label(self, text="Drag and drop your media file here (Image, Audio, Video):", bg='lightgrey')
        self.label_drag.pack(pady=20)

        # Drop target for media files
        self.drop_target = Label(self, text="Drop Here", bg="white", width=60, height=5, relief="solid")
        self.drop_target.pack(pady=20)
        self.drop_target.drop_target_register(DND_FILES)
        self.drop_target.dnd_bind('<<Drop>>', self.handle_drop)

        # File info label
        self.file_info_label = Label(self, text="", bg='lightgrey', fg='blue')
        self.file_info_label.pack(pady=10)

        # Input fields
        self.message_label = Label(self, text="Enter message to hide:", bg='lightgrey')
        self.message_label.pack()
        self.text_entry = Entry(self, width=50)
        self.text_entry.pack()

        # LSB slider - updated to allow up to 8 LSBs
        self.lsb_label = Label(self, text="Select number of LSBs to use:", bg='lightgrey')
        self.lsb_label.pack()
        self.lsb_slider = tk.Scale(self, from_=1, to=8, orient=tk.HORIZONTAL)  # Changed from 4 to 8
        self.lsb_slider.pack()

        # Buttons
        self.hide_button = Button(self, text="Hide Message", command=self.hide_message)
        self.hide_button.pack(pady=10)
        self.extract_button = Button(self, text="Extract Message", command=self.extract_message)
        self.extract_button.pack(pady=10)

        # Explorer Button
        self.browse_button = Button(self, text="Browse File", command=self.open_file_dialog)
        self.browse_button.pack(pady=10)

        # Remove file Button
        self.remove_button = Button(self, text="Remove File", command=self.remove_file)
        self.remove_button.pack(pady=10)

        self.media_path = None
        self.media_type = None  # Could be 'image', 'audio', or 'video'

    def handle_drop(self, event):
        """Handles drag-and-drop file."""
        self.media_path = event.data.strip('{}')
        self.check_file_type()

    def open_file_dialog(self):
        """Opens a file dialog to browse media files."""
        # Updated to allow different file types including image, audio, and video
        self.media_path = filedialog.askopenfilename(filetypes=[
            ("All Supported Media Files", "*.png;*.bmp;*.jpg;*.jpeg;*.gif;*.wav;*.mp3;*.mp4"),
            ("Image Files", "*.png;*.bmp;*.jpg;*.jpeg;*.gif"),
            ("Audio Files", "*.wav;*.mp3"),
            ("Video Files", "*.mp4")])
        self.check_file_type()

    def check_file_type(self):
        """Checks the file type and updates the media type."""
        if self.media_path.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg', '.gif')):
            self.media_type = 'image'
            self.file_info_label.config(text=f"Selected File: {os.path.basename(self.media_path)} (Image)")
        elif self.media_path.lower().endswith(('.wav', '.mp3')):
            self.media_type = 'audio'
            self.file_info_label.config(text=f"Selected File: {os.path.basename(self.media_path)} (Audio)")
        elif self.media_path.lower().endswith('.mp4'):
            self.media_type = 'video'
            self.file_info_label.config(text=f"Selected File: {os.path.basename(self.media_path)} (Video)")
        else:
            messagebox.showerror("Unsupported file", "File type not supported. Please use PNG, BMP, JPG, GIF, WAV, MP3, or MP4.")
            self.media_type = None

    def remove_file(self):
        """Removes the selected file."""
        self.media_path = None
        self.media_type = None
        self.file_info_label.config(text="")
        messagebox.showinfo("File Removed", "Media file has been removed.")

    def hide_message(self):
        """Handles hiding the secret message in the selected media file."""
        if not self.media_path or not self.media_type:
            messagebox.showerror("Error", "Please drop or select a media file first.")
            return

        secret_message = self.text_entry.get()
        if not secret_message:
            messagebox.showerror("Error", "Please enter a message to hide.")
            return

        lsb_count = self.lsb_slider.get()

        # Warn user if more than 4 LSBs are selected
        if lsb_count > 4:
            warning = messagebox.askyesno("Warning", f"Using {lsb_count} LSBs might degrade the quality of the media. Do you want to proceed?")
            if not warning:
                return

        try:
            if self.media_type == 'image':
                image = cv2.imread(self.media_path)
                stego_image = hide_data_in_image(image, secret_message, lsb_count)
                save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                         filetypes=[("PNG Files", "*.png")])
                if save_path:
                    cv2.imwrite(save_path, stego_image)
                    messagebox.showinfo("Success",
                                        f"Message hidden in image using {lsb_count} LSB(s). Saved to {save_path}")
            elif self.media_type == 'audio':
                save_path = filedialog.asksaveasfilename(defaultextension=".wav",
                                                         filetypes=[("WAV Files", "*.wav")])
                if save_path:
                    hide_data_in_audio(self.media_path, secret_message, save_path, lsb_count)
                    messagebox.showinfo("Success",
                                        f"Message hidden in audio using {lsb_count} LSB(s). Saved to {save_path}")
            elif self.media_type == 'video':
                save_path = filedialog.asksaveasfilename(defaultextension=".mp4",
                                                         filetypes=[("MP4 Files", "*.mp4")])
                if save_path:
                    hide_data_in_video(self.media_path, secret_message, save_path, lsb_count)
                    messagebox.showinfo("Success",
                                        f"Message hidden in video using {lsb_count} LSB(s). Saved to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def extract_message(self):
        """Handles extracting the secret message from the selected media file."""
        if not self.media_path or not self.media_type:
            messagebox.showerror("Error", "Please drop or select a media file first.")
            return

        lsb_count = self.lsb_slider.get()

        try:
            if self.media_type == 'image':
                image = cv2.imread(self.media_path)
                hidden_message = extract_data_from_image(image, lsb_count)
            elif self.media_type == 'audio':
                hidden_message = extract_data_from_audio(self.media_path, lsb_count)
            elif self.media_type == 'video':
                hidden_message = extract_data_from_video(self.media_path, lsb_count)

            messagebox.showinfo("Extracted Message", f"Hidden message: {hidden_message}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Run the application
if __name__ == "__main__":
    app = SteganographyApp()
    app.mainloop()
