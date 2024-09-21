import tkinter as tk
from tkinter import filedialog, messagebox, Label, Entry, Button
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
import numpy as np
import cv2
import wave
from moviepy.editor import VideoFileClip


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
    n_bytes = image.size * 3 // 8  # Calculate maximum bytes
    if len(secret_message) > n_bytes:
        raise ValueError("Insufficient bytes, need a larger image or less data.")

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
    frame_bytes = bytearray(list(song.readframes(song.getnframes())))

    secret_message += "#####"  # delimiter
    bits = to_bin(secret_message)

    for i in range(0, len(bits), lsb_count):
        frame_bytes[i] = (frame_bytes[i] & (255 - (1 << lsb_count) + 1)) | int(bits[i:i + lsb_count], 2)

    frame_modified = bytes(frame_bytes)
    with wave.open(output_path, 'wb') as fd:
        fd.setparams(song.getparams())
        fd.writeframes(frame_modified)
    song.close()


def extract_data_from_audio(audio_path, lsb_count):
    """Extracts hidden data from an audio file."""
    song = wave.open(audio_path, mode='rb')
    frame_bytes = bytearray(list(song.readframes(song.getnframes())))

    extracted_bits = [format(frame_bytes[i] & ((1 << lsb_count) - 1), f'0{lsb_count}b') for i in
                      range(len(frame_bytes))]
    extracted_bits = ''.join(extracted_bits)

    message_bytes = [extracted_bits[i:i + 8] for i in range(0, len(extracted_bits), 8)]
    decoded_message = ''.join([chr(int(byte, 2)) for byte in message_bytes])
    return decoded_message.split("#####")[0]


### VIDEO STEGANOGRAPHY ###
def hide_data_in_video(video_path, secret_message, output_path, lsb_count):
    """Hides secret data inside a video file using LSB."""
    video = VideoFileClip(video_path)
    frames = video.iter_frames()

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

    new_video = video.set_duration(video.duration).set_fps(video.fps).with_frames(new_frames)
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
        self.geometry("700x550")
        self.configure(bg='lightgrey')

        # Labels
        self.label_drag = Label(self, text="Drag and drop your media file here (Image, Audio, Video):", bg='lightgrey')
        self.label_drag.pack(pady=20)

        # Drop target for media files
        self.drop_target = Label(self, text="Drop Here", bg="white", width=60, height=5, relief="solid")
        self.drop_target.pack(pady=20)
        self.drop_target.drop_target_register(DND_FILES)
        self.drop_target.dnd_bind('<<Drop>>', self.handle_drop)

        # Input fields
        self.message_label = Label(self, text="Enter message to hide:", bg='lightgrey')
        self.message_label.pack()
        self.text_entry = Entry(self, width=50)
        self.text_entry.pack()

        # LSB slider
        self.lsb_label = Label(self, text="Select number of LSBs to use:", bg='lightgrey')
        self.lsb_label.pack()
        self.lsb_slider = tk.Scale(self, from_=1, to=8, orient=tk.HORIZONTAL)
        self.lsb_slider.pack()

        # Buttons
        self.hide_button = Button(self, text="Hide Message", command=self.hide_message)
        self.hide_button.pack(pady=10)
        self.extract_button = Button(self, text="Extract Message", command=self.extract_message)
        self.extract_button.pack(pady=10)

        self.media_path = None
        self.media_type = None  # Could be 'image', 'audio', or 'video'

    def handle_drop(self, event):
        """Handles drag-and-drop file."""
        self.media_path = event.data.strip('{}')
        if self.media_path.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg')):
            self.media_type = 'image'
            self.drop_target.config(text="Image file selected")
        elif self.media_path.lower().endswith('.wav'):
            self.media_type = 'audio'
            self.drop_target.config(text="Audio file selected")
        elif self.media_path.lower().endswith('.mp4'):
            self.media_type = 'video'
            self.drop_target.config(text="Video file selected")
        else:
            messagebox.showerror("Unsupported file", "File type not supported. Please use PNG, BMP, JPG, WAV, or MP4.")

    def hide_message(self):
        if not self.media_path or not self.media_type:
            messagebox.showerror("Error", "Please drop a media file first")
            return

        secret_message = self.text_entry.get()
        if not secret_message:
            messagebox.showerror("Error", "Please enter a message to hide")
            return

        lsb_count = self.lsb_slider.get()

        if self.media_type == 'image':
            image = cv2.imread(self.media_path)
            stego_image = hide_data_in_image(image, secret_message, lsb_count)
            cv2.imwrite("C:/Users/haziq/Downloads/stego_image.png", stego_image)
            messagebox.showinfo("Success",
                                f"Message hidden in image using {lsb_count} LSB(s). Saved to C:/Users/haziq/Downloads/stego_image.png")
        elif self.media_type == 'audio':
            hide_data_in_audio(self.media_path, secret_message, 'C:/Users/haziq/Downloads/stego_audio.wav', lsb_count)
            messagebox.showinfo("Success",
                                f"Message hidden in audio using {lsb_count} LSB(s). Saved to C:/Users/haziq/Downloads/stego_audio.wav")
        elif self.media_type == 'video':
            hide_data_in_video(self.media_path, secret_message, 'C:/Users/haziq/Downloads/stego_video.mp4', lsb_count)
            messagebox.showinfo("Success",
                                f"Message hidden in video using {lsb_count} LSB(s). Saved to C:/Users/haziq/Downloads/stego_video.mp4")

    def extract_message(self):
        if not self.media_path or not self.media_type:
            messagebox.showerror("Error", "Please drop a media file first")
            return

        lsb_count = self.lsb_slider.get()

        if self.media_type == 'image':
            image = cv2.imread(self.media_path)
            hidden_message = extract_data_from_image(image, lsb_count)
        elif self.media_type == 'audio':
            hidden_message = extract_data_from_audio(self.media_path, lsb_count)
        elif self.media_type == 'video':
            hidden_message = extract_data_from_video(self.media_path, lsb_count)

        messagebox.showinfo("Extracted Message", f"Hidden message: {hidden_message}")


# Run the application
if __name__ == "__main__":
    app = SteganographyApp()
    app.mainloop()
