import struct
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Entry, Button, Frame, ttk
from tqdm import tqdm
import cv2
from tkinterdnd2 import TkinterDnD, DND_FILES
import numpy as np
from PIL import Image
import wave
import threading


def encode_image(image_path, payload_text, output_path, num_lsb):
    image = Image.open(image_path)
    img_data = np.array(image)
    with open(payload_text, 'r') as file:
        payload = file.read()

    # Convert payload to binary and add delimiter
    payload_bin = ''.join(format(ord(char), '08b') for char in payload) + '1111111111111110'

    # Calculate maximum bytes to encode
    max_bytes = img_data.size * num_lsb // 8
    if len(payload_bin) > max_bytes:
        raise ValueError("Payload size exceeds cover image capacity.")

    data_index = 0
    for row in img_data:
        for pixel in row:
            for channel in range(3):  # R, G, B channels
                if data_index < len(payload_bin):
                    pixel[channel] = (pixel[channel] & ~((1 << num_lsb) - 1)) | int(
                        payload_bin[data_index:data_index + num_lsb], 2)
                    data_index += num_lsb
                else:
                    break

    stego_image = Image.fromarray(img_data)
    stego_image.save(output_path)
    return stego_image


# Function to decode text from an image with delimiter
def decode_image(image_path, num_lsb):
    image = Image.open(image_path)
    img_data = np.array(image)

    payload_bin = ""
    total_pixels = img_data.shape[0] * img_data.shape[1]  # Total number of pixels

    for row in tqdm(img_data, desc="Decoding Image", total=img_data.shape[0]):
        for pixel in row:
            for channel in range(3):  # R, G, B channels
                payload_bin += format(pixel[channel] & ((1 << num_lsb) - 1), '0' + str(num_lsb) + 'b')

    # Convert binary to ASCII and check for delimiter
    payload = ""
    for i in range(0, len(payload_bin), 8):
        byte = payload_bin[i:i + 8]
        if byte == '11111111':  # End of data (delimiter)
            break
        payload += chr(int(byte, 2))

    return payload


# Function to encode text into an audio file (MP3 and WAV supported)
def encode_audio(audio_path, payload_text, output_path, lsb_count):
    with open(payload_text, 'r') as file:
        payload = file.read()

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
    binary_payload = ''.join([format(ord(char), '08b') for char in payload])
    binary_payload += '1111111111111110'  # End marker

    max_bits_to_hide = total_samples * lsb_count
    if len(binary_payload) > max_bits_to_hide:
        raise ValueError("Payload too large for the selected audio file.")

    # Modify the LSBs of audio samples
    data_index = 0
    for i in range(len(samples)):
        if data_index + lsb_count <= len(binary_payload):
            # Modify the LSBs
            sample = samples[i]
            # Clear the least significant bits
            sample &= ~((1 << lsb_count) - 1)
            # Set the new least significant bits
            bits_to_hide = binary_payload[data_index:data_index + lsb_count]
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


def decode_audio(audio_path, lsb_count):
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

    # Extract the LSBs with a progress bar
    extracted_bits = ""
    for sample in tqdm(samples, desc="Decoding Audio", total=total_samples):
        # Get the least significant bits
        bits = format(sample & ((1 << lsb_count) - 1), f'0{lsb_count}b')
        extracted_bits += bits

    # Convert bits to bytes
    payload = ""
    for i in range(0, len(extracted_bits), 8):
        byte = extracted_bits[i:i + 8]
        if byte == '11111111':  # End of data (delimiter)
            break
        payload += chr(int(byte, 2))

    return payload


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
    flat_frame = frame.flatten()
    num_pixels = len(flat_frame)

    if mode == 'hide':
        # Hide bits in the LSB of the frame
        for i in range(len(bits)):
            if i < num_pixels:
                flat_frame[i] = (flat_frame[i] & ~((1 << lsb_count) - 1)) | int(bits[i], 2)
    elif mode == 'reveal':
        # Reveal bits from the LSB of the frame (not used in encoding)
        pass

    return flat_frame.reshape(frame.shape)


# Function to encode text into an MP4 video file
def encode_video(video_path, payload_text, output_path, num_lsb):
    video = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Use lossless FFV1 codec
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(video.get(3)), int(video.get(4))))

    with open(payload_text, 'r') as file:
        payload = file.read()

    # Convert the payload to binary (8 bits per character)
    binary_payload = ''.join([format(ord(char), '08b') for char in payload])
    binary_payload += '1111111111111110'  # End marker

    print(f"Binary Payload (to encode): {binary_payload}")

    data_index = 0  # Track the position in the binary payload
    total_bits = len(binary_payload)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames

    # Progress bar with the total number of frames
    with tqdm(total=total_frames, desc="Encoding video", unit="frame") as pbar:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            flat_frame = frame.flatten()

            for i in range(len(flat_frame)):
                if data_index < total_bits:
                    # Embed the specified number of LSBs from the payload
                    for bit_pos in range(num_lsb):  # Modify multiple LSBs (based on the number of LSBs specified)
                        if data_index < total_bits:
                            flat_frame[i] = (flat_frame[i] & ~(1 << bit_pos)) | (
                                    int(binary_payload[data_index]) << bit_pos)
                            data_index += 1
                        else:
                            break

            # Reshape the flat frame back to the original shape and write it to the output
            modified_frame = flat_frame.reshape(frame.shape)
            out.write(modified_frame)

            pbar.update(1)  # Update the progress bar for each frame

    video.release()
    out.release()
    return output_path


def extract_bits_from_frame(frame, num_lsb):
    frame_bits = ""
    for row in frame:
        for pixel in row:
            for channel in range(3):
                frame_bits += format(pixel[channel] & ((1 << num_lsb) - 1), '0' + str(num_lsb) + 'b')
    return frame_bits


def decode_video(video_path, num_lsb):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    batch_size = 10
    binary_message = []
    end_marker = '1111111111111110'  # End marker
    bit_buffer = ""

    # Progress bar with the total number of frames
    with tqdm(total=total_frames, desc="Decoding video", unit="frame") as pbar:
        while video.isOpened():
            frames = []  # Collect a batch of frames
            for _ in range(batch_size):
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(frame)

            if not frames:
                break  # Exit if no frames are read

            for frame in frames:
                flat_frame = frame.flatten()

                for pixel in flat_frame:
                    # Extract the specified number of LSBs from each pixel
                    for bit_pos in range(num_lsb):  # Extract multiple LSBs (based on the number of LSBs specified)
                        bit = (pixel >> bit_pos) & 1  # Extract the bit at the current position
                        bit_buffer += str(bit)

                        if len(bit_buffer) == 8:  # Once we have 8 bits, we form a byte
                            binary_message.append(bit_buffer)
                            bit_buffer = ""

                            # Check for the end marker
                            if ''.join(binary_message[-len(end_marker) // 8:]) == end_marker:
                                binary_message = binary_message[:-len(end_marker) // 8]
                                decoded_message = ''.join([chr(int(b, 2)) for b in binary_message])
                                video.release()
                                return decoded_message

            pbar.update(len(frames))  # Update the progress bar for the number of frames processed

    video.release()

    # If no end marker is found, return the partial decoded message
    decoded_message = ''.join([chr(int(b, 2)) for b in binary_message])
    return decoded_message


class SteganographyApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Steganography Tool")
        self.geometry("700x600")
        self.configure(bg='#f0f0f0')

        # Header
        self.header_label = Label(self, text="Steganography Tool", font=("Helvetica", 18, "bold"), bg='#f0f0f0')
        self.header_label.pack(pady=20)

        # Drag-and-Drop Frame for Media File
        self.media_frame = Frame(self, bg='#e0e0e0', width=600, height=50, relief="sunken")
        self.media_frame.pack(pady=5)
        self.media_frame.pack_propagate(False)

        self.media_label = Label(self.media_frame, text="Drag & Drop Media File Here", font=("Helvetica", 12),
                                 bg='#e0e0e0')
        self.media_label.pack(fill=tk.BOTH, expand=True)

        self.media_frame.drop_target_register(DND_FILES)
        self.media_frame.dnd_bind('<<Drop>>', self.on_media_drop)

        # Drag-and-Drop Frame for Payload File
        self.payload_frame = Frame(self, bg='#e0e0e0', width=600, height=50, relief="sunken")
        self.payload_frame.pack(pady=5)
        self.payload_frame.pack_propagate(False)

        self.payload_label = Label(self.payload_frame, text="Drag & Drop Payload File Here (Text File)",
                                   font=("Helvetica", 12), bg='#e0e0e0')
        self.payload_label.pack(fill=tk.BOTH, expand=True)

        self.payload_frame.drop_target_register(DND_FILES)
        self.payload_frame.dnd_bind('<<Drop>>', self.on_payload_drop)

        # Initialize variables to hold file paths
        self.media_file = ""
        self.payload_file = ""

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

        # Frame for buttons
        button_frame = Frame(self, bg='#f0f0f0')
        button_frame.pack(pady=20)

        # Buttons for hiding and extracting
        self.hide_button = Button(button_frame, text="Hide Message", command=self.threaded_hide_message,
                                  font=("Helvetica", 12), bg='#28a745', fg='white', relief="raised")
        self.hide_button.grid(row=0, column=0, padx=10)

        self.extract_button = Button(button_frame, text="Extract Message", command=self.threaded_extract_message,
                                     font=("Helvetica", 12), bg='#dc3545', fg='white', relief="raised")
        self.extract_button.grid(row=0, column=1, padx=10)

    def on_media_drop(self, event):
        self.media_file = event.data  # Store the media file path
        self.media_label.config(text=self.media_file)

    def on_payload_drop(self, event):
        self.payload_file = event.data
        if self.payload_file.lower().endswith('.txt'):
            self.payload_label.config(text="Payload Loaded")
        else:
            messagebox.showerror("Error", "Please drop a valid text file.")

    def threaded_hide_message(self):
        threading.Thread(target=self.hide_message).start()

    def hide_message(self):
        media_file = self.media_file  # Get media file from instance variable
        payload_message = self.payload_file  # Get payload message from instance variable
        lsb_count = self.lsb_slider.get()
        output_path = media_file.split('.')[0] + '_output.' + media_file.split('.')[-1]

        if media_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            encode_image(media_file, payload_message, output_path, lsb_count)
        elif media_file.lower().endswith(('.mp4', '.avi', '.mov')):
            encode_video(media_file, payload_message, output_path, lsb_count)
        elif media_file.lower().endswith(('.mp3', '.wav')):
            encode_audio(media_file, payload_message, output_path, lsb_count)
        else:
            messagebox.showerror("Error", "Unsupported media file format.")
            return

        messagebox.showinfo("Success", "Message hidden successfully in " + output_path)

    def threaded_extract_message(self):
        threading.Thread(target=self.extract_message).start()

    def extract_message(self):
        media_file = self.media_file  # Get media file from instance variable
        lsb_count = self.lsb_slider.get()
        hidden_message = None

        if media_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            hidden_message = decode_image(media_file, lsb_count)
        elif media_file.lower().endswith(('.mp4', '.avi', '.mov')):
            hidden_message = decode_video(media_file, lsb_count)
        elif media_file.lower().endswith(('.mp3', '.wav')):
            hidden_message = decode_audio(media_file, lsb_count)
        else:
            messagebox.showerror("Error", "Unsupported media file format.")
            return

        if hidden_message:
            messagebox.showinfo("Hidden Message", hidden_message)
        else:
            messagebox.showinfo("No Message", "No hidden message found.")

            # Reset the progress bar after completion
            self.progress['value'] = 0


if __name__ == "__main__":
    app = SteganographyApp()
    app.mainloop()
