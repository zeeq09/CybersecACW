import os
import struct
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button, Frame
from tqdm import tqdm
import cv2
from tkinterdnd2 import TkinterDnD, DND_FILES
import numpy as np
from PIL import Image
import wave
import threading
import math


def encode_image(image_path, payload_text, output_path, num_lsb):
    image = Image.open(image_path)
    # Convert the image to a NumPy array with int64 data type to allow more bit spacesss
    img_data = np.array(image, dtype=np.int64)
    with open(payload_text, 'r') as file:
        payload = file.read()

    # Convert payload to binary and add delimiter to mark the end of the binary payload!
    payload_bin = ''.join(format(ord(char), '08b') for char in payload) + '1111111111111110'
    # Calculate maximum bytes to encode
    max_bytes = img_data.size * num_lsb // 8
    if len(payload_bin) > max_bytes:
        messagebox.showerror("Error", "Payload too large for the selected file.")
    else:
        # THis the start of the encoding process
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

        # Convert back to uint8 before saving, since images use 8-bit values
        stego_image = Image.fromarray(img_data.astype(np.uint8))
        stego_image.save(output_path)
    return stego_image


def decode_image(image_path, num_lsb):
    image = Image.open(image_path)
    img_data = np.array(image)

    payload_bin = ""
    total_pixels = img_data.shape[0] * img_data.shape[1]

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


def encode_audio(audio_path, payload_text, output_path, lsb_count):
    with open(payload_text, 'r') as file:
        payload = file.read()

    # STart of encoding process
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
        messagebox.showerror("Error", "Payload too large for the selected file.")
    else:
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


def encode_video(video_path, payload_text, output_path, num_lsb):
    video = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Use FFV1 codec as its a lossless encoder!
    out = cv2.VideoWriter(output_path, fourcc, video.get(cv2.CAP_PROP_FPS),
                          (int(video.get(3)), int(video.get(4))))

    with open(payload_text, 'r') as file:
        payload = file.read()

    # Convert the payload to binary (8 bits per character)
    binary_payload = ''.join([format(ord(char), '08b') for char in payload])
    binary_payload += '1111111111111110'  # End marker

    data_index = 0  # Track the position in the binary payload
    total_bits = len(binary_payload)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate maximum capacity
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_pixels_per_frame = frame_width * frame_height * 3  # 3 channels (B, G, R)
    bits_per_frame = total_pixels_per_frame * num_lsb
    max_capacity = total_frames * bits_per_frame

    if total_bits > max_capacity:
        messagebox.showerror("Error", "Payload too large for the selected video file.")
    else:
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
                        for bit_pos in range(num_lsb):
                            if data_index < total_bits:
                                mask = ~(1 << bit_pos) & 0xFF  # Ensure mask is within 0-255
                                bit = int(binary_payload[data_index])
                                flat_frame[i] = (flat_frame[i] & mask) | (bit << bit_pos)
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


def decode_video(video_path, num_lsb):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_size = 10
    binary_message = []
    end_marker = '1111111111111110'  # End marker
    bit_buffer = ""

    # Progress bar with the total number of frames
    with tqdm(total=total_frames, desc="Decoding video", unit="frame") as pbar:
        while video.isOpened():
            frames = []
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
                    for bit_pos in range(num_lsb):
                        bit = (pixel >> bit_pos) & 1
                        bit_buffer += str(bit)

                        if len(bit_buffer) == 8:
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


def encode_video_with_file(video_path, payload_file, output_path, num_lsb):
    video = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(output_path, fourcc, video.get(cv2.CAP_PROP_FPS),
                          (int(video.get(3)), int(video.get(4))))

    # Read the payload file as binary
    with open(payload_file, 'rb') as file:
        payload_data = file.read()

    # Get the payload file extension (e.g., '.pdf')
    payload_extension = os.path.splitext(payload_file)[1]
    payload_extension_bytes = payload_extension.encode('utf-8')
    payload_extension_size = len(payload_extension_bytes)

    # Convert the extension size to 8 bits
    extension_size_bits = format(payload_extension_size, '08b')
    # Convert the extension bytes to bits
    extension_bits = ''.join([format(byte, '08b') for byte in payload_extension_bytes])
    # Get the size of the payload
    payload_size = len(payload_data)
    # Convert the size to 64 bits (8 bytes)
    payload_size_bits = format(payload_size, '064b')
    # Convert the payload data to bits
    payload_bits = ''.join([format(byte, '08b') for byte in payload_data])
    # Combine the extension size, extension bits, payload size, and payload bits
    binary_payload = extension_size_bits + extension_bits + payload_size_bits + payload_bits

    data_index = 0
    total_bits = len(binary_payload)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate maximum capacity
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_pixels_per_frame = frame_width * frame_height * 3  # 3 channels (B, G, R)
    bits_per_frame = total_pixels_per_frame * num_lsb
    max_capacity = total_frames * bits_per_frame

    if total_bits > max_capacity:
        raise ValueError("Payload too large for the selected video file.")
    else:
        with tqdm(total=total_frames, desc="Encoding video", unit="frame") as pbar:
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break

                flat_frame = frame.flatten()

                for i in range(len(flat_frame)):
                    if data_index < total_bits:
                        for bit_pos in range(num_lsb):
                            if data_index < total_bits:
                                mask = ~(1 << bit_pos) & 0xFF
                                bit = int(binary_payload[data_index])
                                flat_frame[i] = (flat_frame[i] & mask) | (bit << bit_pos)
                                data_index += 1
                            else:
                                break
                    else:
                        break

                modified_frame = flat_frame.reshape(frame.shape)
                out.write(modified_frame)

                pbar.update(1)

        video.release()
        out.release()
    return output_path


def decode_video_with_file(stego_video_path, num_lsb):
    video = cv2.VideoCapture(stego_video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_pixels_per_frame = frame_width * frame_height * 3
    bits_per_frame = total_pixels_per_frame * num_lsb

    binary_data = ''
    extension_size = None
    payload_extension = ''
    payload_size = None
    total_bits_needed = None

    with tqdm(total=total_frames, desc="Decoding video", unit="frame") as pbar:
        frame_count = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            flat_frame = frame.flatten()

            for i in range(len(flat_frame)):
                for bit_pos in range(num_lsb):
                    bit = (flat_frame[i] >> bit_pos) & 1
                    binary_data += str(bit)

                    # Read extension size (first 8 bits)
                    if extension_size is None and len(binary_data) == 8:
                        extension_size = int(binary_data[:8], 2)
                        print(f"Extension size: {extension_size} bytes")

                    # Read extension bits
                    elif extension_size is not None and len(binary_data) == 8 + (extension_size * 8):
                        extension_bits = binary_data[8:8 + (extension_size * 8)]
                        for j in range(0, len(extension_bits), 8):
                            byte = extension_bits[j:j + 8]
                            payload_extension += chr(int(byte, 2))
                        print(f"Payload extension: {payload_extension}")

                    # Read payload size (next 64 bits after extension)
                    elif payload_size is None and extension_size is not None and len(binary_data) == 8 + (
                            extension_size * 8) + 64:
                        payload_size = int(binary_data[8 + (extension_size * 8):8 + (extension_size * 8) + 64], 2)
                        total_payload_bits = payload_size * 8
                        total_bits_needed = 8 + (extension_size * 8) + 64 + total_payload_bits
                        print(f"Payload size: {payload_size} bytes")
                        # Check if total_bits_needed is realistic
                        max_capacity = (total_frames * bits_per_frame)
                        if total_bits_needed > max_capacity:
                            video.release()
                            raise ValueError("The payload size exceeds the capacity of the stego video.")

                    # Break out once we've read all the bits we need
                    if total_bits_needed and len(binary_data) >= total_bits_needed:
                        break
                else:
                    continue  # Continue if inner loop wasn't broken
                break  # Break outer loop if inner loop was broken

            if total_bits_needed and len(binary_data) >= total_bits_needed:
                break

            frame_count += 1
            pbar.update(1)

    video.release()

    if payload_size is None or not payload_extension:
        video.release()
        raise ValueError("Failed to extract payload size or extension from the video.")

    # Extract the payload data
    start_index = 8 + (extension_size * 8) + 64
    payload_data_bits = binary_data[start_index:start_index + payload_size * 8]
    payload_bytes = bytearray()
    for i in range(0, len(payload_data_bits), 8):
        byte = payload_data_bits[i:i + 8]
        payload_bytes.append(int(byte, 2))

    # Save the extracted file with the correct extension
    output_file = stego_video_path + "_extracted" + payload_extension
    with open(output_file, 'wb') as f:
        f.write(payload_bytes)

    print(f"Extraction complete. File saved as {output_file}")
    return output_file


def image_to_image_encode(cover_image_path, hidden_image_path, output_image_path, num_lsb):
    cover_image = Image.open(cover_image_path)
    hidden_image = Image.open(hidden_image_path)

    # Resize the hidden image to fit the cover image if necessary
    if cover_image.size != hidden_image.size:
        hidden_image = hidden_image.resize(cover_image.size)

    cover_pixels = np.array(cover_image)
    hidden_pixels = np.array(hidden_image)

    # Iterate over each pixel of the image
    for i in range(hidden_pixels.shape[0]):
        for j in range(hidden_pixels.shape[1]):
            for channel in range(3):  # For R, G, B channels
                cover_pixel = cover_pixels[i][j][channel]
                hidden_pixel = hidden_pixels[i][j][channel]

                # Get the most significant bits from the hidden pixel
                hidden_bits = format(hidden_pixel, '08b')[:num_lsb]

                # Get the least significant bits from the cover pixel
                cover_bits = format(cover_pixel, '08b')[:8 - num_lsb]

                # Combine them to form the new pixel value
                new_pixel_bits = cover_bits + hidden_bits
                cover_pixels[i][j][channel] = int(new_pixel_bits, 2)

    # Save the stego image
    stego_image = Image.fromarray(cover_pixels)
    stego_image.save(output_image_path)
    return stego_image


def image_to_image_decode(stego_image_path, num_lsb):
    stego_image = Image.open(stego_image_path)
    stego_pixels = np.array(stego_image)

    # Create an empty array for the hidden image
    hidden_pixels = np.zeros_like(stego_pixels)

    # Iterate over each pixel to extract the hidden image
    for i in range(stego_pixels.shape[0]):
        for j in range(stego_pixels.shape[1]):
            for channel in range(3):  # For R, G, B channels
                stego_pixel = stego_pixels[i][j][channel]

                # Get the least significant bits from the stego pixel
                hidden_bits = format(stego_pixel, '08b')[-num_lsb:]

                # Pad the rest with zeros to form the hidden pixel
                hidden_pixel_bits = hidden_bits + '0' * (8 - num_lsb)
                hidden_pixels[i][j][channel] = int(hidden_pixel_bits, 2)

    # Save and return the decoded hidden image
    hidden_image = Image.fromarray(hidden_pixels)
    hidden_image.show()
    return hidden_image


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(tw, text=self.text, justify='center',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("Verdana", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        tw = self.tooltip_window
        self.tooltip_window = None
        if tw:
            tw.destroy()


class SteganographyApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Steganography Tool")
        self.geometry("720x480")

        # Dark Gray for the background
        self.configure(bg='#4B4B4B')

        # Header
        self.header_label = Label(self, text="Steganography Tool", font=("Verdana", 18, "bold"), bg='#4B4B4B',
                                  fg='white')
        self.header_label.pack(pady=20)

        # Drag-and-Drop Frame for Media File
        self.media_frame = Frame(self, bg='#4B4B4B', width=600, height=50, relief="sunken")
        self.media_frame.pack(pady=5)
        self.media_frame.pack_propagate(False)

        self.media_label = Label(self.media_frame, text="Drag & Drop Media File Here", font=("Verdana", 12),
                                 bg='#D3D3D3', fg='black')
        self.media_label.pack(fill=tk.BOTH, expand=True)

        self.media_frame.drop_target_register(DND_FILES)
        self.media_frame.dnd_bind('<<Drop>>', self.on_media_drop)

        ToolTip(self.media_label, "Drag your media file here (PNG, WAV, MP4)")

        self.media_remove_button = Button(self, text="Remove Media", command=self.remove_media, bg='#008080',
                                          font=("Verdana", 8), fg='white')
        self.media_remove_button.pack(pady=5)

        # Drag-and-Drop Frame for Payload File
        self.payload_frame = Frame(self, bg='#4B4B4B', width=600, height=50, relief="sunken")
        self.payload_frame.pack(pady=5)
        self.payload_frame.pack_propagate(False)

        self.payload_label = Label(self.payload_frame, text="Drag & Drop Payload File Here",
                                   font=("Verdana", 12), bg='#D3D3D3', fg='black')
        self.payload_label.pack(fill=tk.BOTH, expand=True)

        ToolTip(self.payload_label, "Drag your payload file here")

        self.payload_frame.drop_target_register(DND_FILES)
        self.payload_frame.dnd_bind('<<Drop>>', self.on_payload_drop)

        self.payload_remove_button = Button(self, text="Remove Payload", command=self.remove_payload, bg='#008080',
                                            font=("Verdana", 8), fg='white')
        self.payload_remove_button.pack(pady=5)

        # Initialize variables to hold file paths
        self.media_file = ""
        self.payload_file = ""

        # Frame for user input (message and LSBs)
        input_frame = Frame(self, bg='#4B4B4B')
        input_frame.pack(pady=20)

        # LSB slider
        self.lsb_label = Label(input_frame, text="Select number of LSBs to use:", font=("Verdana", 12), bg='#4B4B4B',
                               fg='white')
        self.lsb_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.lsb_slider = tk.Scale(input_frame, from_=1, to=8, orient=tk.HORIZONTAL, length=200, font=("Helvetica", 12),
                                   bg='#4B4B4B', fg='white', troughcolor='#008080', relief="flat")
        self.lsb_slider.set(1)
        self.lsb_slider.grid(row=0, column=1, padx=10, pady=10)
        ToolTip(self.lsb_slider, "Set the number of Least Significant Bits (LSBs) to use for steganography.")

        # Frame for buttons
        button_frame = Frame(self, bg='#4B4B4B')
        button_frame.pack(pady=20)

        # Buttons for hiding and extracting
        self.hide_button = Button(button_frame, text="Hide Message", command=self.threaded_hide_message,
                                  font=("Verdana", 12), bg='#008080', fg='white', relief="raised")
        self.hide_button.grid(row=0, column=0, padx=10)
        self.hide_button.bind("<Enter>", lambda event: self.hide_button.config(bg='#007070'))  # On hover
        self.hide_button.bind("<Leave>", lambda event: self.hide_button.config(bg='#008080'))  # On leave

        self.extract_button = Button(button_frame, text="Extract Message", command=self.threaded_extract_message,
                                     font=("Verdana", 12), bg='#008080', fg='white', relief="raised")
        self.extract_button.grid(row=0, column=1, padx=10)
        self.extract_button.bind("<Enter>", lambda event: self.extract_button.config(bg='#007070'))  # On hover
        self.extract_button.bind("<Leave>", lambda event: self.extract_button.config(bg='#008080'))  # On leave

    def on_media_drop(self, event):
        self.media_file = event.data.strip()  # Strip any extra spaces or newlines

        # In case the file path is enclosed in curly braces, remove them
        if self.media_file.startswith('{') and self.media_file.endswith('}'):
            self.media_file = self.media_file[1:-1]

        # Update the label to display the media file name or path
        self.media_label.config(text="File Loaded: " + os.path.basename(self.media_file))

    def on_payload_drop(self, event):
        self.payload_file = event.data.strip()
        if self.payload_file.startswith('{') and self.payload_file.endswith('}'):
            self.payload_file = self.payload_file[1:-1]

        # Allow various files
        if self.payload_file.lower().endswith(('.txt', '.png', '.jpg', '.jpeg', '.pdf', '.docx', '.xlsx')):
            self.payload_label.config(text="Payload Loaded: " + os.path.basename(self.payload_file))
        else:
            messagebox.showerror("Error", "Please drop a valid text, image, or document file.")

    def remove_media(self):
        """Clear the media file and reset the label."""
        self.media_file = ""
        self.media_label.config(text="Drag & Drop Media File Here")

    def remove_payload(self):
        """Clear the payload file and reset the label."""
        self.payload_file = ""
        self.payload_label.config(text="Drag & Drop Payload File Here")

    def threaded_hide_message(self):
        threading.Thread(target=self.hide_message).start()

    def hide_message(self):
        media_file = self.media_file
        payload_message = self.payload_file
        lsb_count = self.lsb_slider.get()
        output_path = media_file.split('.')[0] + '_output.' + media_file.split('.')[-1]

        try:
            if payload_message.lower().endswith('.txt'):
                if media_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    encode_image(media_file, payload_message, output_path, lsb_count)
                    messagebox.showinfo("Success", f"Text file hidden successfully in {output_path}")
                elif media_file.lower().endswith(('.mp3', '.wav')):
                    encode_audio(media_file, payload_message, output_path, lsb_count)
                    messagebox.showinfo("Success", f"Text file hidden successfully in {output_path}")
                elif media_file.lower().endswith('.avi'):
                    encode_video(media_file, payload_message, output_path, lsb_count)
                    messagebox.showinfo("Success", f"Text file hidden successfully in {output_path}")
                else:
                    messagebox.showerror("Error", "Unsupported media file format for text payload.")
            elif payload_message.lower().endswith(('.png', '.jpg', '.jpeg')):
                if media_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_to_image_encode(media_file, payload_message, output_path, lsb_count)
                    messagebox.showinfo("Success", f"Image file hidden successfully in {output_path}")
                else:
                    messagebox.showerror("Error", "Image payload can only be hidden inside image media files.")
            elif payload_message.lower().endswith(('.pdf', '.docx', '.xlsx')):
                if media_file.lower().endswith('.avi'):
                    encode_video_with_file(media_file, payload_message, output_path, lsb_count)
                    messagebox.showinfo("Success", f"File hidden successfully in {output_path}")
                else:
                    messagebox.showerror("Error", "Document payload can only be hidden inside video media files.")
            else:
                messagebox.showerror("Error", "Unsupported payload file format.")
        except Exception as e:
            messagebox.showerror("Error", f"Hiding failed: Try a another file")

    def threaded_extract_message(self):
        threading.Thread(target=self.extract_message).start()

    def extract_message(self):
        media_file = self.media_file
        lsb_count = self.lsb_slider.get()

        if media_file.lower().endswith(('.mp4', '.avi', '.mov')):
            try:
                # First, try to extract as a file
                extracted_file = decode_video_with_file(media_file, lsb_count)
                if os.path.exists(extracted_file):
                    messagebox.showinfo("Success", f"File extracted successfully to {extracted_file}")
                else:
                    raise ValueError("No file extracted.")
            except ValueError as ve:
                # If extraction as a file fails, try to extract as a text message
                try:
                    hidden_message = decode_video(media_file, lsb_count)
                    if hidden_message:
                        messagebox.showinfo("Hidden Message", hidden_message)
                    else:
                        messagebox.showinfo("No Message", "No hidden message found.")
                except Exception as e2:
                    messagebox.showerror("Error", f"Extraction failed: {str(e2)}")
            except Exception as e:
                messagebox.showerror("Error", f"Extraction failed: {str(e)}")
        elif media_file.lower().endswith(('.png', '.jpg', '.jpeg')) and not self.payload_file:
            try:
                hidden_image = image_to_image_decode(media_file, lsb_count)
                if hidden_image:
                    messagebox.showinfo("Success", "Hidden image extracted and displayed.")
                else:
                    messagebox.showinfo("No Image", "No hidden image found.")
            except Exception as e:
                messagebox.showerror("Error", f"Extraction failed: {str(e)}")
        elif media_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                hidden_message = decode_image(media_file, lsb_count)
                if hidden_message:
                    messagebox.showinfo("Hidden Message", hidden_message)
                else:
                    messagebox.showinfo("No Message", "No hidden message found.")
            except Exception as e:
                messagebox.showerror("Error", f"Extraction failed: {str(e)}")
        elif media_file.lower().endswith(('.mp3', '.wav')):
            try:
                hidden_message = decode_audio(media_file, lsb_count)
                if hidden_message:
                    messagebox.showinfo("Hidden Message", hidden_message)
                else:
                    messagebox.showinfo("No Message", "No hidden message found.")
            except Exception as e:
                messagebox.showerror("Error", f"Extraction failed: {str(e)}")
        else:
            messagebox.showerror("Error", "Unsupported media file format")
            return


if __name__ == "__main__":
    app = SteganographyApp()
    app.mainloop()
