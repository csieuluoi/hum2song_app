import torch
import numpy as np
from torchvision import transforms as T
from sklearn.preprocessing import normalize
import os
import yaml

import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir) 

cwd = os.getcwd()
print(cwd)

import librosa
import audio as Audio
from pydub import AudioSegment

config = yaml.load(open("./src/config/preprocess.yaml", "r"), Loader=yaml.FullLoader)

sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
val_size = config["preprocessing"]["val_size"]

STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )


def load_audio(path, sr):
    # Load the audio file using pydub
    # audio = AudioSegment.from_file(path, format="mp3")

    if path.endswith('.mp3') or path.endswith('.MP3'):
        audio = AudioSegment.from_mp3(path)
    elif path.endswith('.wav') or path.endswith('.WAV'):
        audio = AudioSegment.from_wav(path)
    elif path.endswith('.ogg'):
        audio = AudioSegment.from_ogg(path)
    elif path.endswith('.flac'):
        audio = AudioSegment.from_file(path, "flac")
    elif path.endswith('.3gp'):
        audio = AudioSegment.from_file(path, "3gp")
    elif path.endswith('.3g'):
        audio = AudioSegment.from_file(path, "3gp")

    # Convert the audio to mono if necessary
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Resample the audio if necessary
    if audio.frame_rate != sr:
        audio = audio.set_frame_rate(sr)

    # Extract the raw audio data
    data = audio.get_array_of_samples()

    # Convert the audio data to a numpy array
    data = np.array(data)

    # Normalize the audio
    if audio.sample_width == 2:
        data = data.astype(np.float32)
        data /= np.iinfo(np.int16).max

    return data, sr

def process(audio, max_wav_value, STFT):
    audio = audio.astype(np.float32)
    audio = audio / max(abs(audio)) * max_wav_value
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(audio, STFT)
    return mel_spectrogram.T

def process_file(audio_path, sampling_rate, max_wav_value, STFT, full_song = False, input_shape=(630, 80)):
    audio, _ = librosa.load(audio_path, sr=sampling_rate)
    # audio,_ = load_audio(audio_path, sr=sampling_rate)
    spec = process(audio, max_wav_value, STFT)
    if not full_song:
        spec = get_image(spec, input_shape=input_shape)
    return spec
    
def get_images(data, input_shape = (630, 80), overlap_size = 0.5):
    results = []
    start = int(len(data) * 0.2)
    end = int(len(data) * 0.8)
    data = data[start: end]
    if data.shape[0] < input_shape[0]:
        result = np.zeros(input_shape)
        result[:data.shape[0], :data.shape[1]] = data
        results.append(result)
    else:
        start = 0
        end = input_shape[0]
        while(end < data.shape[0]):
            results.append(data[start:end])
            start += int(input_shape[0]*(1-overlap_size))
            end += int(input_shape[0]*(1-overlap_size))
        
    images = torch.from_numpy(np.array(results)).unsqueeze(1)

    return images.float()


def get_image(data, input_shape=(630, 80)):
    if data.shape[0] >= input_shape[0]:
        result = data[:input_shape[0], :]
    else:
        result = np.zeros(input_shape)
        result[:data.shape[0], :data.shape[1]] = data
    image = torch.from_numpy(result).unsqueeze(0).unsqueeze(0)
    return image.float()

def load_images(npy_path, input_shape = (630, 80), overlap_size = 0.5):
    data = np.load(npy_path)

    results = []
    
    if data.shape[0] < input_shape[0]:
        result = np.zeros(input_shape)
        result[:data.shape[0], :data.shape[1]] = data
        results.append(result)
    else:
        start = 0
        end = input_shape[0]
        while(end < data.shape[0]):
            results.append(data[start:end])
            shift_t = int(input_shape[0]*overlap_size)
            start += shift_t
            end += shift_t
        
    images = torch.from_numpy(np.array(results)).unsqueeze(1)

    return images.float()

def load_image(npy_path, input_shape=(630, 80)):
    data = np.load(npy_path)
    if data.shape[0] >= input_shape[0]:
        result = data[:input_shape[0], :]
    else:
        result = np.zeros(input_shape)
        result[:data.shape[0], :data.shape[1]] = data
    image = torch.from_numpy(result).unsqueeze(0).unsqueeze(0)
    return image.float()


def get_feature(model, image):
    data = image.to(torch.device("cuda"))
    with torch.no_grad():
        output = model(data)
    output = output.cpu().detach().numpy()
    output = normalize(output).flatten()
    return np.matrix(output)


# def writeFile(fileName, content):
#     with open(fileName, 'a') as f1:
#         f1.write(content + os.linesep)
