import torch
from sklearn.preprocessing import normalize
import numpy as np
import yaml

import os
import sys
import inspect
import argparse


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import os, json, faiss
from src.resnet import *
from src.utils import *
from tqdm import tqdm
import time

from tinytag import TinyTag

import pickle 
# import requests
# from bs4 import BeautifulSoup

from googleapiclient.discovery import build
from cachetools import TTLCache
from youtubesearchpython import VideosSearch

def get_video_url(search_text):
    videosSearch = VideosSearch(search_text, limit = 1)
    videosResult = videosSearch.result()
    print(videosResult)
    if len(videosResult['result']) > 0:
        video_url = videosResult['result'][0]["link"]
        thumbnail_url = videosResult['result'][0]["thumbnails"][0]['url']
    else:
        video_url = f"https://www.youtube.com/results?search_query={search_text}"
        thumbnail_url = "https://upload.wikimedia.org/wikipedia/commons/0/09/YouTube_full-color_icon_%282017%29.svg"

    return video_url, thumbnail_url


def get_feature(model, image):
    data = image.to(torch.device("cuda"))
    with torch.no_grad():
        output = model(data)
    output = output.cpu().detach().numpy()
    output = normalize(output).flatten()
    return np.matrix(output)

def get_features(model, images, batch_size = 8):
    data = images.to(torch.device("cuda"))
    with torch.no_grad():
        output = model(data)
    output = output.cpu().detach().numpy()
    output = normalize(output)
    return np.matrix(output)


def get_json_dict(path):
    if os.path.exists(path):
        with open(path, mode='r', encoding='utf-8') as _f:
            return json.load(_f)
    else:
        return {}


def get_vector2index():
    return faiss.IndexFlatL2(512)


class CFG():
    def __init__(self):
        self.vector2index = get_vector2index()

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'CFG':
            return CFG
        return super().find_class(module, name)

def search_vector(model, image, cfg, index2id):
    start_time_load_model = time.time()

    feature = get_feature(model, image)

    print('TIME to load and predict: ', time.time() - start_time_load_model)

    start_time_load_model = time.time()

    _, lst_index = cfg.vector2index.search(feature, k=50)

    print('TIME to search: ', time.time() - start_time_load_model)
    

    lst_result = []
    for index in lst_index[0]:
        result = index2id[str(index)]
        if result not in lst_result:
            lst_result.append(result)
        if len(lst_result) == 10:
            break
        
    start_time_load_model = time.time()
    _result = []
    for _, result in enumerate(lst_result[:10]):
        prompt_text = str(result[1]) + "-" + str(result[2])
        vid_url, thumbnail_url = get_video_url(prompt_text)
       
        # cache[prompt_text] = {
        #         'video_url': vid_url,
        #         'thumbnail_url': thumbnail_url
        #     }
        _result.append(
            {
                "id": result[0],
                "title": result[1],
                "artist": result[2],
                "youtube_link": vid_url,
                "thumbnail_link": thumbnail_url,
            }
        )
    print('TIME to get urls: ', time.time() - start_time_load_model)
    
    return _result


def create_search_dict(root_song, input_shape, model, mp3_root, segment_overlap = 0.5, save_dir="./checkpoints"):
    
    cfg = CFG()
    list_song = os.listdir(root_song)
    index2id = {"-1": ""}
    id = 0
    for name_song in tqdm(list_song):
        path_song = os.path.join(root_song, name_song)
        # image = load_image(path_song, input_shape)
        images = load_images(path_song, input_shape, overlap_size=segment_overlap)
        tag = TinyTag.get(os.path.join(mp3_root, name_song.split(".")[0] + ".mp3")) # failed font
        # audio_file = eyed3.load(os.path.join(mp3_root, name_song.split(".")[0] + ".mp3"))
        for image in images:
            cfg.vector2index.add(get_feature(model, image.unsqueeze(0)))
            index2id[str(id)] = [name_song.split('.')[0], tag.title, tag.artist]
            # index2id[str(id)] = [name_song.split('.')[0], audio_file.tag.title, audio_file.tag.artist]
            # print([name_song.split('.')[0], audio_file.tag.title, audio_file.tag.artist])
            id+=1
    
    # need to save this two thing instead of returning them
    with open(os.path.join(save_dir, 'index2id.pkl'), 'wb') as f:
        pickle.dump(index2id, f)
    
    with open(os.path.join(save_dir, 'cfg.pkl'), 'wb') as f:
        pickle.dump(cfg, f)

def songs2search_dict(root_dir, input_shape=(630, 80), batch_size = 8, segment_overlap = 0.5, audio_position = (0.1, 0.9), device = 'cuda', save_dir = "./checkpoints", checkpoint_name = "resnet18_latest.pth"):
    start_time_load_model = time.time()
    model = wrap_resnet_face18(False)
    model.load_state_dict(torch.load(os.path.join(save_dir, checkpoint_name)))
    model.to(device)
    model.eval()

    cfg = CFG()
    index2id = {"-1": ""}    

    allowed_extension = {'mp3', 'wav', 'flac'}

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extension

    l_index2id = len(index2id) - 1
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in tqdm(files):
            file_path = os.path.join(root, name)
            if allowed_file(name):
                spec = process_file(file_path, sampling_rate, max_wav_value, STFT, full_song=True, load_method="pydub")
                tag = TinyTag.get(file_path) # failed font
                images = get_images(spec, input_shape, audio_position = audio_position, overlap_size=segment_overlap)

                for i in range(0, len(images), batch_size):
                    if i + batch_size < len(images):
                        batch_img = images[i: i+batch_size]
                        features = get_features(model, batch_img, batch_size)
                        for feature in features:
                            cfg.vector2index.add(feature)
                            index2id[str(l_index2id)] = [None, tag.title, tag.artist] # None since new song has no id
                            # print([None, tag.title, tag.artist])
                            l_index2id+=1
                    else:
                        batch_img = images[i:]
                        features = get_features(model, batch_img, batch_size)
                        for feature in features:
                            cfg.vector2index.add(feature)
                            index2id[str(l_index2id)] = [None, tag.title, tag.artist] # None since new song has no id
                            # print([None, tag.title, tag.artist])
                            l_index2id+=1
            
    # need to save this two thing instead of returning them
    with open(os.path.join(save_dir, 'index2id.pkl'), 'wb') as f:
        pickle.dump(index2id, f)
    
    with open(os.path.join(save_dir, 'cfg.pkl'), 'wb') as f:
        pickle.dump(cfg, f)

def adding_songs2cfg(root_dir, input_shape=(630, 80), batch_size = 8, segment_overlap = 0.5, audio_position = (0.1, 0.9), save_dir = "./checkpoints"):
    model, cfg, index2id = load_dependencies(root_path="./checkpoints", checkpoint_name="resnet18_latest.pth")
    
    allowed_extension = {'mp3', 'wav', 'flac'}

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extension

    l_index2id = len(index2id) - 1
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in tqdm(files):
            file_path = os.path.join(root, name)
            if allowed_file(name):
                spec = process_file(file_path, sampling_rate, max_wav_value, STFT, full_song=True, load_method="pydub")
                tag = TinyTag.get(file_path) # failed font
                images = get_images(spec, input_shape, audio_position, overlap_size=segment_overlap)

                for i in range(0, len(images), batch_size):
                    if i + batch_size < len(images):
                        batch_img = images[i: i+batch_size]
                        features = get_features(model, batch_img, batch_size)
                        for feature in features:
                            cfg.vector2index.add(feature)
                            index2id[str(l_index2id)] = [None, tag.title, tag.artist] # None since new song has no id
                            # print([None, tag.title, tag.artist])
                            l_index2id+=1
                    else:
                        batch_img = images[i:]
                        features = get_features(model, batch_img, batch_size)
                        for feature in features:
                            cfg.vector2index.add(feature)
                            index2id[str(l_index2id)] = [None, tag.title, tag.artist] # None since new song has no id
                            # print([None, tag.title, tag.artist])
                            l_index2id+=1
            
    # need to save this two thing instead of returning them
    with open(os.path.join(save_dir, 'index2id.pkl'), 'wb') as f:
        pickle.dump(index2id, f)
    
    with open(os.path.join(save_dir, 'cfg.pkl'), 'wb') as f:
        pickle.dump(cfg, f)



def load_dependencies(root_path = "./checkpoints", checkpoint_name = 'resnet18_best.pth', index2id_name = 'index2id.pkl', cfg_name = 'cfg.pkl'):
    # function to load nescessary objects (model, index2id, cfg)
    start_time_load_model = time.time()
    model = wrap_resnet_face18(False)
    model.load_state_dict(torch.load(os.path.join(root_path, checkpoint_name)))
    model.to('cuda')
    model.eval()
    print('TIME LOAD MODEL: ', time.time() - start_time_load_model)

    # load index2id dict
    with open(os.path.join(root_path, index2id_name), 'rb') as f:
        index2id = pickle.load(f)
    # # load cfg object
    # with open(os.path.join(root_path, cfg_name), 'rb') as f:
    #     cfg = pickle.load(f)
    cfg = CustomUnpickler(open(os.path.join(root_path, cfg_name), 'rb')).load()

    return model, cfg, index2id

def get_result(root_hum, name_hum, model, cfg, index2id):
    path_hum = os.path.join(root_hum, name_hum)
    image = process_file(path_hum, sampling_rate, max_wav_value, STFT, load_method="librosa")
    rsult_song = search_vector(model, image, cfg, index2id)
    return rsult_song




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--song_dir", type=str, required=True, help="path to songs")
    parser.add_argument("--out_dir", type=str, required=False, default = "./checkpoints", help="path to save output files")
    parser.add_argument("--batch_size", type=int, required=False, default=8, help="# segments to generate embeddings at a time")
    parser.add_argument("--overlap_size", type=float, required= False, default = 0.7, help="overlapping rate between 2 consecutive segments")
    parser.add_argument("--adding_song", action = "store_true", required=False, help="if adding new song to created search dict, use --adding_song")
    args = parser.parse_args()
    input_shape = (630, 80)
    audio_position = (0.1, 0.9) # start and end point of a song to start generating embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # checkpoint_path = "./checkpoints"
    # start_time_load_model = time.time()
    # model = wrap_resnet_face18(False)
    # model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'resnet18_best.pth')))
    # model.to('cuda')
    # model.eval()
    # print('TIME LOAD MODEL: ', time.time() - start_time_load_model)

    # root_song = "/media/dungpham/New Volume/AIproject/AI_Audio/hum2song/hum2song/preprocessed/public_test/full_song"
    # mp3_root =  "/media/dungpham/New Volume/AIproject/AI_Audio/hum2song/hum2song/data/public_test/full_song"
    # create_search_dict(root_song, input_shape, model, mp3_root)

    # new_mp3_root = "/media/dungpham/New Volume/music/s22ultra_music"
    if args.adding_song:
        print("adding new songs to current seach dictionary")
        adding_songs2cfg(args.song_dir, input_shape, args.batch_size, args.overlap_size, audio_position, args.out_dir)
    else:
        print("create new search dictionary")
        songs2search_dict(args.song_dir, input_shape, args.batch_size, args.overlap_size, audio_position, device, args.out_dir)


# python torch_utils.py --song_dir "/media/dungpham/New Volume/AIproject/AI_Audio/hum2song/hum2song/data/train/song"
