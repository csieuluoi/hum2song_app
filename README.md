# hum2song_app
simple humming to song web app with flask and pytorch
***just to test out the hum2song method from this repo: https://github.com/vovanphuc/hum2song***
1. Download model checkpoint (resnet18), index2id.pkl and cfg.pkl (contain a faiss IndexFlatL2 object to do similarity search) from
   https://drive.google.com/drive/folders/1Y1y0iQUIPWi2qFrELIPiidKrFx2EgtcB?usp=sharing
   and put them in the "checkpoints" folder within the main directory (hum2song_app)

   To create your own index2id.pkl and cfg.pkl for your own songs:
   1. put all songs in a folder
   2. to create new search dictionary, use command:
      ```
      python torch_utils.py --song_dir "path-to-your-songs-folder"
      ```
      to adding new songs to your current dictionary, use "--adding_song" option:
      ```
      python torch_utils.py --song_dir "path-to-your-songs-folder" --adding_song
      ```
3. Install pytorch:
   https://pytorch.org/get-started/locally/
   
4. Install requirements:
   ```sh
   pip install -r requirements.txt
   ```
6. run flask app:
   ```sh
   flask --app main.py run 
   ```
