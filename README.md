# hum2song_app
simple humming to song web app with flask and pytorch

1. Download model checkpoint (resnet18), index2id.pkl and cfg.pkl (contain a faiss IndexFlatL2 object to do similarity search) from
   https://drive.google.com/drive/folders/1Y1y0iQUIPWi2qFrELIPiidKrFx2EgtcB?usp=sharing
   and put them in the "checkpoints" folder within the main directory (hum2song_app)
2. Install pytorch:
   https://pytorch.org/get-started/locally/
   
4. Install requirements:
   ```sh
   pip install -r requirements.txt
   ```
6. run flask app:
   ```sh
   flask --app main.py run 
   ```
