#!/bin/bash

rm img*.png
python3 CelebA.py > log 2>err 
tar cvzf imgs_celeba.tgz img*.png
cp imgs_celeba.tgz /data

rm img*.png
python3 hentai.py
tar cvzf imgs_hentai.tgz img*.png
cp imgs_hentai.tgz /data


rm img*.png
python3 anime.py
tar cvzf imgs_anime.tgz img*.png
cp imgs_anime.tgz /data
