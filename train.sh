#!/bin/bash

rm img*.png
python3 hentai.py
tar cvzf imgs_hentai.tgz img*.png losses

rm img*.png
python3 anime.py
tar cvzf imgs_anime.tgz  img*.png losses

rm img*.png
python3 CelebA.py
tar cvzf imgs_celeba.tgz img*.png losses
