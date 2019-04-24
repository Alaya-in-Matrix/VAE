#!/bin/bash
rm img*.png
python CelebA.py > log 2>err 
tar cvzf imgs_celeba.tgz img*.png

rm img*.png
python anime.py
tar cvzf imgs_anime.tgz img*.png

rm img*.png
python hentai.py
tar cvzf imgs_hentai.tgz img*.png

/root/shutdown.sh
