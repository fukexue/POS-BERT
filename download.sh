#!/bin/bash

# cd scratch place
cd data/

https://drive.usercontent.google.com/download?id=13iIwx4_TCok1_Aq9qHN7EvbvTdVm8OJ3&export=download&authuser=0&confirm=t&uuid=812aec24-0463-4e1c-8710-32846730c5cd&at=APZUnTXv-2JDGs5emSWtAJeKga2m:1699317380373

# Download zip dataset from Google Drive
filename='modelnet40_test_8192pts_fps.dat'
fileid='13iIwx4_TCok1_Aq9qHN7EvbvTdVm8OJ3'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

# Unzip
# unzip -q ${filename}
# rm ${filename}
cd
