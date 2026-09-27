#!/bin/bash
# view.sh first last  -> renders pages at 75dpi, montages pairs into scratch/v-N.png
S=/tmp/claude-0/-home-user-Claude-practice/0b100809-d109-51a4-a5fe-91fc08dcb556/scratchpad/v
mkdir -p $S; rm -f $S/*
pdftoppm -r ${3:-75} -png -f $1 -l $2 src/main.pdf $S/p
cd $S; files=($(ls p-*.png | sort -V)); n=${#files[@]}
for ((i=0;i<n;i+=2)); do python3 /home/user/Claude_practice/cpp-pointer-notes/montage.py m-$i.png ${files[@]:i:2}; done
ls $S/m-*.png
