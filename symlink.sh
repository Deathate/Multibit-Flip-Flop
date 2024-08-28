#!/usr/bin/bash
filename=$(basename "$1")
ln -sf $filename cases/current.txt
echo "Symlink created for $filename"