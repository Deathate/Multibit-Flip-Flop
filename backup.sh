#!/bin/bash
find * -size +99M | cat >> .gitignore
git add .
git commit -m "Auto commit $(date +%H/%M/%m/%d/%Y)"
git push

