#!/bin/bash
cp .gitignore .gitignore.bak
# find * -size +49M | cat >> .gitignore
git add -A
git commit -m "Auto commit $(date +%H/%M/%m/%d/%Y)"
# rm .gitignore
# mv .gitignore.bak .gitignore
git push -f