pyinstaller main.spec
cp dist/main b_1020_alpha/main
cd b_1020_alpha
tar zcvf a.tgz main
rm main