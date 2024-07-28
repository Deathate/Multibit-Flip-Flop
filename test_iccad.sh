rm -r iccad
scp -r iccad:/home/b_1020/Multibit-Flip-Flop/cases iccad
# ./sanity iccad/sampleCase.txt iccad/sampleOutput.txt
./sanity iccad/testcase1_0614.txt iccad/output/v2o.txt
./sanity iccad/v2.txt iccad/output/output.txt