find ../ADNI -iname \*.nii|while read f; do echo $f; python ./nfi_rotate.py "$f";done
python train_test_split.py data data/train data/test
python train.py -c config.json