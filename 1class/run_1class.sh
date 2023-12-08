rm  -r /project/train/src_repo/1class/datasets 
mkdir /project/train/src_repo/1class/datasets 
python /project/train/src_repo/1class/predata.py
python /project/train/src_repo/1class/train.py --project /project/train/models/train_1class --data /project/train/src_repo/1class/test.yaml