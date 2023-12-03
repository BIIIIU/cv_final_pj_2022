rm  -r /project/train/src_repo/datasets
mkdir /project/train/src_repo/datasets
python /project/train/src_repo/predata.py
python /project/train/src_repo/train.py --project /project/train/models/train_save