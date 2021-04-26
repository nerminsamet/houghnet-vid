
# train
python src/main.py --exp_id ilsvrc2015_temporal_resdcn101 --arch resdcn_101 --batch_size 32 --master_batch 5 --lr 1.0e-4 --gpus 0,1,2,3 --num_workers 4 --lr_step 50 --num_epochs 80 --max_offset 4 --min_offset -4

# delta={-4}
python src/test.py --exp_id ilsvrc2015_temporal_resdcn101 --ref_num 1 --test_offsets -4 --resume --load_model ./models/vid_ilsvrc2015_resdcn101.pth

# delta={-4,4}
python src/test.py --exp_id ilsvrc2015_temporal_resdcn101 --ref_num 2 --test_offsets -4,4 --resume --load_model ./models/vid_ilsvrc2015_resdcn101.pth
