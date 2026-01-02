#!/bin/bash
export CUDA_VISIBLE_DEVICES=5



###### for test only, not whole eval ###### 
python sample/sample_video_buv.py \
--config ./configs/buv/buv_sample_video.yaml \
--ckpt /home/extradisk/liuyaofang/Latte/results_noise_video/2024-09-27_10-55-25/000-LatteVIDEO-B-2-F16S3-buv/checkpoints/0080000.pt \
--save_video_path ./test_buv_results/buv_gen
