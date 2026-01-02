#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

# sleep 4h

# kill 75179
### 别忘t[:,0] = 0！！！

###### for test only, not whole eval ###### 
# python sample/sample_video_buv_i2v.py \
# --config ./configs/buv/buv_sample_video_no_class.yaml \
# --ckpt /home/extradisk/liuyaofang/Latte/results_noise_video/2024-10-01_21-34-24/000-LatteVIDEO-B-2-F16S3-buv/checkpoints/0100000.pt \
# --save_video_path ./test_buv_results/busi_animated_no_class_14

# python sample/sample_video_buv_i2v.py \
# --config ./configs/buv/buv_sample_video_no_class.yaml \
# --ckpt /home/extradisk/liuyaofang/Latte/results_noise_video/2024-10-01_21-34-24/000-LatteVIDEO-B-2-F16S3-buv/checkpoints/0100000.pt \
# --save_video_path ./test_buv_results/busis_animated_no_class_0

# python sample/sample_video_buv_i2v.py \
# --config ./configs/buv/buv_sample_video_no_class.yaml \
# --ckpt /home/extradisk/liuyaofang/Latte/results_noise_video/2024-10-01_21-34-24/000-LatteVIDEO-B-2-F16S3-buv/checkpoints/0100000.pt \
# --save_video_path ./test_buv_results/BUSBRA_animated_no_class_0 

# python sample/sample_video_buv_i2v.py \
# --config ./configs/buv/buv_sample_video_no_class.yaml \
# --ckpt /home/extradisk/liuyaofang/Latte/results_noise_video/2024-10-01_21-34-24/000-LatteVIDEO-B-2-F16S3-buv/checkpoints/0100000.pt \
# --save_video_path ./test_buv_results/BrEaST_animated_no_class_0



# python sample/sample_video_buv_i2v.py \
# --config ./configs/buv/buv_sample_video_no_class.yaml \
# --ckpt /home/extradisk/liuyaofang/Latte/results_noise_video/2024-10-01_21-34-24/000-LatteVIDEO-B-2-F16S3-buv/checkpoints/0090000.pt \
# --save_video_path ./test_buv_results/busi_animated_no_class_0_0090000 \
# --frame_num 0 \
# --input_dir /home/extradisk/liuyaofang/datasets/BrEaST_with_class_crop_resize


# Array of frame numbers
# frame_nums=(0 1 3 5 7 9 11 13 15)
frame_nums=(0 7 15)

# Array of checkpoints
ckpts=("0100000" "0090000" "0080000" "0070000" "0060000" "0050000" "0040000")

datasets=("BUSBRA" "BUSIS" "BrEaST")

# Loop through checkpoint files
for dataset in  "${datasets[@]}"; do
    for ckpt in "${ckpts[@]}"; do
        # Base command
        base_command="python sample/sample_video_buv_i2v.py \
        --config ./configs/buv/buv_sample_video_no_class.yaml \
        --ckpt /home/extradisk/liuyaofang/Latte/results_noise_video/2024-10-01_21-34-24/000-LatteVIDEO-B-2-F16S3-buv/checkpoints/${ckpt}.pt"

        # Loop through frame numbers
        for frame_num in "${frame_nums[@]}"; do
            # Construct the save path
            save_path="./test_buv_i2v_results/${dataset}_animated_no_class_${frame_num}_${ckpt}"
            input_dir="/home/extradisk/liuyaofang/datasets/${dataset}_with_class_crop_resize"
            # Run the command in the background
            $base_command --save_video_path "$save_path" --frame_num "$frame_num" --input_dir "$input_dir" &
            
            # Check the number of background jobs and wait if there are 3
            while (( $(jobs -r | wc -l) >= 4 )); do
                wait -n
            done
        done
    done
done
# Wait for all remaining background jobs
wait

echo "All experiments completed."