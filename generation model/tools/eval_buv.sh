export CUDA_VISIBLE_DEVICES=3
# python tools/calc_metrics_for_dataset.py \
# --real_data_path /path/to/real_data//images \
# --fake_data_path /path/to/fake_data/images \
# --mirror 1 --gpus 1 --resolution 256 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0



# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_16frames \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0190000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 1 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 
 
python tools/calc_metrics_for_dataset.py \
    --real_data_path /home/extradisk/liuyaofang/datasets/Miccai_2022_BUV_Dataset/rawvideos_train_img_crop_resize_fvd \
    --fake_data_path /home/extradisk/liuyaofang/Latte/test/busi_animated_no_class_4_0100000_img \
    --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
    --metrics fvd2048_16f \
    --verbose 0 --use_cache 0 &

python tools/calc_metrics_for_dataset.py \
    --real_data_path /home/extradisk/liuyaofang/datasets/Miccai_2022_BUV_Dataset/rawvideos_train_img_crop_resize_fvd \
    --fake_data_path /home/extradisk/liuyaofang/Latte/test/busi_animated_no_class_4_0100000_img \
    --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
    --metrics fvd2048_16f \
    --verbose 0 --use_cache 0


# # sleep 0.3h
# # base_path="/home/extradisk/liuyaofang/Latte/test_buv_i2v_results"
# base_path="/home/extradisk/liuyaofang/Latte/test_buv_results"

# # List of folder names
# # folders=(
#     # "0020000_fvd"
#     # "0040000_fvd"
#     # "0060000_fvd"
#     # "0080000_fvd"
#     # "0100000_fvd"
#     # "busi_animated_no_class_1_img"
#     # "busi_animated_no_class_2_img"
#     # "busi_animated_no_class_3_img"
#     # "busi_animated_no_class_4_img"
#     # "busi_animated_no_class_5_img"
#     # "busi_animated_no_class_6_img"
#     # "busi_animated_no_class_7_img"
#     # "busi_animated_no_class_8_img"
#     # "busi_animated_no_class_14_img"
#     # "busi_animated_no_class_15_img"
#     # "busi_animated_no_class_1_img"
#     # "busi_animated_no_class_2_img"
#     # "busi_animated_no_class_3_img"
#     # "busi_animated_no_class_4_img"
#     # "busi_animated_no_class_5_img"
#     # "busi_animated_no_class_6_img"
#     # "busi_animated_no_class_7_img"
#     # "busi_animated_no_class_8_img"
#     # "busi_animated_no_class_14_img"
#     # "busi_animated_no_class_15_img"
# #     "busi_animated_no_class_9_img"
# #     "busi_animated_no_class_10_img"
# #     "busi_animated_no_class_11_img"
# #     "busi_animated_no_class_12_img"
# #     "busi_animated_no_class_13_img"
# #     "busi_animated_no_class_9_img"
# #     "busi_animated_no_class_10_img"
# #     "busi_animated_no_class_11_img"
# #     "busi_animated_no_class_12_img"
# #     "busi_animated_no_class_13_img"
# # )

# # Use find to get all directories that contain '_img' in their names
# folders=($(find "$base_path" -maxdepth 1 -type d -name "*_img"))

# # # # #### Process each group of three folders
# # for ((i=0; i<${#folders[@]}; i+=2)); do
# #   for j in {0..1}; do
# #     folder_index=$((i + j))
# #     if [ $folder_index -lt ${#folders[@]} ]; then
# #       folder=${folders[$folder_index]}
# #       fake_data_path="$base_path/$folder"

# #       # Run the python command in the background
# #       python tools/calc_metrics_for_dataset.py \
# #         --real_data_path /media/hdd/yfliu/datasets/Miccai_2022_BUV_Dataset/test_fvd \
# #         --fake_data_path "$fake_data_path" \
# #         --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# #         --metrics fvd2048_16f \
# #         --verbose 0 --use_cache 0 &
# #     fi
# #   done
# #   # Wait for all background processes to finish before starting the next group
# #   wait
# # done

# #### Process one by one
# for folder in "${folders[@]}"; do
#   fake_data_path="$folder"

#   # Run the python command in the background   
#   python tools/calc_metrics_for_dataset.py \
#     --real_data_path /home/extradisk/liuyaofang/datasets/Miccai_2022_BUV_Dataset/rawvideos_train_img_crop_resize_fvd \
#     --fake_data_path "$fake_data_path" \
#     --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
#     --metrics fvd2048_16f \
#     --verbose 0 --use_cache 0
# done

# # Wait for all background processes to finish
# wait