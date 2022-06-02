# 3dcv2022_proposal_group_21
how to run the code:
1. go to scirpts
2. (use python2 )(this code is run the dataset 7scenes in scene chess, two_stream_mode 0  is only RGB,two_stream_mode 1 is only Depth, two_stream_mode 2 is RGB-D)
python2 geoposenet_train.py --dataset 7Scenes --scene chess --config_file configs/geonet.ini --model geoposenet --device 1 --suffix _cmap --d_suffix full_d_cmap --two_stream_mode 1 

python2 geoposenet_train.py --dataset 7Scenes --scene chess --config_file configs/geonet.ini --model geoposenet --device 1 --suffix _rgb --d_suffix full_d_cmap --two_stream_mode 0

python2 geoposenet_train.py --dataset 7Scenes --scene chess --config_file configs/geonet.ini --model geoposenet --device 1 --suffix _stage2 --d_suffix full_d_cmap --two_stream_mode 2 --rgb_cp logs/7Scenes_chess_geoposenet_geonet_rgb/epoch_200.pth.tar --depth_cp logs/7Scenes_chess_geoposenet_geonet_cmap/epoch_150.pth.tar

3. each 50 epoches will be saved in the logs
4. we will choose model which have best performance in single stream(RGB, Depth) as our pre-trained model.

our demo link https://youtu.be/mfyBjN-EgY4
