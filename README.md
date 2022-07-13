本项目针对弱监督舰船目标检测，基于TS-CAM代码修改.(W. Gao, F. Wan, X. Pan, Z. Peng, Q. Tian, Z. Han, B. Zhou, and Q. Ye, “Ts-cam: Token semantic coupled attention map for weakly supervised object localization,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2021, pp. 2886–2895.)
数据集使用CUB格式
训练：python ./tools_cam/train.py --config_file ./configs/CUB/deit_tscam_tiny_patch16_224.yaml --lr 5e-5
演示使用demo.py 将./lib/models/deit.py 替换为deit_for_demo.py
论文引用：
Y. Yang, Z. Pan, Y. Hu and C. Ding, "PistonNet: Object Separating From Background by Attention for Weakly Supervised Ship Detection," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 15, pp. 5190-5202, 2022, doi: 10.1109/JSTARS.2022.3184637.
