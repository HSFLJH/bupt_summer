# 查看显卡设置

```bash
nvidia-smi
```

# 设置显卡，编号。

```bash
export CUDA_VISIBLE_DEVICES=3
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

# torchvision版本：

```bash
cd /home/lishengjie/study/sum_jiahao/bupt_summer/mask-r-cnn/torchvision
```

训练：直接点击右上角运行，或者进入对应目录:

```bash
python train.py
```

可视化数据：

```bash
/home/lishengjie/miniconda3/envs/mask_rcnn/bin/tensorboard --logdir=/home/lishengjie/study/sum_jiahao/bupt_summer/mask-r-cnn/torchvision/result/three/tensorboard/ --port=9999
```

可视化结果：进入对应的文件夹torchvision/，然后：

```bash
python demo.py
```

# pytorch版本：

```bash
cd /home/lishengjie/study/sum_jiahao/bupt_summer/mask-r-cnn/pytorch
```

训练： 

```bash
python main.py --train
```

数据增强预览：

```bash
python main.py --augmentation-preview
```

评估结果预览：

```bash
python main.py --demo
```

训练数据预览：

```bash
/home/lishengjie/miniconda3/envs/mask_rcnn/bin/tensorboard --logdir=/home/lishengjie/study/sum_jiahao/bupt_summer/mask-r-cnn/pytorch/result/one/tensorboard/ --port=9999
```