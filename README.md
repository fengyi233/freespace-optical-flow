# Freespace Optical Flow Modeling

Official implementation of "Freespace Optical Flow Modeling for Automated Driving" ([T-MECH 23']())

Yi Feng, Ruge Zhang, Jiayuan Du, Qijun Chen, and Rui Fan

| [Webpage](https://mias.group/FSOF) 
| [Full Paper](https://ieeexplore.ieee.org/abstract/document/10224328)
| [datasets]()




## **Environment setup**

Install environment using conda: 
```bash
conda create -n fsof python=3.8
conda activate fsof
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## **Data preparation**
### CARLA
We created this dataset using the Open-source simulator [CARLA](https://carla.org//) for 
validation of the proposed `Velocity-Based Optical Flow Model`. 

### KITTI
We use the [KITTI Scene Flow](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php)
dataset for validation of the proposed `Displacement-Based Optical Flow Model`. 
It's worth noting that:
1. The original dataset does not provide semantic labels, so we manually 
labeled some of the images. 
2. The optical flow ground truth is sparse. 

### VKITTI2
We use the [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
dataset for validation of the proposed `Displacement-Based Optical Flow Model`. 
It's worth noting that:
1. This dataset provides ground truth of camera poses, allowing us to verify the model's
ability in precise localization when optical flow map is given.
2. The optical flow ground truth is dense. 

The datasets can be downloaded from:
[Baidu Disk](https://pan.baidu.com/s/1Me9p9VUmLW_h_3Wueom1yg?pwd=tqww) | [Google Drive]().
You can also download them from the official websites. 

Put the datasets into the `./data` directory, and organize them as follows:
```
CARLA
 |-- t10_s0_r-5
 |  |-- optical_flow
 |  |-- rgb
 |  |-- semantic
 |  |-- poses.txt
 |-- t10_s0_r-10
 |  |-- ...
 |-- ...
 KITTI
 |-- flow_noc
 |-- image_2
 |-- semantic
 VKITTI2
 |-- vkitti_2.0.3_classSegmentation
 |-- vkitti_2.0.3_forwardFlow
 |-- vkitti_2.0.3_rgb
 |-- vkitti_2.0.3_textgt
```


## **Run the Code**
We provide three python scripts for model validation and evaluation on different datasets.
It is recommended to run them in your `Python Console` instead of the Terminal for 
the sake of visualization.

### CARLA Experiment
```bash
python carla.py --theta_real -10 --pic_num 70
```
### KITTI Experiment
```bash
python kitti.py --pic_num 72
```
### VKITTI2 Experiment
```bash
python vkitti2.py --scene 1 --pic_num 4
```
You can change the parameters to validate the model on other pictures
or scenes. 


## **Citation**
If you find our work useful for your research, please consider citing the paper:

```
@article{feng2023freespace,
  title={Freespace Optical Flow Modeling for Automated Driving},
  author={Feng, Yi and Zhang, Ruge and Du, Jiayuan and Chen, Qijun and Fan, Rui},
  journal={IEEE/ASME Transactions on Mechatronics},
  year={2023},
  publisher={IEEE}
}
```