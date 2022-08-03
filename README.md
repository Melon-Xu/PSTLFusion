# PSTLFusion
This is the official Tensorflow implementation of "Infrared and Visible Image Fusion via Parallel Scene and Texture Learning".
## Overall Framework
![image](https://user-images.githubusercontent.com/52787358/182607951-fe64c6fa-5213-4cac-9db6-9a1d5d5afbc0.png)
The overall framework of the fusion process of proposed method.
## Network Architecture
![image](https://user-images.githubusercontent.com/52787358/182608264-1f54b303-854e-4f4c-9584-faa6ca8e0b7b.png)
The architecture of the proposed infrared and visible image fusion method via parallel scene and texture learning.
## To Train
Run **python main.py --phase train** to train your model. The format of the training data must be HDF5.
## To Test
Run **python main.py --phase guide** to test the model.
## Fusion Example
![image](https://user-images.githubusercontent.com/52787358/182609289-aac3701e-2920-4a57-abf4-4c31fecad546.png)
Qualitative comparison of PSTLFusion with 7 state-of-the-art methods on TNO and RoadScene datasets.
## Detection Result
![image](https://user-images.githubusercontent.com/52787358/182609636-65d0f220-8d01-4b72-99c2-78ca5a910133.png)
These are some object detection results for infrared, visible and fused images from the MFNet dataset. We pre-train the YOLOv5 detector on the CoCo dataset and deploy it on our fused result.
## If this work is helpful to you, please cite it as:
~~~
@article{xu2022infrared,
  title={Infrared and visible image fusion via parallel scene and texture learning},
  author={Xu, Meilong and Tang, Linfeng and Zhang, Hao and Ma, Jiayi},
  journal={Pattern Recognition},
  pages={108929},
  year={2022},
  publisher={Elsevier}
}
~~~


