We are very busy recently. We are currently writing instructions for using the program to ensure that all researchers can run it successfully. All codes will be added after testing.
In this project, I will upload some experimental data and code related to our new work, **SamPose**. I believe this work significantly relaxes the prior conditions for object pose estimation, making pose estimation less complex. 
We can simply understand the structure of the method through the following figure:
![image](https://github.com/user-attachments/assets/62cb56cc-cb68-45ee-be1c-130ac84b03a3)




# 1. Prepare

SamPose is trained on the [Google Scanned Object Dataset](https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/webdatasets/) and the [MegaDepth-1500V2](https://www.cs.cornell.edu/projects/megadepth/). Your directory structure should look like this:
`````````MegaPose-Training-Data/
├── MegaPose-GSO/
│   ├── google_scanned_objects/
│   ├── templates/
│   └── train_pbr_web/
└── MegaDepth/
    ├── Cameras/
    │   ├── 0000.txt
    │   ├── 0001.txt
    │   └── ...
    ├── densepose/
    │   ├── 0000_IUV.png
    │   ├── 0001_IUV.png
    │   └── ...
    ├── depth/
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    ├── images/
    │   ├── 0000.jpg
    │   ├── 0001.jpg
    │   └── ...
    └── pairs/
        ├── 0000.txt
        ├── 0001.txt
        └── ...
`````````
Some additional datasets need to be downloaded, and you can customize some tests by replacing the corresponding files in the demo. [Linemod].()、[YCBV].() 、[T-LESS].(https://bop.felk.cvut.cz/datasets/#T-LESS) datasets for SamPose. We provide some test examples for each dataset in the damo folder. 
Each researcher can replace the images and their corresponding files in Linemod, YCBV, T-LESS to implement any test they want.
**You can experiment with localization by running the following code:**
# **Visual DINOv2 feature**
`````````
python3 visual_dinov2.py
`````````
# **Visual Segment Anything Model**
`````````
python3 visual_sam.py
`````````
# **Visual 3D BBox test**
`````````
python3 visual_3dbbox_Linemod.py

python3 visual_3dbbox_YCB-V.py

python3 visual_3dbbox_LT-LESS.py
`````````







      



