We are very busy recently. We are currently writing instructions for using the program to ensure that all researchers can run it successfully. All codes will be added after testing.
In this project, I will upload some experimental data and code related to our new work, **SamPose**. I believe this work significantly relaxes the prior conditions for object pose estimation, making pose estimation less complex. 
We can simply understand the structure of the method through the following figure:
<img src="https://github.com/user-attachments/assets/62cb56cc-cb68-45ee-be1c-130ac84b03a3" width="50%" />
#  Prepare
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
Some additional datasets need to be downloaded, and you can customize some tests by replacing the corresponding files in the demo. [Linemod](https://drive.google.com/file/d/1W1puYf21XHWgMxGtTf3La0CUu_GrFMBd/view?usp=drive_link)、[YCBV](https://drive.google.com/file/d/1gukwaUfJ06d9EbmaH8qDqQb7hRzI0PAk/view?usp=drive_link)、[T-LESS](https://bop.felk.cvut.cz/datasets/#T-LESS) datasets for SamPose.
We provide some test examples for each dataset in the damo folder. 
Each researcher can replace the images and their corresponding files in Linemod, YCBV, T-LESS to implement any test they want. As we have added many packages during the development process, you may need to install some additional development packages based on the information during runtime.

**Try out localization by running the following code:**
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
#**Due to the page limit of the paper, we would like to publish some additional interesting work here:**

We studied the impact of the overlap between two views on the accuracy of pose estimation and reported the recall rate of pose estimation of our method under different overlaps. 
The quantitative and qualitative presentations are shown in the figure:
![image](https://github.com/user-attachments/assets/d98cbc88-cf3a-4426-b56e-539460b3377f)
![Screenshot_1](https://github.com/user-attachments/assets/1cb95cb0-184a-409b-a5d0-c30c7299029c)

Our pose estimation method is a model-free method. The methods in the BOP ranking are all based on model rendering views, and their AR indicators mainly include MSSD, MSPD and VSD. The indicator baselines followed by model-free methods are ADD-0.1d and Prj-5. We did not find a program that can directly test AR indicators. We combined the model data in BOP and the definitions of MSSD, MSPD and VSD indicators in the BOP paper to write a new evaluation program. Any method that can estimate the object pose {R, t} can be combined with the object model to perform similar indicator calculations.
For detailed calculations, refer to `BOP_metrix.py`.
![Screenshot_2](https://github.com/user-attachments/assets/e05c2422-32ad-4a9e-9e6d-4a79940a31db)










      



