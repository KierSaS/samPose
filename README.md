We are very busy recently. Some projects are closing until December 12th. All codes and experimental pictures will be sorted out before 2024/12/13.
In this project, I will upload some experimental data and code related to our new work, **SamPose**, for localized evaluation. I believe this work significantly relaxes the prior conditions for object pose estimation, making pose estimation less complex. 

#1.Prepare
SamPose is trained on the [Google Scaned Object Dataset ]([http://example.com](https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/webdatasets/)) and the [MegaDepth-1500V2 ]([http://example.com](https://www.cs.cornell.edu/projects/megadepth/)).
Your directory structure should look like this:
$DATA_DIR/
MegaPose-Training-Data/
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



      


**You can experiment with localization by running the following code:**
