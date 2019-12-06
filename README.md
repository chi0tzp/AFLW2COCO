# AFLW2COCO: convert AFLW annotation into COCO format

![AFLW](aflw_cover.jpg)

Annotated Facial Landmarks in the Wild ([AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/#)) [1] provides a large-scale collection of annotated face images gathered from the web, exhibiting a large variety in appearance (e.g., pose, expression, ethnicity, age, gender) as well as general imaging and environmental conditions. In total about **25k faces** are annotated with up to **21 landmarks** per image.



##### Step 1: Clone repo

~~~
git clone https://github.com/chi0tzp/AFLW2COCO.git
~~~



##### Step 2: Download dataset

Download dataset following the instruction described [here](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/). More specifically, download the following zip files:

[...]

and extract them under the same directory; for instance under `AFLWDataset/`. After extraction, `AFLWDataset/` should include the following sub-directories:

~~~
AFLWDataset/
├── ...
└── flickr
    └── images
~~~



##### Step 3: Convert original annotations into COCO format 

~~~
python convert2coco.py --dataset_root=<dataset_root>
~~~

where `<dataset_root>` is the directory where dataset has been extracted (e.g.,  `AFLWDataset/`). After conversion, a .json file will be stored under `<dataset_root>` as follows:

​	`aflw_annotations_train.json`



##### Optional step: dataset visualization 

For this purpose, [COCOAPI](https://github.com/cocodataset/cocoapi) needs to be installed first.



~~~
python visualize_dataset.py
~~~







[1] Koestinger, Martin, et al. "Annotated facial landmarks in the wild: A large-scale, real-world database for 
facial landmark localization." *2011 IEEE international conference on computer vision workshops (ICCV  workshops)*. IEEE, 2011.