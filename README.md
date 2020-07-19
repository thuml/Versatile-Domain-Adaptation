# MCC
Code Release for "Minimum Class Confusion for Versatile Domain Adaptation"(ECCV2020)
## Dataset
### Office-31
Office-31 dataset can be found [here](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/).

### Office-Home
Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).

### VisDA-2017
VisDA 2017 dataset can be found [here](https://github.com/VisionLearningGroup/taskcv-2017-public) in the classification track.

## Training
### Unsupervised DA
```
cd pytorch
python train_image_office.py --gpu_id 0 --net ResNet50 --dset office --s_dset_path ../data/office/amazon_list.txt --t_dset_path ../data/office/webcam_list.txt --output_dir mcc-uda-office
```
## Contact
If you have any problem about our code, feel free to contact jiny18@mails.tsinghua.edu.cn
or describe your problem in Issues.