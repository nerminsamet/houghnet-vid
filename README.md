# HoughNet-VID: Spatial and temporal voting for video object detection

Official PyTroch implementation of HoughNet for video object detection. More details could be found in the paper:

> [**HoughNet: Integrating near and long-range evidence for visual detection**](https://arxiv.org/abs/2104.06773),            
> [Nermin Samet](https://nerminsamet.github.io/), [Samet Hicsonmez](https://giddyyupp.github.io/), [Emre Akbas](http://user.ceng.metu.edu.tr/~emre/),        
> *TPAMI, 2022. ([arXiv pre-print](https://arxiv.org/abs/2104.06773))*          

## Summary

The original HoughNet applies voting only in the spatial domain - for object detection in still images. 
We extended this idea to the temporal domain by developing a new method, which takes the difference of features from two frames, and 
applies spatial and temporal voting using our “temporal voting module” to detect objects. 
We showed the effectiveness of our method  on ILSVRC2015 dataset. 

## Video Object Detection Results on ILSVRC2015

| Method          | mAP | mAP_Fast | mAP_Medium | mAP_Slow |  
|:---------------:|:----------:|:----------:|:----------:|:----------:|
|HoughNet (single frame baseline)    | 68.8 | 45.8 | 66.1 | 79.1 |
|HoughNet-VID | 73.9 |     50.4 | 71.5 | 82.8|

Temporal voting model can be donwloaded [here](https://drive.google.com/file/d/1sMDug6hkR5jUyrEVV73GvxjiLtfq_zw1/view?usp=sharing).

## Installation

Please refer to  [installation instructions of HoughNet](https://github.com/nerminsamet/houghnet/blob/master/readme/INSTALL.md).


## Dataset Preparation

Please download ILSVRC2015 DET and ILSVRC2015 VID datasets from [here](http://image-net.org/challenges/LSVRC/2015/2015-downloads). 
Next, please place the data as the following. Alternatively you could also create symlink.


```
./data/ILSVRC2015/
./data/ILSVRC2015/Annotations/DET
./data/ILSVRC2015/Annotations/VID
./data/ILSVRC2015/Data/DET
./data/ILSVRC2015/Data/VID
./data/ILSVRC2015/ImageSets
```

We used train and test images from [MEGA](https://github.com/Scalsol/mega.pytorch). You could download the same splits,
as ImageSets [here](https://drive.google.com/drive/folders/1OrLHksrGBYPHVk5UlnmAaugGsvdAULRI?usp=sharing).

## Evaluation and Training

For evaluation and training please refer to [here](experiments/houghnet-vid.sh).

## Acknowledgement

This work was supported the Scientific and Technological Research Council of Turkey (TUBITAK) through the project titled "Object Detection in Videos with Deep Neural Networks" (grant number 117E054). The numerical calculations reported in this paper were partially performed at TUBITAK ULAKBIM,  High Performance and Grid Computing Center (TRUBA resources).

## License

HoughNet-VID is released under the MIT License (refer to the [LICENSE](LICENSE) file for details). 
 
## Citation

If you find HoughNet-VID useful for your research, please cite our paper as follows.

> N. Samet, S. Hicsonmez, E. Akbas, "HoughNet: Integrating near and long-range evidence for visual detection",
> arXiv, 2021.

BibTeX entry:
```
@misc{HoughNet2021,
      title={HoughNet: Integrating near and long-range evidence for visual detection}, 
      author={Nermin Samet and Samet Hicsonmez and Emre Akbas},
      year={2021}, 
}
```
