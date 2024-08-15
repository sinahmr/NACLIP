# NACLIP: Training-Free Open-Vocabulary Semantic Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2404.08181-b31b1b.svg)](https://arxiv.org/abs/2404.08181)

<div align="center">
<img src="./.assets/naclip-illustration.png" width="25%">
<p></p>
</div>

> **Abstract:** Despite the significant progress in deep learning for dense visual recognition problems, such as semantic segmentation, traditional methods are constrained by fixed class sets. Meanwhile, vision-language foundation models, such as CLIP, have showcased remarkable effectiveness in numerous zero-shot image-level tasks, owing to their robust generalizability. Recently, a body of work has investigated utilizing these models in open-vocabulary semantic segmentation (OVSS). However, existing approaches often rely on impractical supervised pre-training or access to additional pre-trained networks. In this work, we propose a strong baseline for training-free OVSS, termed Neighbour-Aware CLIP (NACLIP), representing a straightforward adaptation of CLIP tailored for this scenario. Our method enforces localization of patches in the self-attention of CLIP's vision transformer which, despite being crucial for dense prediction tasks, has been overlooked in the OVSS literature. By incorporating design choices favouring segmentation, our approach significantly improves performance without requiring additional data, auxiliary pre-trained networks, or extensive hyperparameter tuning, making it highly practical for real-world applications. Experiments are performed on 8 popular semantic segmentation benchmarks, yielding state-of-the-art performance on most scenarios.


## &#x1F3AC; Getting Started

### &#x1F4E6; Requirements

To run NACLIP, please install the following packages. We used `Python 3.9` in our experiments.

```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install openmim
mim install mmcv==2.0.1 mmengine==0.8.4 mmsegmentation==1.1.1
pip install ftfy regex yapf==0.40.1
```


### &#x1F5C2; Datasets
We include the listed dataset configurations in this repo, following [SCLIP](https://github.com/wangf3014/SCLIP): PASCAL VOC (with and without the background category), PASCAL Context (with and without the background category), Cityscapes, ADE20k, COCO-Stuff164k, and COCO-Object.

Please follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and pre-process the datasets. The COCO-Object dataset can be converted from COCO-Stuff164k by executing the following command:

```
python ./datasets/cvt_coco_object.py PATH_TO_COCO_STUFF164K -o PATH_TO_COCO_OBJECT
```

**Remember to modify the dataset paths (`data_root`) in the config files in** `./configs/`.


## &#x1F9EA; Evaluation
To evaluate our approach on a single benchmark, run the following command:
```
python eval.py --config ./configs/cfg_{benchmark_name}.py
```

You can also do the evaluation on all the benchmarks using the `test_all.sh` script, whose general syntax is:
```
bash test_all.sh {arch} {attn} {gaussian_std} {pamr} {gpu} {log_path}
```
Values of `reduced` for `{arch}`, and `naclip` for `{attn}` represent our method.
For example, to reproduce the main results, run:
```
bash test_all.sh reduced naclip 5 on {gpu} {log_path}
```


### &#x1F4CA; Results

With the default setup in this repo, the following results (mIoU) should be achieved:

<table>
<thead>
  <tr>
    <th>VOC21</th>
    <th>PC60</th>
    <th>COCO Obj</th>
    <th>VOC20</th>
    <th>Cityscapes</th>
    <th>PC59</th>
    <th>ADE20K</th>
    <th>COCO Stuff</th>
    <th>Avg</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>62.36</td>
    <td>34.99</td>
    <td>36.19</td>
    <td>80.60</td>
    <td>38.27</td>
    <td>38.35</td>
    <td>19.05</td>
    <td>25.18</td>
    <td>41.87</td>
  </tr>
</tbody>
</table>


## &#x1F64F; Acknowledgments

We gratefully thank the authors of [SCLIP](https://github.com/wangf3014/SCLIP), [CLIP](https://github.com/openai/CLIP), and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), on which our code is based. The illustration is generated using Google Gemini.


## &#x1F4DA; Citation

If you find this project useful, please consider citing:

```bibtex
@article{hajimiri2024naclip,
  title={Pay Attention to Your Neighbours: Training-Free Open-Vocabulary Semantic Segmentation},
  author={Hajimiri, Sina and Ben Ayed, Ismail and Dolz, Jose},
  year={2024}
  journal={arXiv preprint},
  url={https://arxiv.org/abs/2404.08181},
}
```