# POS-BERT: Point Cloud One-Satge BERT Pre-training 

PyTorch implementation and pretrained models for POS-BERT.

<div align="center">
  <img width="100%" alt="Our framework" src="fig.png">
</div>

### Install
Please install [PyTorch](https://pytorch.org/). This codebase has been developed with python version 3.8.10, PyTorch version 1.8.0, CUDA 10.2 and torchvision 0.9.0.
```
conda create -n posbert python=3.8
conda activate posbert
# follow PyTorch website install torch=1.8.0
pip -r requirements.txt
cd install/pointnet2_ops_lib
python setup.py install
```

### Model weights
download the model weights from [Google Drive](https://drive.google.com/drive/folders/1bvd9w5RuPyuvzbzpWCxGWr2srcbrujfp?usp=sharing) and put it in the `weight` folder.

### Pretrain Eval
__experiment results link to Table1 (ET1)__
```angular2html
chmod a+x 1exp_pretrain_eval_svm.sh
./1exp_pretrain_eval_svm.sh
./run.sh
```
### Downstream Tasks
```angular2html
cd ../segmentation
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
If you find this repository useful, please consider giving a citation:
```
@article{eswa2023posbert,
  title={POS-BERT: Point Cloud One-Stage BERT Pre-Training},
  author={Kexue Fu, Peng Gao, Shaolei Liu, Linhao Qu, Longxiang Gao, Manning Wang},
  journal={Expert Systems With Applications},
  year={2023}
}
```
