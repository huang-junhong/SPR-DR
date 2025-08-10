# SPR-DR
The Official implementation of SPR-DR

# Quick Test Model

* Clone this repo.
```bash
git clone https://github.com/huang-junhong/SPR-DR.git
cd SPR-DR
```

* Install dependencies. (Suggest python 3.12 + CUDA 12.4 in Anaconda)
```bash
pip install -r requirements.txt
```

* Download pretrain model & test-set  
Download our pre-trained SPR-DR models and part comparison models from [here][pretrain-model].  
Download BSD100 testSet from [here][bsd100]  
Download PIRM testset from [here][pirm]  
Download General-100 testset from [here][g100]  
Download Urban-100 testset form [here][u100]  
Download DIV2K-Valid testset from [here][div2k]  
Download Manga-109 testset from [here][manga109]

* Run test script  
One example is:
```bash
python test_model.py \
  --model_path ./model/SPR-DR-SRF4.pth \
  --model_type SRRes \
  --SRF 4 \
  --lr_folder ./datasets/BSD100/LR_x4 \
  --hr_folder ./datasets/BSD100/HR \
  --save_folder ./results/SPR-DR-SRF4 \
  --IQAs psnr ssim lpips dists
```
The IQA calculate by [pytorch-iqa][pyiqa]

# Ranker convolution kern effect validation (RSGD validation)
We provide a proctype code to valuate 'effectless' kenr by RSGD in test_RSGD.py. This code not clean up, so all the model and data path is abs-path in our server, so you may change it to yours.
We will update this code for on 2025.8.26.

# Train Model
Coming soon


[pretrain-model]: https://1drv.ms/f/c/c961ef6a7e95bfe2/EqKlfNuFjCJCgOru9AzYAl4BL4w30N2EzFHn9JKNrzsF9g
[bsd100]: https://huggingface.co/datasets/eugenesiow/BSD100
[pirm]: https://pirm.github.io/
[g100]: https://huggingface.co/datasets/goodfellowliu/General100
[u100]: https://www.kaggle.com/datasets/harshraone/urban100
[div2k]: https://data.vision.ee.ethz.ch/cvl/DIV2K/
[manga109]: http://www.manga109.org/en/
[pyiqa]: https://github.com/chaofengc/IQA-PyTorch
