<div align=center>
  <img src="overview.png" width="500px" />
</div>

# I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference

This repository contains an adjusted implementation of the implementations show in the paper
*["I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference"](https://arxiv.org/abs/2207.01405).*  
It is adapted and extended by Lionnus Kesting to run in true 8-bit, and using a framework that accepts **both individual I-ViT and I-BERT layer implementations** to find the optimal combination.

Below are instructions of PyTorch code to reproduce the accuracy results of quantization-aware training (QAT). [**TVM benchmark**](https://github.com/zkkli/I-ViT/tree/main/TVM_benchmark)
is the TVM deployment project for reproducing latency results.

## Installation

* TVM version is recommended to be 0.9.dev0, this is not included in the `requirements.txt`.

* **Clone and setup I-ViT**:

  ```bash
  git clone https://github.com/zkkli/I-ViT.git
  cd I-ViT
  ```

* **Create a Python virtual environment** and install dependencies:

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

## QAT Experiments

* **Quantization-aware training script** now supports Weights & Biases for experiment tracking.  Example commands:

  ```bash
  # Single-run training with W&B logging
  python quant_train.py \
    --model deit_tiny \
    --data /path/to/imagenet \
    --epochs 30 \
    --lr 5e-7 \
    --bitwidth 8 \
    --layer_type ivit \
    --wandb-project <YOUR-PROJECT> \
    --wandb-entity <YOUR-ENTITY>

  # Sweep example (grid over ivit/ibert and 8/16-bit)
  wandb sweep sweep.yaml
  wandb agent <YOUR-ENTITY>/<YOUR-PROJECT>/<SWEEP_ID>
  ```

* **Approximation implementations**: choose between `ivit` and `ibert` integer approximations for GELU, Softmax, and LayerNorm via:

  ```bash
  # individual flags
  --gelu ivit --softmax ibert --layernorm ivit

  # or shorthand to set all three at once
  --layer_type ibert
  ```

## Analysis Scripts

Under the `/scripts` directory you’ll find analysis notebooks and scripts that:

* Compare I-ViT (`ivit`) vs. I-BERT (`ibert`) integer approximations
* Benchmark against the float implementations of GELU, Softmax, and LayerNorm
* Generate visual reports of accuracy and error

## Results

TBD
<!-- Below are the Top-1 (%) accuracy results of our proposed I-ViT that you should get on ImageNet dataset.

|  Model |  FP32 | INT8 (I-ViT) | Diff. |
| :----: | :---: | :----------: | :---: |
|  ViT-S | 81.39 |     81.27    | -0.12 |
|  ViT-B | 84.53 |     84.76    | +0.23 |
| DeiT-T | 72.21 |     72.24    | +0.03 |
| DeiT-S | 79.85 |     80.12    | +0.27 |
| DeiT-B | 81.85 |     81.74    | -0.11 |
| Swin-T | 81.35 |     81.50    | +0.15 |
| Swin-S | 83.20 |     83.01    | -0.19 | -->

## Citation

The citation for I-ViT is:

```bibtex
@inproceedings{li2023vit,
  title={I-vit: Integer-only quantization for efficient vision transformer inference},
  author={Li, Zhikai and Gu, Qingyi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={17065--17075},
  year={2023}
}
```

The citation for I-BERT is:

```bibtext
@article{kim2021bert,
  title={I-BERT: Integer-only BERT Quantization},
  author={Kim, Sehoon and Gholami, Amir and Yao, Zhewei and Mahoney, Michael W and Keutzer, Kurt},
  journal={International Conference on Machine Learning (Accepted)},
  year={2021}
}
```
