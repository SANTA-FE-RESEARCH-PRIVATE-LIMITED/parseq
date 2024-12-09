# PARSeq: Scene Text Recognition with Permuted Autoregressive Sequence Models

## Installation

### System
- OS : Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- GPU : 
  - Nvidia L4 (24GB VRAM)
  - CUDA Toolkit : v11.8 (11.8.89)
  - Driver : v535 (535.104.05)
- vCPU : 25 x Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz
- RAM : 110GB

### General
- Update and upgrade the system
```bash
apt-get update -y && apt-get upgrade -y
```
- Install python and its dependencies
```bash
add-apt-repository ppa:deadsnakes/ppa
apt-get install -y software-properties-common build-essential
apt install python3.12 python3.12-venv python3.12-dev
```
- Install pip and virtualenv
```bash
apt install python3-pip
pip3 install --user --upgrade pip
pip3 install --user virtualenv
```
- Create a virtual environment and activate it.
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### CPU
- This is for inference only.
```bash
# Generate requirements files for specified PyTorch platform
make torch-cpu
# Install the project and core + train + test dependencies. Subsets: [dev,train,test,bench,tune]
pip install -r requirements/core.cpu.txt -e .[test,bench]
```

### GPU
```bash
# Generate requirements files for specified PyTorch platform
make torch-cu118
# Install the project and core + train + test dependencies. Subsets: [dev,train,test,bench,tune]
pip install -r requirements/core.cu118.txt -e .[train,test,bench,tune]
```

### Datasets
- Make a directory for the datasets.
```bash
mkdir -p data
```
- Download the [datasets](Datasets.md) from the following links:
    - [LMDB archives](https://drive.google.com/drive/folders/1NYuoi7dfJVgo-zUJogh8UQZgIMpLviOE) for MJSynth, SynthText, IIIT5k, SVT, SVTP, IC13, IC15, CUTE80, ArT, RCTW17, ReCTS, LSVT, MLT19, COCO-Text, and Uber-Text.
    - [LMDB archives](https://drive.google.com/drive/folders/1D9z_YJVa6f-O0juni-yG5jcwnhvYw-qC) for TextOCR and OpenVINO.
- Download the archives to the `data/` directory.
- Follow the respective `README.md` files for extraction.

## Sanity Check
- Make directory for storing results of the checks.
```bash
mkdir -p logs
```
### CPU
- Run `read.py` to check the single image inference for installation.
```bash
python -u -W ignore read.py pretrained=parseq --images demo_images/* --device=cpu 2>&1 | tee logs/read-cpu.log
```
- Run `test.py` to generate metrics for test split.
```bash
python -u -W ignore test.py pretrained=parseq --device cpu 2>&1 | tee logs/test-cpu.log
```
- Run `bench.py` to benchmark the forward pass.
```bash
python -u -W ignore bench.py model=parseq model.decode_ar=false model.refine_iters=3 device=cpu 2>&1 | tee logs/bench-cpu.log
```

### GPU
- Run `read.py` to check the single image inference for installation.
```bash
python -u -W ignore read.py pretrained=parseq --images demo_images/* --device=cuda 2>&1 | tee logs/read-gpu.log
```
- Run `test.py` to generate metrics for test split.
```bash
python -u -W ignore test.py pretrained=parseq --device cuda 2>&1 | tee logs/test-gpu.log
```
- Run `bench.py` to benchmark the forward pass.
```bash
python -u -W ignore bench.py model=parseq model.decode_ar=false model.refine_iters=3 device=cuda 2>&1 | tee logs/bench-gpu.log
```

## Usage

### Training
- Create directory for storing logs.
```bash
mkdir -p logs/trial
```
- Run `train.py` to train the model.
```bash
./train.py --config-name main-hindi 2>&1 | tee logs/trial/run_01.log
```
- The output are stored in `outputs/parseq/` directory.

## Citation
```bibtex
@InProceedings{bautista2022parseq,
  title={Scene Text Recognition with Permuted Autoregressive Sequence Models},
  author={Bautista, Darwin and Atienza, Rowel},
  booktitle={European Conference on Computer Vision},
  pages={178--196},
  month={10},
  year={2022},
  publisher={Springer Nature Switzerland},
  address={Cham},
  doi={10.1007/978-3-031-19815-1_11},
  url={https://doi.org/10.1007/978-3-031-19815-1_11}
}
```
