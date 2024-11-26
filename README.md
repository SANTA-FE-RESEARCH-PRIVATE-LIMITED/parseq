# PARSeq: Scene Text Recognition with Permuted Autoregressive Sequence Models

## Installation

### General
```bash
# Update and upgrade the system
sudo apt-get update -y && apt-get upgrade -y
# Install python and its dependencies
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install -y software-properties-common build-essential
sudo apt install python3.12 python3.12-venv python3.12-dev
# Install pip and virtualenv
sudo apt install python3-pip
pip3 install --user --upgrade pip
pip3 install --user virtualenv
```

### CPU
- This is for inference only.
```bash
# Generate requirements files for specified PyTorch platform
make torch-cpu
# Install the project and core + train + test dependencies. Subsets: [dev,train,test,bench,tune]
pip install -r requirements/core.cpu.txt -e .[train,test,bench]
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
