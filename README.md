## Pre-requisites
The project was developed using python 3 with the following packages.
- Pytorch
- Opencv
- Numpy
- Pytorch Image Models
- Pillow

1. Install [Pytorch](https://pytorch.org/get-started/locally/)
2. Install with pip:
```bash
pip install -r requirements.txt
```

## Datasets
- Rain 13k - Test: [Here](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs)
- Place it in `datasets`

## Evaluation
```bash
python test.py
```
or
```bash
python test.py --ckp_path <model_path> --in_path <input_folder> --out_path <results_path>
```

<!--
To train from scratch, change the training directory in the `Load Data` section and set the last line of the notebook to
* `run(train_net = True, loadCkp = False, loadBest = False, new_dataset = False)`
-->
