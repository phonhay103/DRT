# DRT: A Lightweight Single Image Deraining Recursive Transformer
By Yuanchu Liang, Saeed Anwar, Yang Liu
College of Engineering and Computer Science, The Australian National University

![DRT Network Architecture](https://github.com/YC-Liang/DRT/blob/main/Images/Network.png)

## Evaluation
```
python test.py
```
or
```
python test.py --weights <model_weights> --input_dir <input_path> --result_dir <result_path>
```

<!--
To train from scratch, change the training directory in the `Load Data` section and set the last line of the notebook to
* `run(train_net = True, loadCkp = False, loadBest = False, new_dataset = False)`
-->