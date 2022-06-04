## Datasets
- Rain 13k - Test: [Here](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs)
- Place it in `datasets`

## Evaluation
```
python test.py
```
or
```
python test.py --ckp_path <model_path> --in_path <input_folder> --out_path <results_path>
```

<!--
To train from scratch, change the training directory in the `Load Data` section and set the last line of the notebook to
* `run(train_net = True, loadCkp = False, loadBest = False, new_dataset = False)`
-->
