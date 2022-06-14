# AI_final_OCR
## NYCU Intro_to_AI
Most code is cited from [crnn-pytorch](https://github.com/GitYCC/crnn-pytorch).
## Purpose
This repo is trying to findout the resistence of CRNN model to handle OCR under various image conditions.
## Usage
The basic usage is in [crnn-pytorch](https://github.com/GitYCC/crnn-pytorch).
First of all, install dependency packages.
```command
$ pip install -r requirements.txt
```
### Download Synth90k dataset
Be aware that this process will cost more than one day to complete.
```command
$ cd data && chmod +x download_synth90k.sh && bash download_synth90k.sh
```

```
@InProceedings{Jaderberg14c,
  author       = "Max Jaderberg and Karen Simonyan and Andrea Vedaldi and Andrew Zisserman",
  title        = "Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition",
  booktitle    = "Workshop on Deep Learning, NIPS",
  year         = "2014",
}

@Article{Jaderberg16,
  author       = "Max Jaderberg and Karen Simonyan and Andrea Vedaldi and Andrew Zisserman",
  title        = "Reading Text in the Wild with Convolutional Neural Networks",
  journal      = "International Journal of Computer Vision",
  number       = "1",
  volume       = "116",
  pages        = "1--20",
  month        = "jan",
  year         = "2016",
}
```
### Train
Then, modify hyper-parameters, dataset settings to control training workflow in config.py.
Read comments in the file to understand the meanings of parameters.
If all parameters are set, run train.py to train a model.
```command
$ python train.py
```
### Evaluate
Once the training process is done, add the checkpoint directory to lookup directory in config.py.
Run test_dirs.py to test all pt files under lookup_dirs you set.
The evaluation result (loss, accuracy, average cer and wrong cases) will be stored under the same directory with name set in config.py.
```command
$ python test_dirs.py
```
### Collect Results
If you prefer to gather all evaluation result to have a quick summary or plot line graphs, run summarize.py.
```command
$ python summarize.py "{prefix of the evaluation result files to include}"
```
### Analyze results
Plot the graph of training models.
```command
$ python plot.py 
```
Explore wrong cases.
```command
$ python convert_wrong_cases.py {path to an evaluation result file} {path to annotation text file used}
```

