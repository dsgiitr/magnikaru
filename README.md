# Magnikaru


## Create an environment
```
> conda create -n env_name python=3.12
> conda activate env_name
```

## install dependancies
```
> pip install -r req-torch.txt
```
If the Above doesn't work then install the packages manually

```
> pip install numpy matplotlib pandas scikit-learn tqdm chess
> flask flask_cors
```

Pytorch installation (Changes from time to time)
```
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> pip install lightning
```


## download the dataset
```
> import kagglehub
> path = kagglehub.dataset_download("datasnaek/chess")
> print("Path to dataset files:", path)
```

Make sure to change the path of the data file in ```config.py```

## General Instructions
1. All the necessary file paths and configuration including hyper paramters are to be set using the config.py file.

2. Run ```split.py``` to create ```train.csv``` and ```test.csv``` in their required paths

3. Run ```lightning_main.py``` to start training the model!

## NOTE:
1. Make sure that you have the right version of Cuda installed inside the environment. Use use the following commands to check for required informations!
```
> nvidia-smi 
```