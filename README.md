
# What makes the difference? A Systematic Review on Multimodal Language Fusion Strategies

This repository includes SOTA modality fusion approaches for sentiment analysis and emotion recognition tasks.
All models have implemented in a unified PyTorch framework for conducting an empirical comparison across different fusion approaches.
Upon the acceptance of the paper, we will share the code for reproducing the results presented in the paper.

## Instructions to run the code

### Download the datasets

+ Monologue Datasets: https://www.dropbox.com/s/7z56hf9szw4f8m8/cmumosi_cmumosei_iemocap.zip?dl=0
  + Containing CMUMOSI, CMUMOSEI and IEMOCAP datasets
  + Each dataset has the CMU-SDK version and Multimodal-Transformer version (with different input dimensionalities)

### Do A Single Run (train/valid/test) 

1. Set up the configurations in config/run.ini
2. python run.py -config config/run.ini

#### Configuration setup
+ Monologue
  + **mode = run**
  + **dataset_type = multimodal**
  + **pickle_dir_path = /path/to/datasets/**. The absolute path of the folder storing the datasets.
  + **dataset_name in `{'cmumosei','cmumosi','iemocap'}`**. Name of the dataset.
  + **features in `{'acoustic', 'visual', 'textual'}`**. Multiple modality names should be joined by ','. 
  + **label in `{'sentiment','emotion'}`**. Multiple labels should be joined by ','. 
  + **wordvec_path**. The relative path of the pre-trained word embedding file.
  + **dialogue_format = False**. Disable the dialogue format.
  + **dialogue_context = False**. Disable the use of dialogue context.
  + **embedding_trainable in `{'True','False'}`**. Whether you want to train the word embedding for textual modality. Usually set to be True.
  + **case_study in `{'True','False'}`**. Whether you want to generate per-sample model predictions to files.
    + **model_prediction in `{'True','False'}`**. Whether model prediction for each sample will be exported to a file. Requires **case_study = True**.
    + **true_labels in `{'True','False'}`**. Whether true label for each sample will be exported to a file. Requires **case_study = True**.
    + **per_sample_analysis in `{'True','False'}`**. Whether true label + model prediction for each sample will be exported to a file. Requires **case_study = True**.
  + **seed**. The random seed for the experiment.
  + **load_model_from_dir in `{'True','False'}`**. Whether the model is loaded from a saved file.
    + **dir_name**. The directory storing the model configurations and model parameters. Requires **load_model_from_dir = True**.
  + **fine_tune in `{'True','False'}**. Whether you want to train the model with the data. 
  + **model specific parameters**. For running a model on the dataset, uncomment the respective area of the model and comment the areas for the other models. Please refer to the model implementations in /models/monologue/ for the meaning of each model specific parameter.
    + supported models include but are not limited to:
      + EF-LSTM
      + LF-LSTM
      + RMFN
      + TFN
      + LMF
      + MARN
      + Multimodal-Transformer (only for word-aligned data)
      + MMUU-BA
      + RAVEN (only for word-aligned data)
      + MFN

  
### Grid Search for the Best Parameters
1. Set up the configurations in config/grid_search.ini. Tweak a couple of fields in the single run configurations, as instructed below.
2. Write up the hyperparameter pool in config/grid_parameters/.
3. python run.py -config config/grid_search.ini

#### Configuration setup
+ **mode = run_grid_search**
+ **grid_parameters_file**. The name of file storing the parameters to be searched, under the folder /config/grid_parameters. 
  + the format of a file is:
    + [COMMON]
    + var_1 = val_1;val_2;val_3
    + var_2 = val_1;val_2;val_3
+ **search_times**. The number of times the program searches in the pool of parameters.
+ **output_file**.  The file storing the performances for each search in the pool of parameters. By default, it is eval/grid_search_`{dataset_name}`_`{network_type}`.csv






