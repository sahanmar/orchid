# The Orchid

The Orchid project is an active learning platform built around [Word-Level Coreference Resolution model](https://github.com/vdobrovolskii/wl-coref). 

## Table of contents 📚
1. Description
2. Preparation
3. Training
4. Evaluation

## Description 📖

The project covers several fields of research

1. State-of-art corefernce resolution (CR) results attempt by dimensionality reduction algorithms addition.
2. Training optimization by generalizing a problem as coreference resolution active learning problem.

The platform is designed in a way of easily substitutable buidling blocks, such as:

    - Encoder model as a module
    - Generalised CR model (Comming soon)
    - Uncertainty incorporation as a module
    - Dimensionality reduction algorithm as a module

## Preparation

Some parts of the section content were taken from [Word-Level CR](https://github.com/vdobrovolskii/wl-coref)

The following instruction has been tested with Python 3.7.

You will need:
* **OntoNotes 5.0 corpus** (download [here](https://catalog.ldc.upenn.edu/LDC2013T19), registration needed)
* **Python 2.7** to run conll-2012 scripts
* **Java runtime** to run [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.shtml)
* **Python 3.7+** to run the model
* **Perl** to run conll-2012 evaluation scripts
* **CUDA**-enabled machine (40 GB to train, 4 GB to evaluate)

### Data  💽

1. Extract OntoNotes 5.0 archive. In case it's in the repo's root directory:

        tar -xzvf ontonotes-release-5.0_LDC2013T19.tgz

2. Switch to Python 2.7 environment (where `python` would run 2.7 version). This is necessary for conll scripts to run correctly. To do it with with conda:

        conda create -y --name py27 python=2.7 && conda activate py27
3. Run the conll data preparation scripts (~30min):

        sh get_conll_data.sh ontonotes-release-5.0 data

4. Download conll scorers and Stanford Parser:

        sh get_third_party.sh

### Environemnt ⚙

1. Prepare your environment. To do it with conda:

        conda env create -f environment.yml

    **IMPORTANT!  ⬇️** If you provide training od AMD GPU 40gb, please provide the installation as: 

        conda env create -f environment_rci.yml
    
    The most crucial part here is the type of `pytorch` given the GPU. If for some reasons none of the environments works for you, try to find a more suitable `torch` version for yourself as it is done in [evnrinoment config](https://github.com/sahanmar/orchid/blob/main/environment.yml)
  
    When the installation is done, just activate the environment:

        conda activate orchid

2. Build the corpus in jsonlines format (~20 min):

        python convert_to_jsonlines.py data/conll-2012/ --out-dir data
        python convert_to_heads.py

You're all set!

## Training 🏋️

If you have completed all the steps in the previous section, then just run:

    python run.py train roberta

Use `-h` flag for more parameters and `CUDA_VISIBLE_DEVICES` environment variable to limit the cuda devices visible to the script. Refer to `config.toml` to modify existing model configurations or create your own.

### Evaluation

Make sure that you have successfully completed all steps of the [Preparation](#preparation) section.

1. [Download](https://www.dropbox.com/s/vf7zadyksgj40zu/roberta_%28e20_2021.05.02_01.16%29_release.pt?dl=0) and save the pretrained model to the `data` directory.

        https://www.dropbox.com/s/vf7zadyksgj40zu/roberta_%28e20_2021.05.02_01.16%29_release.pt?dl=0

2. Generate the conll-formatted output:

        python run.py eval roberta --data-split test

3. Run the conll-2012 scripts to obtain the metrics:

        python calculate_conll.py roberta test 20

### Slurm

Due to the limiting access to computational resources we were obligated to use [Slurm](https://slurm.schedmd.com/documentation.html). If you use Slurm too, you can use [rci_run.batch](https://github.com/sahanmar/orchid/blob/main/rci_run.batch) file and run the training as 

```
sbatch rci_run.batch
```
from orchid root directory.
