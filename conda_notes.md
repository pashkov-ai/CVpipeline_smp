`conda create -n CVpipeline-smp python=3.11 && conda activate CVpipeline-smp`
`conda install pre-commit pytest`
`conda install pytorch pytorch-cuda=12.4 pytorch-lightning segmentation-models-pytorch albumentations hydra-core mlflow-skinny -c pytorch`


`conda deactivate && conda env remove -n CVpipeline-smp`
`conda env export --no-builds | grep -v "^prefix:" > environment.nobuilds.yaml`
`conda env create -f environment.nobuilds.yaml`