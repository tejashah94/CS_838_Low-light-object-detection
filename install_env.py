import os
import pathlib


if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
elif not pathlib.Path('models').exists():
  !git clone --depth 1 https://github.com/tensorflow/models

%%bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.

%%bash 
cd models/research
pip install .
