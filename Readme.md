# th23_hmd_repository

## 1. Make Container.
docker run --gpus all -it --name Container_Name -e LC_ALL=C.UTF-8 --ipc=host -v /Parent_path/hmd2/notebooks_th:/Data1 -v /Datapath:/Data2 -v /var/lib/docker:/Files pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel /bin/bash

## 2. Start Container.
docker start Container_Name  

docker attach Container_Name

## 3. Run bash file for Experiment settings.
bash hmd_bash.sh

### NOTE 
#### 1. For understanding Heart Murmur Detection Task, note file /Data1/notes/EDA.ipynb
#### 2. To use trained model, note /Data1/notes/Inference_Example.ipynb