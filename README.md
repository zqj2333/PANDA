# PANDA
PANDA: Architecture-Level Power Evaluation by Unifying Analytical and Machine Learning Solutions

# SparseComm

## Setup Docker
```
docker pull zqj2333/spiderdocker:v5
docker volume create --name snap
docker run -itd --name spider --gpus=all -v snap zqj2333/spiderdocker:v5
docker exec -it spider /bin/bash
```

## Clone this repo.
```
git clone git@github.com:YukeWang96/spider-comm.git -b optimization
```

## Get dataset
```
cd spider-comm/optimization
bash getdataset.sh
bash generete_bin.sh
```

## Compile and Profile
```
cd spider-comm/optimization
bash X_XXXXX.sh
```

## Generate .csv
```
cd spider-comm/optimization
python generete_result.py
```

## Description of optimization
```
vector_optimization: vector_parallelism(multiple warps process an embedding), coarsening(rather than serial loop, a warp load multiple element of a vector and then multiple add)
edgebalance: rather than each GPU process the same number of node, each GPU process the same number of edge
edgebalance_in_device: each GPU process the same number of node, while each warp process the same number of edge
remotebalance: each GPU process the same number of remote access
combine: edgebalance+edgebalance_in_device
combine_coarsening: edgebalance+edgebalance_in_device+coarsening
combine_rowspatial: edgebalance+edgebalance_in_device+vector_parallelism
```
