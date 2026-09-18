#! /bin/bash
blockMesh | tee log_blockMesh



rm -rf 0
cp -r 0.orig 0

setFields | tee log_setFields
decomposePar
mkdir spheres


#In yade parallel
#mpirun --allow-run-as-root -n 2 /home/zhiwar/yade-install/bin/yade-Unknown yade.py | tee log.txt

#For running on a single machine
mpirun --allow-run-as-root --mca oob_tcp_if_include lo -n 2 /home/zhiwar/yade-install/bin/yade-Unknown yade.py | tee log.txt
