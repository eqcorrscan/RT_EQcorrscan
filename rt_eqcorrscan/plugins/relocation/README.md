# Relocator plugin help

This document outlines the requirements for the relocator plugin. To use this
plugin you will need to install several additional tools and configure them
seperately. You will also need to put some control files in your local folder
described below.

The relocator contains three primary steps:
1. Compute initial absolute locations of detections;
2. Compute cross-correlation differential times for all earthquakes detected;
3. Compute relative relocations of all earthquakes detected.

As of relocator version 0.0.1 (July 2023), step 1 uses the **hypocentre** program
installed via seisan, and step 3 uses the growclust3d.jl julia code. You will
need to install seisan and growclust3d.jl yourself.

## Install seisan

Code below tested for docker image creation on miniforge docker image (ubuntu)

```bash
apt-get update && apt-get install -y curl tar gfortran libx11-6 libx11-dev g++ make && \
    apt-get clean && \
    curl "https://www.geo.uib.no/seismo/SOFTWARE/SEISAN/seisan_v12.0_linux_64.tar.gz" -o "seisan_v12_linux_64.tar.gz" && \
    mkdir seisan && mv seisan_v12_linux_64.tar.gz seisan/. && cd seisan && \
    tar -xzf seisan_v12_linux_64.tar.gz && \
    cd PRO && make clean && \
    export SEISARCH=linux64 && make all

# Install SEISAN
# RUN sed -i 's/\/home\/s2000\/seismo/\/seisan/g' /seisan/COM/SEISAN.bash && \
#     echo "source /seisan/COM/SEISAN.bash" >> ~/.bashrc
cp -v /seisan/PRO/* /usr/local/bin/. && cp -v /seisan/COM/* /usr/local/bin/.
```

## Install GrowClust3d.jl

1. Install julia using your preferred package manager
2. Start juilia repl: `julia`
3. Install GrowClust3d.jl: ` pkg> add https://github.com/dttrugman/GrowClust3D.jl`
4. Test GrowClust3d: `pkg> test GrowClust3D`

Code below to install on miniforge docker image (ubuntu)

```bash
julia -e 'import Pkg; Pkg.update()'
julia -e 'import Pkg; Pkg.add("DataFrames")'
julia -e 'import Pkg; Pkg.add("Proj")'
julia -e 'import Pkg; Pkg.add(url="https://github.com/dttrugman/GrowClust3D.jl"); using GrowClust3D'

```