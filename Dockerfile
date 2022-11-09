FROM ubuntu:22.04


# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential  && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh &&  /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch


ADD environment.yml /tmp/environment.yml
WORKDIR /tmp 
RUN conda env create -f environment.yml

ADD . /workspace
WORKDIR /workspace

# Make RUN commands use the new environment:
RUN echo "conda activate pytorch-stardist" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

WORKDIR /workspace/stardist_tools_
RUN python setup.py build_ext --help-compiler
RUN python setup.py install

WORKDIR /workspace