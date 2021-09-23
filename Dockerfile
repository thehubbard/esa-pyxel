#
# Pyxel with Jupyter notebook server
#

# Use LTS version
FROM ubuntu:20.04

# Set the timezone
ENV TZ=Europe/Amsterdam

ARG PYXEL_HOME="/home/pyxel"

RUN apt-get update --fix-missing \
    && apt-get install -y wget bzip2 git git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Pyxel source code
COPY . $PYXEL_HOME/src
RUN mkdir -p $PYXEL_HOME/jupyter

# Get Pyxel data code
RUN git lfs install
RUN git clone https://gitlab.com/esa/pyxel-data.git $PYXEL_HOME/jupyter/pyxel-data

# Add a new user (no need to run as roo)
RUN mkdir -p $PYXEL_HOME \
    && groupadd -g 999 pyxel \
    && useradd --shell=/bin/bash -r -u 999 -g pyxel pyxel \
    && chown -R pyxel:pyxel $PYXEL_HOME 

USER pyxel
WORKDIR $PYXEL_HOME

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -u -p ~/.local \
    && rm ~/miniconda.sh \
    && ~/.local/bin/conda clean -tipsy \
    && ~/.local/bin/conda init

# Make 'bash' the default shell
SHELL [ "/bin/bash", "--login", "-c" ]

# Make non-activate conda commands available
ENV PATH=~/.local/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". ~/.local/etc/profile.d/conda.sh" >> ~/.profile

# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

# Build the conda environment
RUN conda env create -f ~/src/environment.yml

# Install Pyxel from the source code
RUN conda activate pyxel-dev && \
    pip install -e ~/src && \
    conda deactivate

RUN conda activate pyxel-dev && \
    python -c "import pyxel; print('Pyxel version:', pyxel.__version__)" && \
    conda deactivate

# Install Jupyterlab extensions
#RUN conda activate pyxel-dev && \
#    jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
#    conda deactivate

# Add aliases
RUN echo "alias ll='ls -alF'" >> ~/.bashrc \
    && echo "alias ls='ls --color=auto'" >> ~/.bashrc

# Expose Jupyter notebook port
EXPOSE 8888
CMD conda activate pyxel-dev && \
    jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.quit_button=True --notebook-dir=~/jupyter 
