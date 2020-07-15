#
# Pyxel with Jupyter notebook server
#

FROM ubuntu:20.04

# Set the timezone
ENV TZ=Europe/Amsterdam

ARG PYXEL_HOME="/home/pyxel"

RUN apt-get update --fix-missing \
    && apt-get install -y wget bzip2 git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Pyxel source code
COPY . $PYXEL_HOME/src
COPY examples $PYXEL_HOME/notebooks/examples
COPY data $PYXEL_HOME/notebooks/data

# Add a new user (no need to run as roo)
RUN mkdir -p $PYXEL_HOME \
    && groupadd -g 999 pyxel \
    && useradd --shell=/bin/bash -r -u 999 -g pyxel pyxel \
    && chown -R pyxel:pyxel $PYXEL_HOME 

USER pyxel
WORKDIR $PYXEL_HOME

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -u -p ~/.local \
    && rm ~/miniconda.sh \
    && ~/.local/bin/conda clean -tipsy \
    && ~/.local/bin/conda init

# Install Pyxel environment
RUN ~/.local/bin/conda env create -f ~/src/environment.yml

RUN echo "conda activate pyxel-dev" >> ~/.bashrc \ 
    && echo "alias ll='ls -alF'" >> ~/.bashrc \
    && echo "alias ls='ls --color=auto'" >> ~/.bashrc

# Install Pyxel from the source code
RUN ~/.local/envs/pyxel-dev/bin/python -m pip install -e ~/src

#RUN ~/.local/envs/pyxel-dev/bin/python -c "import pyxel; print('Pyxel version:', pyxel.__version__)"

# Expose Jupyter notebook port
EXPOSE 8888

CMD ~/.local/envs/pyxel-dev/bin/jupyter notebook --ip=0.0.0.0 --no-browser --notebook-dir=~/notebooks --NotebookApp.quit_button=True
