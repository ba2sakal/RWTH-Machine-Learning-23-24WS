FROM rwthika/acdc-notebooks:latest

USER root

# Install all OS dependencies for fully functional notebook server
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    vim-tiny \
    git \
    inkscape \
    libsm6 \
    libxext-dev \
    libxrender1 \
    lmodern \
    netcat \
    openssh-client \
    # ---- nbconvert dependencies ----
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic \
    # ----
    tzdata \
    unzip \
    nano-tiny && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create alternative for nano -> nano-tiny
RUN update-alternatives --install /usr/bin/nano nano /bin/nano-tiny 10

RUN apt-get update && \
	apt install -y git

USER ${NB_USER}

RUN pip install \
	'jupyterlab_iframe==0.4.0' \
	'git+https://github.com/stv0g/nbgitpuller@f735265f7b2a429a17a8fab70cfd3557f060640d' \
	'rwth-nb==0.1.8' \
	'tqdm'

RUN conda install --name base --no-update-deps \
		'conda==4.10.3' && \
	conda install --name base --quiet --yes --no-update-deps \
		'jupyterlab-git==0.32.1' \
		'ujson==4.0.2' \
		'ipympl==0.9.3' \
		'matplotlib-base==3.4.3' && \
	conda clean --all


RUN conda install --name base --quiet --yes --no-update-deps \
		'jupyterlab-drawio==0.9.0' \
		'jupyterlab_iframe==0.4.0' \
		'jupyterlab_latex=3.0.0' \
		'jupyter_core==4.11.2' && \
	conda clean --all

RUN conda install --name base --quiet --yes --no-update-deps --channel conda-forge \
                'jupyterlab_widgets' \
		'jupyterhub=1.1.0' \
		'ipywidgets' && \
        conda clean --all

# Fix traitlets pip package
RUN pip uninstall -y traitlets
RUN pip install traitlets==5.9.0

# Workaround LaTex
#RUN git clone https://github.com/joequant/jupyterlab-latex /tmp/jupyterlab-latex && cd /tmp/jupyterlab-latex && git checkout cbb66825786ecf11a35fd92df797f8ccee719ad4 && pip install -ve .
#RUN jupyter lab build


USER root

# Use RWTH Mirror
#RUN sed -i 's|http://archive.ubuntu.com|http://ftp.halifax.rwth-aachen.de|g' /etc/apt/sources.list

RUN apt-get update && \
	apt-get -y install \
		language-pack-de \
		texlive-latex-recommended \
		openssh-client && \
	rm -rf /var/lib/apt/lists/*

# Add more locales
RUN locale-gen de_DE && \
	locale-gen de_DE.UTF-8 && \
	update-locale

USER ${NB_USER}

ENV JUPYTER_ENABLE_LAB=yes

# Until HERE RWTH BASE IMAGE 

# Install cv2
USER root

# Install Essentials + cv2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        cmake \
        build-essential \
        curl \
        wget \
        gnupg2 \
        lsb-release \
        ca-certificates \
        python3-opencv \
    && rm -rf /var/lib/apt/lists/*




RUN pip install jupyterlab-urdf==0.2.1
RUN pip install jupyterlab-zethus
RUN pip install torch torchvision


ENV ROS_HOSTNAME=localhost
ENV ROS_MASTER_URI=http://localhost:11311
