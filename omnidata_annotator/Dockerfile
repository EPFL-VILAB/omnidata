FROM ubuntu:xenial

#############################
# Install Packages 
#############################

RUN apt-get update && \
	apt-get install -y \
		curl \
        wget \
        sudo \
		bzip2 \
		libfreetype6 \
		libgl1-mesa-dev \
		libglu1-mesa \
		libxi6 \
		libxrender1 \
        imagemagick \
        jp2a \
        && \
	apt-get -y autoremove && \
	rm -rf /var/lib/apt/lists/*


#############################
# Create conda environment
#############################

# Install miniconda
RUN wget --progress=bar:force:noscroll https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:$PATH"
RUN mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b

# Create conda environment
RUN conda init bash \
    && exec bash \
    && source ~/.bashrc \
    && . ~/.bashrc \
    && conda create --name test-env python=3.5 \
    && conda activate test-env 

#############################
# Install Blender
#############################
ENV BLENDER_MAJOR 2.79
ENV BLENDER_VERSION 2.79b
ENV BLENDER_BZ2_URL https://download.blender.org/release/Blender$BLENDER_MAJOR/blender-$BLENDER_VERSION-linux-glibc219-x86_64.tar.bz2

RUN mkdir /usr/local/blender && \
	curl -SL "$BLENDER_BZ2_URL" -o blender.tar.bz2 && \
	tar -jxvf blender.tar.bz2 -C /usr/local/blender --strip-components=1 && \
	rm blender.tar.bz2

RUN rm -rf /usr/local/blender/$BLENDER_MAJOR/python/lib/python3.5/site-packages/numpy

# Add packages to Blender python
RUN cd /usr/local/blender/$BLENDER_MAJOR/python/bin \
    && ./python3.5m -m ensurepip \
    && ./python3.5m -m pip install numpy --upgrade \
    && ./python3.5m -m pip install trimesh natsort==7.0.1 networkx==2.1 scipy==1.2.0 opencv-python==3.1.0 scikit-image transforms3d plyfile meshlabxml pytransform3d

#########################################
# Install MeshLab and Point Cloud Library
#########################################
 RUN apt-get update && apt-get install -y \
          cmake \
          libpcl-dev \
          libproj-dev \
          valgrind \
          meshlab \
          xvfb \
          ffmpeg \
      && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.2.so /usr/lib/libvtkproj4.so

RUN pip install pymeshlab==0.1.7

ENV PATH="/usr/local/blender:$PATH"


