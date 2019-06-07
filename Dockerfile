##
## base image: Ubuntu 18.04
##
FROM ubuntu:18.04 AS base


#
# conda support
#
FROM base AS conda
ARG CONDA_INSTALLER_URL=https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh
# download conda installer
ADD ${CONDA_INSTALLER_URL} /tmp/install-conda.sh
# inspired by continuumio/miniconda3
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH="/opt/conda/bin:$PATH"
# install dependencies&conda and clean up afterwards to minimize image size
RUN apt-get --quiet update --fix-missing && \
    apt-get --quiet --yes install --no-install-recommends ca-certificates && \
    bash /tmp/install-conda.sh -bfp /opt/conda && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    apt-get --quiet --yes autoremove && apt-get --quiet --yes clean && \
    /opt/conda/bin/conda clean --all --yes && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log /tmp/install-conda.sh

#
# conda environment
#
FROM conda AS env
# create environment (use "base" environment)
ADD environment.yml /tmp/environment.yml
RUN /opt/conda/bin/conda env update --name base --file /tmp/environment.yml && \
    /opt/conda/bin/conda clean --all --yes

#
# code & repository
#
FROM env AS env-and-code
ADD . /code

##
## base image w/ CUDA support (18.04)
##
# requires nvidia-docker v2
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04 as gpu-base

#
# gpu + conda environment
#
FROM gpu-base as gpu-env
# activate conda binaries
COPY --from=env /opt/conda/ /opt/conda/
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH="/opt/conda/bin:$PATH"
# install additional tensorflow-gpu package and activate conda command in bash
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    /opt/conda/bin/conda install --yes tensorflow-gpu && /opt/conda/bin/conda clean --all --yes

#
# gpu + conda environment + code
#
FROM gpu-env AS gpu-env-and-code
COPY --from=env-and-code /code /code

CMD [ "/bin/bash" ]
