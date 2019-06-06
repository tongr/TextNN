#
# base system
#
ARG UBUNTU_VERSION=18.04
FROM ubuntu:${UBUNTU_VERSION} AS base

#
# conda support
#
FROM base AS miniconda
# inspired by continuumio/miniconda3
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV CONDA_VERSION=Miniconda3-4.6.14

# install dependencies&conda and clean up afterwards to minimize image size
RUN apt-get --quiet update --fix-missing && \
    apt-get --quiet --yes install --no-install-recommends curl bzip2 ca-certificates && \
    curl -sSL https://repo.anaconda.com/miniconda/${CONDA_VERSION}-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /opt/conda && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    apt-get --quiet --yes remove curl bzip2 && apt-get --quiet --yes autoremove && apt-get --quiet --yes clean && \
    /opt/conda/bin/conda clean --all --yes && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log /tmp/miniconda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc

# conda environment
FROM miniconda AS env
# create environment (use "base" environment)
ADD environment.yml /tmp/environment.yml
RUN /opt/conda/bin/conda env update --name base --file /tmp/environment.yml && \
    /opt/conda/bin/conda clean --all --yes

#
# code & repository
#
FROM env AS env-and-code
ADD . /code

#
# CUDA support
#
# requires nvidia-docker v2
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu${UBUNTU_VERSION} as gpu-base

# gpu + conda environment
FROM gpu-base as gpu-env
COPY --from=env /opt/conda/ /opt/conda/
COPY --from=env /root/.bashrc /root/.bashrc
RUN /opt/conda/bin/conda install tensorflow-gpu && /opt/conda/bin/conda clean --all --yes

# gpu + conda environment + code
FROM gpu-env AS gpu-env-and-code
COPY --from=env-and-code /code /code

CMD [ "/bin/bash" ]
