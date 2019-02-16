# requires nvidia-docker v2
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

# install dependencies&conda and clean up afterwards to minimize image size
RUN apt-get -qq update && apt-get install -y --no-install-recommends curl bzip2 ca-certificates && \
    curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /opt/conda && \
    apt-get -qq -y remove curl bzip2 && apt-get -qq -y autoremove && apt-get -qq -y clean && \
    /opt/conda/bin/conda clean --all --y && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log /tmp/miniconda.sh

# create conda environment.yml and add it to the path (pull the environment name out of the environment.yml)
# inspired by https://medium.com/@chadlagore/conda-environments-with-docker-82cdc9d25754
ADD environment.yml /tmp/environment.yml
RUN /opt/conda/bin/conda env create -f /tmp/environment.yml &&  \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" >> ~/.bashrc

# Set the ENTRYPOINT to use bash
ENTRYPOINT [ "/bin/bash" ]