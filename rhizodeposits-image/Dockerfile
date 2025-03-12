FROM quay.io/fenicsproject/stable:latest
USER root
RUN apt-get -qq update && \
    apt-get -y upgrade && \
    apt-get -y install python3-pip && \
    apt-get clean
RUN pip3 install --upgrade pip && \
    pip3 install scipy==1.5.4 && \
    pip3 install scikit-learn && \
    pip3 install scikit-optimize && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
USER root
