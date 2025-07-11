FROM ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    nano \
    cmake \
    g++ \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*    
RUN pip install --no-cache-dir --upgrade pip wheel\
    && pip install --no-cache-dir \
        numpy==1.24.4\
        scipy \
        matplotlib \
        pygmsh \
        meshio \
        scikit-learn \
        scikit-optimize
USER root
