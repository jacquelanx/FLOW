# Tag: flow-bixbench-env:1.0
# Build: docker build -t flow-bixbench-env:1.0 -f docker/Dockerfile .
#
# This image is the ONLY place agent-written code executes. It pins a broad Python + R +
# Bioconductor bioinformatics stack so the agent can perform RNA-seq DE, flow cytometry,
# single-cell, and general analyses (NOTE: R stack doesn't work yet.) The persistent kernel 
# server is mounted at runtime (and copied here as a fallback) and is started by the host as 
# `docker run -i ... python /flow/kernel_server.py`.

FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    # Put the baked first-run scaffold on the path so the anchored template's
    # `import anchored` resolves in-container even when a project hasn't vendored its own copy.
    PYTHONPATH=/flow/firstrun

# --- System + R/Bioconductor toolchain --------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash coreutils procps \
        build-essential gfortran \
        r-base r-base-dev \
        libcurl4-openssl-dev libssl-dev libxml2-dev libfontconfig1-dev \
        libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Python scientific + bioinformatics stack -------------------------------------------
RUN pip install --upgrade pip && pip install \
        numpy pandas scipy scikit-learn \
        matplotlib seaborn statsmodels \
        anndata scanpy \
        fcsparser flowio flowkit \
        umap-learn leidenalg igraph \
        pyarrow \
        jupyter ipykernel nbclient nbformat

# --- R + Bioconductor packages (flow cytometry + RNA-seq DE) ----------------------------
# Installed best-effort; the agent can also install at runtime when network is enabled.
RUN R -e "install.packages(c('BiocManager','IRkernel'), repos='https://cloud.r-project.org')" \
    && R -e "IRkernel::installspec(user=FALSE)" \
    && R -e "BiocManager::install(c('flowCore','openCyto','CATALYST','FlowSOM','DESeq2','edgeR','limma'), ask=FALSE, update=FALSE)" || \
       echo "WARNING: some Bioconductor packages failed to build; the agent may install them at runtime."

# --- Non-root user + kernel server + first-run R pipeline --------------------------------
RUN useradd -m -u 1000 flow && mkdir -p /flow /work /data && chown -R flow:flow /flow /work
COPY src/flow/env/kernel_server.py /flow/kernel_server.py
COPY src/flow/firstrun/ /flow/firstrun/

USER flow
WORKDIR /work

# The host launches the kernel server explicitly; this CMD documents the contract.
CMD ["python", "/flow/kernel_server.py"]
