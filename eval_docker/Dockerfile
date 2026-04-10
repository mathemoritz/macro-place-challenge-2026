FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /challenge

# Copy the challenge evaluation infrastructure
COPY macro_place/ /challenge/macro_place/
COPY pyproject.toml /challenge/pyproject.toml
COPY requirements.txt /challenge/requirements.txt

# Copy benchmarks (read-only at runtime, but needed in image)
COPY external/MacroPlacement/Testcases/ICCAD04/ /challenge/external/MacroPlacement/Testcases/ICCAD04/

# Install challenge package and base deps
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir scipy tqdm

# Pre-install common extra deps that submissions may need
# (cheaper to bake in than install at runtime)
RUN pip install --no-cache-dir cvxpy dccp numba

# Entrypoint: run the evaluator on a mounted placer file
# Usage: docker run ... -v /path/to/placer.py:/submission/placer.py evaluate /submission/placer.py --all
ENTRYPOINT ["python", "-m", "macro_place.evaluate"]
