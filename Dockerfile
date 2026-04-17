FROM debian:bookworm
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user matching typical host UID
RUN useradd -m -u 1000 dev
USER dev

# Install uv for this user
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/dev/.local/bin:$PATH"

WORKDIR /app
CMD ["sleep", "infinity"]