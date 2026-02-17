FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install K2DO package
COPY pyproject.toml README.md LICENSE ./
COPY k2do/ k2do/
RUN uv pip install --system --no-cache .

# Runtime directories
RUN mkdir -p /root/.k2do/workspace

# Health endpoint / gateway port
EXPOSE 18790

ENTRYPOINT ["k2do"]
CMD ["status"]
