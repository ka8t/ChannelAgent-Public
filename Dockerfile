# ChannelAgent runtime image.
#
# python:3.14-slim: the same interpreter as the native virtualenv (start.sh --native),
# CI and the dependency locks, so a test run here and one in the image agree.
FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
# Alembic migrations — app/db/session.py's init_db() runs these
# at startup, so both must be present in the image, not just alembic
# the Python package.
COPY alembic.ini .
COPY alembic/ ./alembic/

# The application does not run as root. Fixed uid/gid 10001, so the
# host-side ownership of a bind-mounted data directory is predictable:
# `chown -R 10001:10001 data` on a Linux host. The
# code stays owned by root and read-only to the application.
RUN groupadd --system --gid 10001 channelagent \
    && useradd --system --uid 10001 --gid 10001 --no-create-home \
        --shell /usr/sbin/nologin channelagent

# Real data (SQLite file) lives here — mount a volume over it in
# docker-compose.yml so it survives container recreation. An empty named
# volume takes this directory's ownership on first use.
RUN mkdir -p /app/data && chown 10001:10001 /app/data

# Inside the container the Admin API must listen on every interface or the
# published port cannot reach it. What is exposed to the outside is decided
# by the port mapping in docker-compose.yml (loopback by default).
ENV API_SERVER_HOST=0.0.0.0

USER 10001:10001

# Healthy while the application's heartbeat file is recent (app/health.py);
# no port involved, so it works with or without the Admin API. The start period
# covers the database migration at startup.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-m", "app.health"]

CMD ["python", "-m", "app.main"]
