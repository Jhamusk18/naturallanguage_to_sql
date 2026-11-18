#!/usr/bin/env bash
# Docker-free alternative to `make db-up` for Debian/Ubuntu: installs PostgreSQL
# if missing, creates the t2s role and shop database, and loads schema + seed.
# Idempotent: re-running drops and recreates the shop database.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v psql >/dev/null 2>&1; then
  sudo apt-get update -q
  sudo apt-get install -y -q postgresql
fi

sudo pg_ctlcluster "$(ls /usr/lib/postgresql | sort -V | tail -1)" main start 2>/dev/null || true

for _ in $(seq 1 15); do
  sudo -u postgres pg_isready -q && break
  sleep 1
done

sudo -u postgres psql -qtc "SELECT 1 FROM pg_roles WHERE rolname='t2s'" | grep -q 1 \
  || sudo -u postgres psql -qc "CREATE USER t2s PASSWORD 't2s'"
sudo -u postgres psql -qc "DROP DATABASE IF EXISTS shop"
sudo -u postgres psql -qc "CREATE DATABASE shop OWNER t2s"

PGPASSWORD=t2s psql -q -h localhost -U t2s -d shop \
  -f db/init/01_schema.sql -f db/init/02_seed.sql >/dev/null

echo "Ready: DATABASE_URL=postgresql://t2s:t2s@localhost:5432/shop"
