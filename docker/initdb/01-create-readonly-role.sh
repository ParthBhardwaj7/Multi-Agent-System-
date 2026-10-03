#!/bin/bash
# Creates the read-only role used by the AI Database Agent (runs once at first
# postgres startup via docker-entrypoint-initdb.d).
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'orchestrator_ro') THEN
            CREATE ROLE orchestrator_ro LOGIN PASSWORD '${POSTGRES_RO_PASSWORD:-orchestrator_ro}';
        END IF;
    END
    \$\$;
    GRANT USAGE ON SCHEMA public TO orchestrator_ro;
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO orchestrator_ro;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO orchestrator_ro;
EOSQL

echo "Read-only role orchestrator_ro ready."
