-- Creates the read-only database role used exclusively by the AI Database Agent.
-- Run against the application database as a superuser/owner:
--   psql "$DATABASE_ADMIN_URL" -f scripts/create_readonly_role.sql
-- Idempotent. The role can SELECT whitelisted tables only; it cannot write,
-- create, or alter anything (verified by tests/security/test_database_agent.py).

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'orchestrator_ro') THEN
        CREATE ROLE orchestrator_ro LOGIN PASSWORD 'orchestrator_ro';
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO orchestrator_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO orchestrator_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO orchestrator_ro;
