-- create ai_agent user
CREATE USER ai_agent WITH PASSWORD 'insecure-llm';

-- allow connection to the database
GRANT CONNECT ON DATABASE store TO ai_agent;

-- dedicated schema for the AI
CREATE SCHEMA ai_access;

-- ony allowed to see schema ai_access
GRANT USAGE ON SCHEMA ai_access TO ai_agent;
REVOKE USAGE ON SCHEMA public FROM ai_agent;

-- future object access only applied to ai_access schema
ALTER DEFAULT PRIVILEGES IN SCHEMA ai_access
-- only allow read
GRANT SELECT ON TABLES TO ai_agent;

-- enable DuckDB querying (vectorization)
GRANT USAGE ON SCHEMA duckdb TO ai_agent;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA duckdb TO ai_agent;

-- force the agent into read-only mode at the session level
ALTER ROLE ai_agent SET default_transaction_read_only = on;