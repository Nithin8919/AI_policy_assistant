-- PostgreSQL Schema
-- Metrics table for storing educational data
CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    value NUMERIC,
    unit VARCHAR(50),
    district VARCHAR(100),
    academic_year VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bridge table for topic -> source mapping
CREATE TABLE bridge_table (
    id SERIAL PRIMARY KEY,
    topic VARCHAR(255) NOT NULL,
    source_id VARCHAR(255) NOT NULL,
    source_type VARCHAR(50),
    relevance_score NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_metrics_district ON metrics(district);
CREATE INDEX idx_bridge_topic ON bridge_table(topic);






