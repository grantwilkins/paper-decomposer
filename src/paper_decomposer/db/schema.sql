-- paper_decomposer schema
-- Postgres + pgvector + pg_trgm. Designed for hundreds of thousands of papers.
-- Methods and settings are DAGs (multi-parent allowed); a canonical_parent_id
-- on each node selects the preferred path for tree-style display.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;

-- ─── papers ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS papers (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    arxiv_id           text UNIQUE,
    doi                text UNIQUE,
    title              text NOT NULL,
    authors            jsonb NOT NULL DEFAULT '[]'::jsonb,
    year               int,
    venue              text,
    source_pdf_sha256  text,
    ingested_at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS papers_year_idx ON papers (year);

-- ─── extraction runs and evidence ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS extraction_runs (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id     uuid NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    started_at   timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    metadata     jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS extraction_runs_paper_idx ON extraction_runs (paper_id);

CREATE TABLE IF NOT EXISTS evidence_spans (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id          uuid NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    extraction_run_id uuid NOT NULL REFERENCES extraction_runs(id) ON DELETE CASCADE,
    local_span_id     text NOT NULL,
    section_title     text NOT NULL,
    section_kind      text NOT NULL,
    page_start        int,
    page_end          int,
    artifact_id       text,
    source_kind       text NOT NULL,
    text              text NOT NULL,
    UNIQUE (extraction_run_id, local_span_id)
);

CREATE INDEX IF NOT EXISTS evidence_spans_paper_idx ON evidence_spans (paper_id);
CREATE INDEX IF NOT EXISTS evidence_spans_local_idx ON evidence_spans (extraction_run_id, local_span_id);

-- ─── methods (DAG nodes) ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS methods (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id            uuid REFERENCES papers(id) ON DELETE CASCADE,
    extraction_run_id   uuid REFERENCES extraction_runs(id) ON DELETE CASCADE,
    local_node_id       text,
    canonical_name      text NOT NULL,
    description         text,
    canonical_parent_id uuid REFERENCES methods(id) ON DELETE SET NULL,
    metadata            jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding           vector(1536),
    UNIQUE (extraction_run_id, local_node_id)
);

CREATE INDEX IF NOT EXISTS methods_name_trgm_idx ON methods USING gin (canonical_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS methods_canonical_parent_idx ON methods (canonical_parent_id);
CREATE INDEX IF NOT EXISTS methods_embedding_hnsw_idx
    ON methods USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS method_aliases (
    method_id  uuid NOT NULL REFERENCES methods(id) ON DELETE CASCADE,
    alias      text NOT NULL,
    source_paper_id uuid REFERENCES papers(id) ON DELETE SET NULL,
    PRIMARY KEY (method_id, alias)
);

CREATE INDEX IF NOT EXISTS method_aliases_alias_trgm_idx
    ON method_aliases USING gin (alias gin_trgm_ops);

-- DAG edges. relation in {is_a, specializes, composes, refines}.
CREATE TABLE IF NOT EXISTS method_edges (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_id   uuid NOT NULL REFERENCES methods(id) ON DELETE CASCADE,
    child_id    uuid NOT NULL REFERENCES methods(id) ON DELETE CASCADE,
    relation    text NOT NULL CHECK (relation IN ('is_a', 'specializes', 'composes', 'refines')),
    confidence  double precision NOT NULL DEFAULT 1.0,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (parent_id, child_id, relation),
    CHECK (parent_id <> child_id)
);

CREATE INDEX IF NOT EXISTS method_edges_child_idx ON method_edges (child_id);

-- ─── settings (datasets / tasks / applications / workloads / hardware) ───
CREATE TABLE IF NOT EXISTS settings (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id            uuid REFERENCES papers(id) ON DELETE CASCADE,
    extraction_run_id   uuid REFERENCES extraction_runs(id) ON DELETE CASCADE,
    local_setting_id    text,
    kind                text NOT NULL CHECK (kind IN ('dataset','task','application','workload','hardware','model_artifact','metric')),
    canonical_name      text NOT NULL,
    description         text,
    canonical_parent_id uuid REFERENCES settings(id) ON DELETE SET NULL,
    metadata            jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding           vector(1536),
    UNIQUE (extraction_run_id, local_setting_id)
);

CREATE INDEX IF NOT EXISTS settings_name_trgm_idx ON settings USING gin (canonical_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS settings_canonical_parent_idx ON settings (canonical_parent_id);
CREATE INDEX IF NOT EXISTS settings_embedding_hnsw_idx
    ON settings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS setting_edges (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_id   uuid NOT NULL REFERENCES settings(id) ON DELETE CASCADE,
    child_id    uuid NOT NULL REFERENCES settings(id) ON DELETE CASCADE,
    relation    text NOT NULL CHECK (relation IN ('is_a', 'specializes', 'composes', 'refines')),
    confidence  double precision NOT NULL DEFAULT 1.0,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (parent_id, child_id, relation),
    CHECK (parent_id <> child_id)
);

CREATE INDEX IF NOT EXISTS setting_edges_child_idx ON setting_edges (child_id);

-- Cross-family method → setting applicability. Do not encode these links in
-- method_edges, which is reserved for method-to-method DAG structure.
CREATE TABLE IF NOT EXISTS method_setting_links (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id    uuid NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    method_id   uuid NOT NULL REFERENCES methods(id) ON DELETE CASCADE,
    setting_id  uuid NOT NULL REFERENCES settings(id) ON DELETE CASCADE,
    relation    text NOT NULL CHECK (relation IN ('applies_to','evaluated_on')),
    confidence  double precision NOT NULL DEFAULT 1.0,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (paper_id, method_id, setting_id, relation)
);

CREATE INDEX IF NOT EXISTS method_setting_links_method_idx ON method_setting_links (method_id);
CREATE INDEX IF NOT EXISTS method_setting_links_setting_idx ON method_setting_links (setting_id);

-- ─── outcomes (metrics, deltas) ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS outcomes (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id            uuid NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    method_id           uuid REFERENCES methods(id) ON DELETE SET NULL,
    setting_id          uuid REFERENCES settings(id) ON DELETE SET NULL,
    metric_name         text NOT NULL,
    value               double precision,
    delta_value         double precision,
    baseline_method_id  uuid REFERENCES methods(id) ON DELETE SET NULL,
    units               text,
    metadata            jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS outcomes_paper_idx ON outcomes (paper_id);
CREATE INDEX IF NOT EXISTS outcomes_method_idx ON outcomes (method_id);
CREATE INDEX IF NOT EXISTS outcomes_setting_idx ON outcomes (setting_id);
CREATE INDEX IF NOT EXISTS outcomes_metric_name_idx ON outcomes (metric_name);

-- ─── claims ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS claims (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id    uuid NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    claim_type  text NOT NULL,
    strength    double precision,
    statement   text NOT NULL,
    embedding   vector(1536),
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS claims_paper_idx ON claims (paper_id);
CREATE INDEX IF NOT EXISTS claims_type_idx ON claims (claim_type);
CREATE INDEX IF NOT EXISTS claims_embedding_hnsw_idx
    ON claims USING hnsw (embedding vector_cosine_ops);

-- Many-to-many link from claim → (method | setting | outcome).
CREATE TABLE IF NOT EXISTS claim_links (
    claim_id     uuid NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    target_kind  text NOT NULL CHECK (target_kind IN ('method','setting','outcome')),
    target_id    uuid NOT NULL,
    PRIMARY KEY (claim_id, target_kind, target_id)
);

CREATE INDEX IF NOT EXISTS claim_links_target_idx ON claim_links (target_kind, target_id);

-- Per-claim provenance back to the source PDF section/artifact.
CREATE TABLE IF NOT EXISTS claim_evidence (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id      uuid NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    section_path  text,
    page          int,
    artifact_id   text
);

CREATE INDEX IF NOT EXISTS claim_evidence_claim_idx ON claim_evidence (claim_id);

CREATE TABLE IF NOT EXISTS evidence_links (
    evidence_span_id uuid NOT NULL REFERENCES evidence_spans(id) ON DELETE CASCADE,
    target_kind      text NOT NULL CHECK (
        target_kind IN (
            'method',
            'method_edge',
            'setting',
            'setting_edge',
            'method_setting_link',
            'outcome',
            'claim'
        )
    ),
    target_id        uuid NOT NULL,
    PRIMARY KEY (evidence_span_id, target_kind, target_id)
);

CREATE INDEX IF NOT EXISTS evidence_links_target_idx ON evidence_links (target_kind, target_id);
