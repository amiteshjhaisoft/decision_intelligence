Hello World!

Define scope & architecture

LLM for generation: Claude (Anthropic).

Embeddings: pick a provider (local sentence-transformers or a managed embedding API) since Claude is for generation; you can swap later.

Vector DB: Milvus (managed via Zilliz Cloud or self-hosted).

Object storage: pick one (S3 / Azure Blob / GCS) as your Knowledge Base (KB) and artifact store.

App: Streamlit single-page chat + admin/ingestion page.

Create the GitHub repository & baseline structure

Folders: src/ (Streamlit app + ingestion pipeline), config/ (YAML), .streamlit/ (local secrets), docs/, tests/.

Add .gitignore to exclude secrets (.env, .streamlit/secrets.toml, any credential files).

Add README.md with run instructions and config conventions.

Add requirements.txt (app + Milvus + loaders + Anthropic SDK).

Design the YAML configuration (no secrets in Git!)

Create config/app.yaml with sections:

streamlit: page title, layout, theming, toggles.

storage: provider (aws|azure|gcp), bucket/container name, base prefix (e.g., kb/), paths for raw/, chunks/, indexes/.

vector_store: type=milvus, host/port (or URI), collection names, index params (e.g., HNSW/IVF, dim, metric), consistency level.

rag: chunk_size, overlap, splitter, top_k, score_threshold, reranker on/off.

llm: provider=anthropic, model name, temperature, max_tokens.

logging: level, destination.

security: allowlist domains, redaction on/off.

Reference secrets via env placeholders (e.g., {{ env.ANTHROPIC_API_KEY }}) you’ll resolve at runtime.

Set up secrets management (per environment)

Local dev: put real keys in .streamlit/secrets.toml or .env.

Cloud: use a secrets manager (AWS Secrets Manager / Azure Key Vault / GCP Secret Manager) and inject as env vars at deploy time.

Required secrets: Anthropic API key, cloud storage credentials (or managed identity), Milvus auth (if enabled).

Provision cloud storage (choose one)

AWS: create S3 bucket; apply least-priv IAM policy scoped to the bucket/prefix; enable server-side encryption.

Azure: create Storage Account + Blob container; assign RBAC (Blob Data Contributor) to your app’s identity or use a connection string; enable encryption.

GCP: create GCS bucket; create service account with limited role; enable uniform bucket-level access.

Standardize folder prefixes: kb/raw/, kb/chunks/, kb/ingested/, kb/index/.

Decide and provision Milvus

Option A (recommended to start): Zilliz Cloud (managed Milvus). Create a cluster, note public endpoint, create DB + collections, whitelist your app IPs.

Option B: Self-host Milvus 2.x

Docker Compose (dev) or Kubernetes with Milvus Operator (prod).

Configure object storage backend (S3/MinIO; for Azure/GCP use a compatible gateway or the operator’s supported backends).

Expose a stable endpoint for the app.

Define Milvus schema & index strategy

Choose embedding dimension based on embedding model.

Pick metric (cosine/L2) and index (HNSW or IVF_x) + auto-index params.

Plan collections: e.g., documents (metadata), chunks (vectors+payload), optional snapshots for KB versioning.

Prepare the ingestion pipeline (offline job or admin page action)

Loaders: PDF, DOCX, PPTX, HTML, CSV, etc.

Clean & deduplicate documents; persist originals to kb/raw/.

Split into chunks (size/overlap per YAML); persist chunk manifests to kb/chunks/.

Compute embeddings; upsert to Milvus with metadata (doc_id, chunk_id, source_path, hash, timestamp, tags).

Write an ingestion ledger to storage for idempotency and re-indexing.

Implement retrieval orchestration (design only for now)

On user query: embed query → search Milvus (top_k, threshold) → optional rerank → assemble context with token-budgeting → build a grounded prompt → call Claude → stream response.

Capture citations (doc_id + chunk_id) for transparency.

Plan Streamlit UI flows

Chat tab: message history, settings drawer (top_k, temperature), source citations panel.

Knowledge Base tab: upload docs, show ingestion status, reindex button, collection stats (vector count, last updated).

Settings tab: config preview (read-only values from YAML), health checks (Milvus + storage + LLM).

Wire up configuration loading & validation

At app start: load config/app.yaml, resolve env placeholders, validate required fields, log effective config (without secrets).

Fail fast with clear error messages if Milvus/storage/LLM not reachable.

Local development & smoke tests

Add a few sample docs to kb/raw/, run ingestion once.

Validate vectors exist in Milvus; run a couple of test queries; verify citations and scores.

Exercise edge cases (empty results, large responses, rate-limit handling).

CI basics

Add pre-commit (black/ruff/isort) and unit tests for config parsing and health checks.

GitHub Actions: lint + minimal tests on PR; optional build artifact (wheel or container).

Deployment target selection

Quick start: Streamlit Community Cloud (if you don’t need private networks).

Production: containerize and deploy to your cloud:

AWS (ECS Fargate or EKS),

Azure (Web App for Containers or AKS with managed identity to Blob),

GCP (Cloud Run or GKE).

Inject secrets as env vars; restrict outbound to Milvus + storage endpoints; add HTTPS.

Observability & governance

Centralized logging (CloudWatch / App Insights / Cloud Logging).

Prompt/response audit logs (with redaction).

Milvus and storage metrics (vector count, latency, failures).

Budget alerts and usage caps on LLM calls.

Data lifecycle & guardrails

Snapshot KB versions (manifest file) and keep “latest” pointer for retrieval.

TTL/retention policy for old chunks; dedupe by hash.

Safety: PII redaction on ingestion (optional), allowlist domains for web-ingest, max doc sizes, MIME checks.

Documentation & runbooks

“How to add a new document” (non-technical steps).

“How to rotate keys / change clouds / change collection settings.”

“Disaster recovery” (restore KB from storage, rebuild Milvus).

Cut a tagged release & acceptance checklist

Config reviewed, secrets in place, smoke tests pass, health check green, rollback plan documented.

Tag the repo, open a release note, and deploy.
