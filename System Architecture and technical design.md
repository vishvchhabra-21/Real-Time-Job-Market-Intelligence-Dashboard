# System Architecture & Technical Design Document

## Multi-Project Portfolio Architecture: Real-Time Job Market Aggregator & Transient In-Memory Fintech Analytics Suite

**Version:** 1.0

**Date:** June 2026

**Status:** Baseline Specification

**Target Environments:** Streamlit Cloud Tier / Local Execution Prototyping

---

# 1. Document Overview & Architectural Goals

This document details the system architecture, component-level engineering design, and algorithmic mechanics for two complementary full-stack portfolio systems:

1. **Real-Time Job Market Intelligence Dashboard**: A data engineering and NLP aggregator that scrapes, processes, and persists Indian tech industry listings to a relational core.
2. **AI Personal Finance & Loan Health Advisor**: A secure, zero-persistence financial intelligence system that parses local banking records in-memory to compute underwriting risk vectors and behavioral anomalies.

## 1.1 Core Architectural Principles

- **Zero Infrastructure Cost (₹0 Net Spend):** Both platforms run completely within free-tier compute limitations, leveraging Streamlit Cloud for web servers, SQLite for analytical caching, and specialized API engines for inference.
- **Strict Privacy Guardrails:** Financial asset transactions are processed entirely within transient memory cycles and are never serialized or persisted to disk or cloud target databases.
- **Live Interview Democraft:** Built specifically to provide robust, zero-latency micro-demos during technical interviews by incorporating robust data-mocking routines and rapid computation.

---

# 2. System Architecture & High-Level Patterns

The portfolio architecture applies two distinct architectural patterns tailored to the underlying lifecycle of the processed data:

- **Persistent Storage / Aggregator Pattern** for market telemetry
- **Transient Session State Pipeline** for sensitive financial transactions

## 2.1 Aggregator Pipeline Pattern (Job Market Intelligence Dashboard)

```text
┌─────────────────┐      ┌────────────────────┐      ┌───────────────┐      ┌──────────────────┐
│  Data Sources   │ ───> │ Ingestion / Sched  │ ───> │ Relational DB │ ───> │ Interactive UI   │
│ (JSearch,       │      │ (APScheduler,      │      │ (Local SQLite │      │ (Streamlit App,  │
│ Scrapers)       │      │ Normalization)     │      │   jobs.db)    │      │ Plotly Visuals)  │
└─────────────────┘      └────────────────────┘      └───────────────┘      └────────▲─────────┘
                                                                                     │
                                                                            ┌────────▼─────────┐
                                                                            │ ML Inference /   │
                                                                            │ Groq LLM Engine  │
                                                                            └──────────────────┘
```

The market data ingest architecture is built around an autonomous background daemon pattern:

- **Ingestion Schedule:** An `APScheduler` loop spins up every 4 hours to poll remote target APIs.
- **Deduplication:** Raw JSON payloads are filtered by unique source identification tokens before being normalized into standard schemas.
- **Persistent Store:** A localized SQLite instance serves read operations to the frontend UI through an internal analytical caching layer.

## 2.2 Transient Session State Pipeline (AI Personal Finance Advisor)

```text
┌──────────────────┐      ┌────────────────────┐      ┌─────────────────┐      ┌──────────────────┐
│ Secure Interface │ ───> │ In-Memory Parser   │ ───> │ ML Model Suite  │ ───> │ Groq Advisor     │
│ (File Drag-Drop) │      │ (Transient Memory) │      │ (Scikit / XGB)  │      │ (Anonymized RAG) │
└──────────────────┘      └────────────────────┘      └─────────────────┘      └──────────────────┘
```

The financial analysis application functions as a zero-disk stateless processor:

- **Ingestion Webhook:** Raw PDFs or CSVs map directly into memory arrays as volatile state objects.
- **Inference Routing:** Objects pass through parsing engines, classification trees, and spatial cluster maps in sequence.
- **State Dissolution:** On user session termination or browser window reload, all data structures are instantly flushed out of memory.

---

# 3. Component-Level Design & ML/AI Engine Mechanics

## 3.1 Text Vectorization & Semantic Search Engine (Resume & Job Fit Matching)

The platform uses an algebraic space model to match candidate profiles with job roles.

### Mathematical Formulation

The textual raw components are converted into statistical feature weightings using the **TF-IDF** formula:

```math
tf-idf(t,d,D) = tf(t,d) × log(|D| / (1 + |{d ∈ D : t ∈ d}|))
```

Where:

- `t` = token
- `d` = document
- `D` = corpus

Once converted into vector arrays, alignment is derived via **Cosine Similarity**:

```math
Similarity(A,B) =
(A · B) / (||A|| ||B||)

=
Σ(AiBi)
/ ( √Σ(Ai²) × √Σ(Bi²) )
```

### Pipeline Execution Sequence

1. Text arrays undergo downcasing, stopword stripping, and alphanumeric normalization.
2. A collective feature array is instantiated across both the target resumes and active job database.
3. Spatial distance calculations yield localized scores mapped between 0% and 100% match accuracy.

---

## 3.2 Unsupervised Spatial Categorization (Role Clustering)

To handle unstructured job title varieties across multiple platforms, the aggregator dynamically categorizes openings via **K-Means Clustering**.

### Feature Vector Input

Text vectors are extracted from job description fields.

### Optimization Objective

The engine partitions data points into `K` discrete segments (`K ∈ [3,7]`) by minimizing the Within-Cluster Sum of Squares (WCSS):

```math
WCSS = Σ(j=1→K) Σ(xi ∈ Sj) ||xi - μj||²
```

Where:

- `μj` = centroid vector
- `Sj` = cluster point set

---

## 3.3 Linear Forecasting Engine

The dashboard projects skill demand variations across a 7-day trailing timeline using a standard Linear Regression model:

```math
y = β₀ + β₁x + ε
```

Where:

- `x` = time vector
- `y` = keyword frequency
- Coefficients are derived using Ordinary Least Squares (OLS).

---

## 3.4 Financial Underwriting Risk Matrix & Model Interpretability (SHAP)

The loan repayment tool evaluates candidate creditworthiness using an **XGBoost** model.

### Extraction Parameters

- **Income Cadence Stability**
  - Measures variance in recurring salary credits.
- **Debt Inflow Ratio**
  - Compares debt repayments against verified income.
- **Liquidity Inflow Rates**
  - Tracks balance stability and low-water marks.

### Interpretability Integration

To explain model outputs, predictions are converted into SHAP factor contributions:

```math
φi(v) =
Σ(S ⊆ N\{i})
(
|S|!(|N|-|S|-1)!
/
|N|!
)
(
v(S ∪ {i}) - v(S)
)
```

This calculates the marginal contribution of each feature across all feature subsets.

```text
High Debt Inflow Ratio --------> (+18.4) ──┐
Missed Automated Debits -------> (+22.0) ──┼──> Net Output Risk Score: 68 / 100
Stable Income Frequency -------> (-11.5) ──┘
```

---

## 3.5 Isolated Enclosure Spatial Fault Flagging (Transaction Anomaly Detection)

Unsupervised anomaly detection is managed via an **Isolation Forest** engine.

```text
Normal Data Points (Deep Tree Paths)

         [Root Node]
         /         \
      [...]       [...]
      /   \       /   \
   [...] [...] [...] [...]

Anomalous Data Points (Short Isolated Splits)

         [Root Node]
         /         \
   [Anomaly]      [...]
```

The algorithm constructs recursive isolating trees across:

- Transaction amounts
- Transaction timing
- Merchant categories

Outliers are flagged using significantly shorter path lengths.

---

# 4. Database Schema & Data Models

## 4.1 Persisted Relational Architecture (Job Market Database)

### Table: `jobs`

| Attribute | Storage Type | Constraint | Purpose |
|------------|-------------|------------|----------|
| id | TEXT | PRIMARY KEY | Unique source-generated identifier |
| title | TEXT | NOT NULL | Standardized role title |
| company | TEXT | NOT NULL | Hiring organization |
| location | TEXT | NOT NULL | Canonical location |
| salary | TEXT | NULLABLE | Salary range |
| description | TEXT | NOT NULL | Truncated job description |
| skills | TEXT | NULLABLE | Extracted skill keywords |
| posted_at | TEXT | ISO8601 | Listing timestamp |
| fetched_at | TEXT | ISO8601 | Pipeline fetch timestamp |

---

## 4.2 Non-Persisted Transient Memory Layout (Personal Finance Engine)

```text
InMemory DataFrame

├── transaction_date : datetime64ns
├── raw_description  : object
├── canonical_vendor : object
├── channel_type     : category
├── cash_outflow     : float64
├── cash_inflow      : float64
└── active_balance   : float64
```

### Channel Types

- UPI
- NetBanking
- IMPS
- Cash
- POS

---

# 5. Sequence of Operations & Execution Flows

## 5.1 Automated Aggregation Execution Sequence

```text
┌─────────────┐       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│ APScheduler │ ───>  │ API Scraper │ ───>  │ Normalizer  │ ───>  │ SQL Engine  │ ───>  │ UI Refresh  │
│ Trigger     │       │ Engine      │       │ Data Loops  │       │ Operations  │       │ Cache Reset │
└─────────────┘       └─────────────┘       └─────────────┘       └─────────────┘       └─────────────┘
```

### Workflow

1. APScheduler reaches a 4-hour interval.
2. Concurrent API requests are dispatched.
3. HTML and location normalization routines clean the payload.
4. Records are stored using `INSERT OR IGNORE`.
5. UI cache TTL is refreshed.

---

## 5.2 Banking Inference Execution Sequence

```text
┌─────────────┐       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│ PDF Upload  │ ───>  │ pdfplumber  │ ───>  │ ML Stack    │ ───>  │ Context RAG │ ───>  │ Chat Output │
│ Drag/Drop   │       │ Cell Parse  │       │ Matrix Runs │       │ Generation  │       │ Render Box  │
└─────────────┘       └─────────────┘       └─────────────┘       └─────────────┘       └─────────────┘
```

### Workflow

1. User uploads a bank statement.
2. `pdfplumber` extracts text content.
3. Regex-based NLP parsing structures transactions.
4. XGBoost and Isolation Forest models process data.
5. Summary metrics are prepared for context.
6. Groq Llama generates recommendations without persistence.

---

# 6. Interface Design & API Integration Matrix

## 6.1 Remote Endpoint Execution Manifest

| Endpoint Target | Integration Layer | Traffic Profile | Failover Mechanism |
|----------------|------------------|----------------|-------------------|
| JSearch Gateway | `https://jsearch.p.rapidapi.com/search` | Every 4 hours | Fallback scraper modules |
| Groq Llama Pipeline | `https://api.groq.com/openai/v1/chat/completions` | On demand | Local rule-based summaries |

---

## 6.2 Prompt Structural Ingestion Layouts

```text
+------------------------------------------------------------------------------------+
| SYSTEM PROMPT INGESTION FRAMEWORK                                                  |
+------------------------------------------------------------------------------------+
| You are a technical financial advisor specializing in Indian UPI systems.          |
| Operate strictly on summarized telemetry metadata. Never extrapolate.              |
+------------------------------------------------------------------------------------+
| CONTEXT SUMMARY                                                                    |
+------------------------------------------------------------------------------------+
| Total Inflow                : ₹85,000                                              |
| Total Outflow               : ₹74,460                                              |
| Savings Rate                : 12.4%                                                |
| Underwriting Risk Score     : 68 / 100                                             |
| High Risk Anomalies         : 2                                                    |
| Primary Spending Category   : Rent & Housing (52.9%)                               |
+------------------------------------------------------------------------------------+
| USER QUESTION                                                                      |
+------------------------------------------------------------------------------------+
| Why did the system flag my underwriting profile as high-risk?                      |
+------------------------------------------------------------------------------------+
```

---

# 7. Non-Functional & Security Frameworks

## 7.1 Security & Privacy Compliance Architecture

### Credential Protection

- Secrets are stored in Streamlit's encrypted secrets manager.
- No API keys are committed to source control.

### Zero Disk Footprint

- Financial data is processed in volatile memory only.
- Session termination clears all financial records.

### Context Anonymization

The parser removes:

- Account numbers
- Customer names
- Other personally identifiable information (PII)

before downstream processing.

---

## 7.2 Fault-Tolerance Metrics & Performance Targets

```text
┌─────────────────────────────────────────────────────────────┐
│ PERFORMANCE TARGETS                                         │
├─────────────────────────────────────────────────────────────┤
│ Initial Screen Render      : < 5 Seconds                    │
│ Inference Processing       : < 3 Seconds                    │
│ Chat Completion Roundtrip  : < 5 Seconds                    │
└─────────────────────────────────────────────────────────────┘
```

### Fault Resiliency Protocols

- Safe handling of API timeout failures.
- Graceful degradation to rule-based recommendations.
- UI-level fallback alerts during external service outages.
- Automatic retry strategies for transient failures.

---

*End of Engineering Design Specification*