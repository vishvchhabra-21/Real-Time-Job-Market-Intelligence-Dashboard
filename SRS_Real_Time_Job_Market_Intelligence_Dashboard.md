# Software Requirements Specification (SRS)
## Real-Time Job Market Intelligence Dashboard

**Version:** 1.0  
**Date:** June 2026  
**Status:** Approved Specification  
**Author:** Vishv Chhabra  

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [Functional Requirements](#3-functional-requirements)
4. [External Interface Requirements](#4-external-interface-requirements)
5. [System Architecture & Technical Design](#5-system-architecture--technical-design)
6. [Data Requirements & Schema](#6-data-requirements--schema)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [Build Timeline & Milestones](#8-build-timeline--milestones)
9. [Glossary](#9-glossary)

---

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification (SRS) outlines the functional, non-functional, data, and architectural requirements for the **Real-Time Job Market Intelligence Dashboard**. This document serves as the single source of truth for design, development, evaluation, and production deployment on Streamlit Cloud.

### 1.2 Scope
The application is a full-stack, ML-powered job market aggregator and analyzer tailored for data science and AI job seekers in India. The system schedules data pulling loops to extract open market listings from remote APIs, normalizes and cleans the raw payloads into a local relational database, applies NLP skill extraction, handles unsupervised role clustering, scores resume similarities, and surfaces interactive chat features via a GenAI large language model interface. 

The primary business outcome is a highly performant portfolio application that demonstrates proficiency in data engineering pipelines, text engineering, machine learning pipelines, and cost-optimized infrastructure deployment ($0 architecture footprint).

### 1.3 Intended Audience
* **Developer / Product Owner:** For pipeline construction, data orchestration, model tuning, and interface design.
* **Technical Recruiters and Interviewers:** For validating engineering rigors, system choices, and evaluating live product demonstrations during placement interview cycles.

### 1.4 References
* Business Requirements Document (BRD): Real-Time Job Market Intelligence Dashboard v1.0
* System Architecture & Technical Design Specification Document (June 2026)
* Wireframe & UI/UX Specification Document: Multi-Project Portfolio Dashboards v1.0

---

## 2. Overall Description

### 2.1 Product Perspective
The Real-Time Job Market Intelligence Dashboard operates as a standalone data application. It interacts with external scheduling utilities, third-party job aggregation interfaces, text embedding processors, and remote LLM execution environments.

```
┌──────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│ External APIs    │ ───>  │ Core Ingestion Layer │ ───>  │ Analytic DB & Frontend │
│ (JSearch / Groq) │       │ (APScheduler Loop)   │       │ (SQLite + Streamlit)   │
└──────────────────┘       └──────────────────────┘       └────────────────────────┘
```

The system preserves low-latency read operations by separating data fetching tasks from client rendering states using automated cron loops and cached interface wrappers.

### 2.2 Product Functions (High-Level)
1. **Automated multi-query API ingestion** spanning major tech hubs.
2. **Raw string tokenization, cleaning, and normalization** of salary ranges and geo-locations.
3. **Relational record deduplication** using source-specific unique identification constraints.
4. **Keyword extraction and dictionary alignment** through phrase token matching loops.
5. **Unsupervised cluster grouping** of variant roles into distinct baseline market segments.
6. **Vectorized semantic similarity evaluation** across raw user resumes and database descriptions.
7. **Ordinary Least Squares (OLS) trend modeling** projecting keyword counts over a 7-day outlook.
8. **Context-augmented Large Language Model chat workspace** optimized for parameter-efficient prompting.

### 2.3 User Classes and Characteristics
* **Primary End-User:** Job hunters looking for structural data insights into growing skills and hiring firms.
* **Technical Reviewers:** Discerning industry interviewers tracking consistency, design system fluency, system stability, and resource management choices.

### 2.4 Design and Implementation Constraints
* **Zero Budget Threshold:** Operational infrastructure costs must equal $0 across computing nodes, network proxies, storage files, and external APIs.
* **Volatile Document States:** Uploaded or input resume documents must stay inside transient operational threads and are never persisted to disk storage or network nodes.
* **Compute Bounds:** Execution metrics must survive free-tier rate limits, specifically adhering to a $14,400	ext{ tokens/minute}$ cap on Groq API processing streams.

---

## 3. Functional Requirements

### 3.1 Data Ingestion & Orchestration Pipeline
* **FR-1 [API Ingestion]:** The system shall programmatically query the JSearch API via RapidAPI gateways using dynamic search query strings and targeted locations.
* **FR-2 [Scraper Integration]:** The pipeline shall extract complementary India-centric listings by executing REST interfaces across an open-source Indeed scraper wrapper.
* **FR-3 [Automation Loop]:** The collection sequence shall trigger automatically at fixed 4-hour intervals managed by an internal `BlockingScheduler` utility instance.
* **FR-4 [Query Cardinality]:** The collection layer must support a minimum of four default search streams covering key tech roles and locations:
  * `"data scientist" in "Delhi"`
  * `"ML engineer" in "Bangalore"`
  * `"AI engineer" in "remote"`
  * `"data analyst" in "Noida"`
* **FR-5 [Deduplication Routing]:** The ingestion node shall validate entries against a unique `job_id` constraint, discarding duplicate records before triggering write operations to storage.

### 3.2 In-Memory Data Cleaning & Normalization
* **FR-6 [Geographic Standardization]:** The cleaner module shall parse inconsistent geostrings into unified canonical location descriptors (e.g., converting `"NCR"`, `"New Delhi"`, or `"Delhi/NCR"` into `"Delhi"`).
* **FR-7 [Salary Processing]:** The data parser shall convert variable financial strings or null tokens into explicit, unified numerical arrays.
* **FR-8 [Description Filtering]:** The normalization loop shall strip raw job description fields of HTML tags, formatting markdown, and redundant spaces, truncating final strings to a 2,000 character maximum.

### 3.3 Natural Language & Machine Learning Pipeline
* **FR-9 [NLP Skill Extraction]:** The text processing core shall identify specific technical keywords in descriptions using a token alignment match loop against a predefined list of tools and frameworks:
  * *Python, SQL, TensorFlow, PyTorch, Scikit-learn, LangChain, Docker, MLOps, LLM, RAG, XGBoost, NLP, Pandas, Power BI, Tableau, Spark, Kafka, FastAPI.*
* **FR-10 [Role Clustering]:** The machine learning stack shall categorize incoming roles into a configurable range of 3 to 7 clusters by processing TF-IDF description matrices through a K-Means Clustering algorithm.
* **FR-11 [Resume Alignment Match]:** The evaluation framework shall calculate a relative matching percentage (0–100%) between an uploaded resume string and available job descriptions by running a Cosine Similarity function across computed TF-IDF feature weights.
* **FR-12 [Timeline Trend Forecasting]:** The prediction framework shall estimate specific skill changes over a 7-day window by running an Ordinary Least Squares linear regression model across historical daily listing aggregations.

### 3.4 Streamlit Dashboard Views & Interface Layout
* **FR-13 [Global Workspace Metrics]:** The entry shell must render three high-level KPI cards at the top of the interface displaying: Total Extracted Listings, Count of Unique Hiring Entities, and Count of Monitored Metro Locations.
* **FR-14 [Skill In-Demand Heatmap View]:** The system shall render a horizontal Plotly bar graph displaying the top 20 technical keywords identified across active database records.
* **FR-15 [Timeline Forecast Chart View]:** The system shall plot rolling averages alongside a linear projection line highlighting expected changes in job volumes over the next 7 days.
* **FR-16 [Resume Match Engine View]:** The interface shall provide a free-text entry area for raw resumes and render a ranked table displaying the top 20 matching jobs based on semantic similarity.
* **FR-17 [Live Postings Table View]:** The dashboard shall present an interactive data table displaying active listings, filterable by city, core technical domain, and posting age (using a 1–30 day slider widget).
* **FR-18 [Render Cache Control]:** Database extraction tasks shall be wrapped in an `st.cache_data` decorator with a 1-hour Time-To-Live (TTL) constraint to prevent redundant disk read operations.

### 3.5 GenAI Market Analyst Chat Interface
* **FR-19 [Interface Container]:** The dashboard footer space shall integrate a fixed, threaded conversational box for handling natural language user inputs.
* **FR-20 [Context Summarization Frame]:** The chat pipeline shall parse and summarize data structures from the active filtered database (top 10 skills, top 5 cities, top 5 hiring companies) to inject as context into the LLM system prompt.
* **FR-21 [Model Route Allocation]:** Conversational transactions shall map directly to the `llama-3.3-70b-versatile` inference engine via the external Groq API gateway.
* **FR-22 [Query Resolution Scope]:** The assistant must return logical answers to specific contextual themes, including regional skill requirements, missing technical qualifications, hiring trends, and general upskilling advice.

---

## 4. External Interface Requirements

### 4.1 User Interface Design
The front end shall utilize the responsive, open-source Streamlit framework, enforcing a single-page split architecture:
* **Sidebar Portal Panel:** Dedicated to data manipulation toggles, regional filters, timeframe adjustments, and metadata health indicators.
* **Main Context Workspace Tabs:** Organized into clean horizontal tab panels (`st.tabs`) to keep visual analytics structured and prevent excessive vertical scrolling.

### 4.2 Application Programming Interfaces (APIs)
The architecture depends on the following remote endpoints for data sourcing and language model inferences:

| Connection Endpoint | Service Role | Operational Tier | Fallback Strategy |
| :--- | :--- | :--- | :--- |
| `https://jsearch.p.rapidapi.com/search` | Sourcing live job posting documents from search engine records. | Free Level: 200 calls/month pool restriction bounds. | Log collection drop alerts, step down frequency, and execute backup scrapers. |
| `https://api.groq.com/openai/v1/chat/completions` | Serving user chat messages and generating context-driven replies. | Free Level: 14,400 tokens/minute throughput limit. | Graceful timeout intercept; switch conversational cards to local rules. |

---

## 5. System Architecture & Technical Design

### 5.1 Text Engineering Engine Mechanics
Candidate text matching operations follow an algebraic space model. Text blocks are transformed into sparse metric dimensions via the Term Frequency-Inverse Document Frequency ($	ext{TF-IDF}$) equation:

$$	ext{tf-idf}(t, d, D) = 	ext{tf}(t, d) 	imes \log\left(rac{|D|}{1 + |\{d \in D : t \in d\}|}ight)$$

Where $t$ indicates a specific token, $d$ indicates an individual document block, and $D$ represents the complete corpus array. Alignment scores are derived using a Cosine Similarity algorithm:

$$	ext{Similarity}(A, B) = rac{A \cdot B}{\|A\| \|B\|} = rac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

### 5.2 Dynamic Orchestration Lifecycles
The pipeline runs an internal automation loop separate from the client rendering components, ensuring the interface remains fast and responsive.

```
[ APScheduler Trigger ] ──(Every 4 Hours)──> [ Poll Remote API Endpoints ]
                                                       │
                                            [ Normalize & Deduplicate ]
                                                       │
                                            [ Write to Local SQLite ]
                                                       │
                                         [ Reset Streamlit Cache Timer ]
```

---

## 6. Data Requirements & Schema

### 6.1 Persistent Relational Schema
All structured entities are written directly to a local, file-based SQLite relational core (`data/jobs.db`).

#### Database Table Definition: `jobs`
| Field Attribute | Storage Datatype | Structural Constraints | Engineering Objective & Context |
| :--- | :--- | :--- | :--- |
| `id` | `TEXT` | `PRIMARY KEY` | Unique source hash code token used to prevent duplicate entries. |
| `title` | `TEXT` | `NOT NULL` | The original raw job title string pulled from the source platform. |
| `company` | `TEXT` | `NOT NULL` | The clean, parsed hiring company identifier string. |
| `location` | `TEXT` | `NOT NULL` | Standardized geo-location tag assigned by normalization steps. |
| `salary` | `TEXT` | `NULLABLE` | Standard numerical wage string or range array values. |
| `description` | `TEXT` | `NOT NULL` | Main text block, cleaned of markup and capped at 2,000 characters. |
| `skills` | `TEXT` | `NULLABLE` | Comma-delimited list of keywords flagged by the extraction loop. |
| `posted_at` | `TEXT` | `ISO8601 Compliance` | Structured UTC string indicating listing creation time. |
| `fetched_at` | `TEXT` | `ISO8601 Compliance` | Audit timestamp tracking when the record was ingested into the database. |

---

## 7. Non-Functional Requirements

### 7.1 Performance Metrics
* **NFR-1 [Render Speed]:** The initial dashboard interface must load and render completely in under 5 seconds on standard mobile or desktop connections.
* **NFR-2 [Model Inference Latency]:** Semantic similarity calculations across 1,000 data rows must execute and return sorted results in under 3 seconds.
* **NFR-3 [LLM Response Target]:** Groq streaming completions must generate and display answers in under 5 seconds from the initial request timestamp.

### 7.2 Safety, Privacy, and Confidentiality
* **NFR-4 [Zero Local Storage Retention]:** User resume documents pasted into text fields must process entirely within volatile memory threads and must never be written to persistent disks or database records.
* **NFR-5 [Data Scrubbing Rules]:** The parsing pipeline must strip out identifying personal data elements—such as cell phone details, emails, or street addresses—before passing text data to external LLM connections.

### 7.3 Portability & Maintainability Rules
* **NFR-6 [Cross-Platform Execution]:** The system code base must run predictably on any compute environment operating Python 3.10+ without relying on OS-specific compilation binaries.
* **NFR-7 [Externalized Technical Configs]:** Target dictionary terms, threshold tolerances, and interface constraints must reside within dedicated YAML configuration files instead of being hardcoded into application files.

---

## 8. Build Timeline & Milestones

The project delivery plan is structured into a tight 3-week timeline, avoiding complex multi-node systems:

```
+-------------------------------------------------------------------------+
| PHASE 1: DATA PIPELINE BASELINE (Days 1–7)                              |
| Complete API integrations, parsing components, and scheduler logic.     |
| Deliverable: A working jobs.db holding 500+ clean listings.            |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
| PHASE 2: ALGORITHMIC CORE & MODEL TUNING (Days 8–14)                   |
| Build spaCy skill match scripts, K-Means clustering, and TF-IDF engines. |
| Deliverable: Models verified and scoring accurately on local test cases. |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
| PHASE 3: INTERFACE LAYOUT & STREAMLIT CLOUD DEPLOY (Days 15–21)          |
| Wire up Plotly charts, add Groq chat interfaces, and launch publicly.   |
| Deliverable: Application live on Streamlit Cloud with an extensive README.  |
+-------------------------------------------------------------------------+
```

---

## 9. Glossary

* **TF-IDF:** *Term Frequency-Inverse Document Frequency*; a statistical vector weighting model used to track term importance across document groups.
* **Cosine Similarity:** A metric that determines the alignment intensity of two non-zero vectors by calculating the cosine of the angle between them.
* **K-Means:** An unsupervised geometric partitioning algorithm that assigns data records into K distinct cluster segments.
* **Streamlit Cloud:** An open-source hosting platform optimized for deploying interactive Python data applications with minimal overhead.
* **Groq Inference Engine:** A high-speed hardware and software API platform used to execute LLM transactions with exceptionally low latency.
* **APScheduler:** *Advanced Python Scheduler*; a lightweight runtime library used to handle recurring background tasks inside Python processes.

---
*End of Specification Document*
