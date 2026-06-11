# Wireframe & UI/UX Specification Document
## Multi-Project Portfolio Dashboards

**Version:** 1.0  
**Date:** June 2026  
**Status:** Approved Specification  
**Target Architecture:** Streamlit Cloud Framework (Responsive Web)  

---

## 1. Document Overview & Design System Constraints

This document provides exhaustive visual and structural wireframe layouts for two data science portfolio projects:
1. **Real-Time Job Market Intelligence Dashboard** (ML/NLP Market Aggregator)
2. **AI Personal Finance & Loan Health Advisor** (Fintech In-Memory Analytics Suite)

### 1.1 Streamlit Architecture Constraints
To achieve a production-grade look on Streamlit's open-source framework without external CSS/JS baggage, the following layout rules apply:
- **Sidebar Integration:** Reserved for global controls, primary upload portals, configuration toggles, and metadata.
- **Metric Grids:** Implemented using exact multi-column layouts (`st.columns(3)`), with structured visual hierarchy (large value primary, small text label secondary).
- **Tabbed Sub-navigation:** Complex features are compartmentalized into horizontal tabs (`st.tabs()`) to prevent endless vertical scrolling.
- **In-Memory Volatility (Finance Dashboard):** No file caching or database serialization across refreshes. The UI must cleanly switch between an **Empty Upload State** and an **Active Session State**.

---

## 2. Real-Time Job Market Intelligence Dashboard

### 2.1 Global Application Layout (Shell Structure)

```
+---------------------------------------------------------------------------------------------------+
|  [Icon] REAL-TIME DATA SCIENCE & AI JOB MARKET INTELLIGENCE DASHBOARD (INDIA)                     |
+---------------------------------------------------------------------------------------------------+
| SIDEBAR CONTROLS         | MAIN CONTENT HEADER                                                    |
|                          |                                                                        |
| 🌐 GEOGRAPHIC FILTER     |  [ KPI: Total Listings ]    [ KPI: Unique Companies ]   [ KPI: Cities ]|
| [ Delhi/NCR          v ] |  |    1,420 records      |    |      284 brands       |   |   12 covered|  |
|                          |  +----------------------+    +-----------------------+   +-------------+ |
| 🛠️ DOMAIN SELECTION      |                                                                        |
| [x] Data Scientist       | +--------------------------------------------------------------------+ |
| [x] ML Engineer          | |  Tab 1: Skill Heatmap  |  Tab 2: Forecasts  |  Tab 3: Resume Scorer | |
| [ ] Data Analyst         | +--------------------------------------------------------------------+ |
|                          | |                                                                    | |
| 📅 POSTING FRESHNESS     | |  [ACTIVE TAB VIEWPORTS DISPLAY HERE]                               | |
| Slider: [---o--------]   | |                                                                    | |
| Filter: Last 7 Days      | |                                                                    | |
|                          | +--------------------------------------------------------------------+ |
| ⚡ SYSTEM METRICS        | | 🤖 GENAI JOB MARKET ANALYST CHAT INTERFACE                         | |
| DB Refresh: Every 4 Hours| | > Ask any question (e.g., "What are the core skill gaps in Delhi?")  | |
| Infra Cost: ₹0 (Free)    | | 📥 [Type your query here...                                      ] | |
+--------------------------+------------------------------------------------------------------------+
```

---

### 2.2 Tab 1: Skill In-Demand Heatmap View

```
+---------------------------------------------------------------------------------------------------+
|  Tab 1: Skill Heatmap  |  Tab 2: Forecasts  |  Tab 3: Resume Scorer                               |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  📊 Top 20 Most Demanded Skills Across Filtered Listings                                          |
|                                                                                                   |
|  Skill          Volume   Distribution Chart                                                       |
|  ------------   ------   -----------------------------------------------------------------------  |
|  Python           840    [████████████████████████████████████████████████████████████] (59.1%)    |
|  SQL              612    [██████████████████████████████████████████████] (43.0%)                 |
|  PyTorch          422    [████████████████████████████████] (29.7%)                               |
|  SQL/NoSQL        390    [██████████████████████████████] (27.4%)                                 |
|  LangChain        310    [████████████████████████] (21.8%)                                       |
|  LLMs/RAG         298    [███████████████████████] (20.9%)                                        |
|  Scikit-Learn     240    [██████████████████] (16.9%)                                             |
|  Docker/MLOps     195    [███████████████] (13.7%)                                                |
|  FastAPI          140    [███████████] (9.8%)                                                     |
|                                                                                                   |
|  ℹ️ Tooltip: Core skills extracted via spaCy custom Named Entity Recognition (NER) pipeline     |
+---------------------------------------------------------------------------------------------------+
```

---

### 2.3 Tab 2: 7-Day Forecasting Trend Chart

```
+---------------------------------------------------------------------------------------------------+
|  Tab 1: Skill Heatmap  |  Tab 2: Forecasts  |  Tab 3: Resume Scorer                               |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  📈 7-Day Rolling Demand Trend & Linear Regression Forecast                                       |
|  [ Select Focus Skill to Forecast: LangChain             v ]                                      |
|                                                                                                   |
|  Job Count                                                                                        |
|    60 |                                                     *--[Predicted Rise]--* |
|    50 |                                              *---* |
|    40 |                                       *---* |
|    30 |                                *---* |
|    20 |                         *---* |
|    10 |                  *---* |
|     0 +------------------o------------o------------o------------o------------o------------+----  |
|       June 04      June 06      June 08      June 10      June 12 (F)  June 14 (F)  June 16 (F)    |
|                                                                                                   |
|  💡 Model Insight: LangChain shows a statistically significant upward demand vector (m = +4.2)     |
|     for Delhi/NCR region listings over the upcoming 7-day trailing cycle.                         |
+---------------------------------------------------------------------------------------------------+
```

---

### 2.4 Tab 3: Personalized Resume Fit Scorer View

```
+---------------------------------------------------------------------------------------------------+
|  Tab 1: Skill Heatmap  |  Tab 2: Forecasts  |  Tab 3: Resume Scorer                               |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  📝 Personal Resume Compatibility Engine                                                          |
|  Paste raw resume text or markdown text to compute instant semantic matching score:               |
|  +----------------------------------------------------------------------------------------------+ |
|  | Senior Data Scientist Candidate. Core expertise: Python, SQL, predictive modelling using     | |
|  | Scikit-learn, deep learning architectures using PyTorch. Built RAG applications...           | |
|  +----------------------------------------------------------------------------------------------+ |
|  [ Button: Calculate Fit Profiles ]                                                               |
|                                                                                                   |
|  🎯 MATCHING REAL-TIME OPENINGS (Ranked by TF-IDF Cosine Similarity)                             |
|                                                                                                   |
|  Rank  Job Title               Company             Location       Match Score   Status            |
|  ----  ---------------------   -----------------   ------------   -----------   ---------------   |
|  #01   Generative AI Engineer  Kore.ai             Noida (Remote)   94.2%       [ Exceptional ]   |
|  #02   Data Scientist - LLM    Paytm               Delhi/NCR        88.5%       [ High Match  ]   |
|  #03   Machine Learning Eng.   PhonePe             Bangalore        71.4%       [ Moderate ]      |
|  #04   Analytics Consultant    Fractal Analytics   Gurgaon          62.1%       [ Borderline  ]   |
|                                                                                                   |
|  🚧 Identified Skill Deficiencies (Not detected in your resume text):                            |
|  ⚠️ Missing: [ Docker ]  [ MLOps ]  [ FastAPI ]                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

### 2.5 GenAI Market Analyst Chat Interface (Fixed Footer Container)

```
+---------------------------------------------------------------------------------------------------+
| 🤖 GENAI JOB MARKET ANALYST CHAT INTERFACE (Powered by Groq Llama-3.3-70B)                         |
+---------------------------------------------------------------------------------------------------+
| Context Transmitted: Snapshot of 1,420 listings | Top Skills: Python, SQL | Active Region: Delhi  |
|                                                                                                   |
| User: What specific resume updates will maximize my hit rate for Paytm roles in Delhi NCR?         |
|                                                                                                   |
| Bot: Based on cross-referencing your resume with 42 active Paytm listings in our SQLite database: |
|      1. Explicitly list 'FastAPI' in your project section. 32% of Paytm AI descriptions mandate it.  |
|      2. Quantify your experience with PyTorch - Paytm's current listings favor it 3:1 over TF.     |
|      3. Reframe your NLP projects to mention Retrieval-Augmented Generation (RAG) models.         |
|                                                                                                   |
| 📥 [ Ask a follow-up question about salary benchmarks or missing skills...                    ]  |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. AI Personal Finance & Loan Health Advisor

### 3.1 Session State 1: Initial State (Empty/Awaiting Document Upload)

```
+---------------------------------------------------------------------------------------------------+
|  [Icon] AI PERSONAL FINANCE & LOAN REPAYMENT HEALTH ADVISOR                                       |
+---------------------------------------------------------------------------------------------------+
| SIDEBAR CONTROLS         | WELCOME PORTAL & DATA PROTECTION SECURE INGESTION                      |
|                          |                                                                        |
| 📁 DATA INGESTION SUITE  | Welcome to the AI Personal Finance Suite. This enterprise-grade engine |
| Drag and Drop Portal:    | extracts transaction signals, tags spending vectors, evaluates credit  |
| +----------------------+ | default risk indicators, and surfaces structural anomalies.            |
| |  [ Click to Browse ] | |                                                                        |
| |                      | |  🔒 IN-MEMORY PRIVACY GUARANTEE:                                       |
| |   Accepts HDFC, SBI, | |  Your transaction documentation is stored exclusively within transient |
| |   ICICI, AXIS CSV/PDF| |  RAM memory execution states. Raw transaction logs are never serialized|
| +----------------------+ |  to disk storage arrays, databases, or third-party target architectures.|
|                          |                                                                        |
| ⚙️ DEMO PARSING PROXY     | 💡 INTERVIEWER LIVE DEMO SHORTCUT:                                     |
| [ Button: Load Sample  ] |  Don't have a statements file on hand? Click the sidebar button to      |
|   (Injects 500 records)  |  populate the framework with a multi-month synthetic Indian bank stream|
|                          |  featuring real-world UPI strings, EMIs, and intentional anomalies.    |
| 🔒 Session Status:       |                                                                        |
| [ Disconnected/No File ] | 🛑 Awaiting statement ingestion to render core execution blocks...     |
+--------------------------+------------------------------------------------------------------------+
```

---

### 3.2 Session State 2: Active Dashboard Shell (Post-Ingestion Summary)

```
+---------------------------------------------------------------------------------------------------+
|  [Icon] AI PERSONAL FINANCE & LOAN REPAYMENT HEALTH ADVISOR                                       |
+---------------------------------------------------------------------------------------------------+
| SIDEBAR CONTROLS         | SESSION SNAPSHOT: HDFC_STATEMENT_UNMASKED.PDF                         |
|                          |                                                                        |
| 📁 DATA INGESTION SUITE  |  [ KPI: Total Vol. Spend ]  [ KPI: Total Ingested Items ] [ KPI: Savings Rate ]|
| File: HDFC_Unmasked.pdf  |  |      ₹1,42,850.00     |  |       342 Entries       | |     28.4%     |  |
| [ Clear Active File ]    |  +-----------------------+  +-------------------------+ +---------------+  |
|                          |                                                                        |
| ⚙️ MODEL CONFIGURATION   | +--------------------------------------------------------------------+ |
| Clustering Model:        | | Tab 1: Category Streams | Tab 2: Loan Default Risk | Tab 3: Fraud  | |
| [ Isolation Forest   v ] | +--------------------------------------------------------------------+ |
| Anomaly Threshold:       | |                                                                    | |
| [ 5% Sensitivity     v ] | |  [TABULAR EXECUTIONS DISPLAY WITHIN THIS SECTION]                  | |
|                          | |                                                                    | |
| 🔒 Session Status:       | +--------------------------------------------------------------------+ |
| [ In-Memory RAM Live  ]  | | 🤖 INTERACTIVE LLM PORTFOLIO SAVINGS ADVISOR                       | |
|                          | | > Ask: "Analyze my monthly transaction anomalies and suggest goals"| |
+--------------------------+------------------------------------------------------------------------+
```

---

### 3.3 Tab 1: Spending Analytics & Classification Breakdown View

```
+---------------------------------------------------------------------------------------------------+
|  Tab 1: Category Streams  |  Tab 2: Loan Default Risk  |  Tab 3: Fraud                             |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  📊 Transaction Stream Segmentation via XGBoost Classifier                                        |
|                                                                                                   |
|  Spend Category       Aggregated   Percentage Allocation Chart                                    |
|  ------------------   ----------   -------------------------------------------------------------  |
|  Rent & Housing       ₹45,000.00   [████████████████████████████████████████] (31.5%)             |
|  EMI / Loan Payback   ₹32,400.00   [██████████████████████████████] (22.7%)                       |
|  Food & Dining        ₹24,150.00   [█████████████████████] (16.9%)                                |
|  Shopping (E-Com)     ₹18,200.00   [████████████████] (12.7%)                                     |
|  Travel & Fuel        ₹11,300.00   [███████████] (7.9%)                                           |
|  Entertainment/Subs   ₹7,200.00    [██████] (5.0%)                                                |
|  Miscellaneous        ₹4,600.00    [████] (3.3%)                                                  |
|                                                                                                   |
|  💳 Normalized Merchant Extraction Substream (NLP Parsed Strings)                                 |
|  • ZOMATO*ORDER / GPay  -> Assigned to [ Food & Dining ]   | Canonical Merchant: Zomato           |
|  • HDFC-EMI-AP742 / ACH -> Assigned to [ EMI / Loan Payback] | Canonical Merchant: HDFC Bank        |
+---------------------------------------------------------------------------------------------------+
```

---

### 3.4 Tab 2: Loan Repayment Risk Score & Interpretability Engine (SHAP)

```
+---------------------------------------------------------------------------------------------------+
|  Tab 1: Category Streams  |  Tab 2: Loan Default Risk  |  Tab 3: Fraud                             |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  🛡️ Machine Learning Credit Risk Evaluation Matrix                                                |
|                                                                                                   |
|  [ GAUGED RISK VALUE: 68 / 100 ] --> TIER STATUS CLASSIFICATION: [ ⚠️ MODERATE-HIGH RISK STATE ]    |
|                                                                                                   |
|  ⚖️ Explainable AI Model Interpretability Stream (SHAP Value Local Feature Contributions)         |
|                                                                                                   |
|  Feature Context                      SHAP Impact Vector Contribution Richtung                    |
|  ----------------------------------   ----------------------------------------------------------  |
|  EMI-to-Income Ratio (>45%)                  +18.4 [████████████████████ -> Pushes Risk Up]       |
|  Discretionary Spend Factor (>35%)            +9.2 [█████████ -> Pushes Risk Up]                  |
|  Missed Auto-Debit Signal Detected           +22.0 [████████████████████████ -> Pushes Risk Up]   |
|  Stable Base Inflow Cadence                  -11.5 [<- ████████████  Pulls Risk Down/Good]        |
|  High Balance Liquidity Factor                -4.1 [<- ████  Pulls Risk Down/Good]                |
|                                                                                                   |
|  🧠 Executive Summary Analysis:                                                                   |
|     Your calculated default threat index sits within an elevated tier due primarily to a missed   |
|     payment string on May 14th (UPI bounce description code) and a high cumulative debt burden.   |
+---------------------------------------------------------------------------------------------------+
```

---

### 3.5 Tab 3: Anomaly & Fraud Detection Stream View

```
+---------------------------------------------------------------------------------------------------+
|  Tab 1: Category Streams  |  Tab 2: Loan Default Risk  |  Tab 3: Fraud                             |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  🚨 Statistical Transaction Anomalies Isolated via Unsupervised Isolation Forest                  |
|                                                                                                   |
|  Date        Merchant/Description       Amount     Time      Isolation Reason        User Override|
|  ----------  -------------------------  ---------  --------  ----------------------  -------------|
|  2026-05-18  RELIANCE DIGITAL METRO     ₹48,999.00 02:14 AM  Out-of-hours / Scale    [ Dismiss ]  |
|  2026-05-22  CRRED*CLUB*REWARDS_MEMB    ₹12,500.00 11:45 PM  Freq. Threshold Spike   [ Dismiss ]  |
|  2026-06-02  WINE SHOP NOIDA SECTOR 15  ₹6,400.00  04:11 PM  Uncharacteristic target [ Dismiss ]  |
|                                                                                                   |
|  ℹ️ UX Note: Clicking '[ Dismiss ]' passes a session-bound flag into state to exclude the         |
|     transaction line item from downstream summaries transmitted into the Groq LLM prompt system.  |
+---------------------------------------------------------------------------------------------------+
```

---

### 3.6 Groq Chat Advisor Interface & Structured Savings Target Card

```
+---------------------------------------------------------------------------------------------------+
| 🤖 INTERACTIVE GROQ LLM ADVISOR CHAT (Aggregated context ingestion safely localized)             |
+---------------------------------------------------------------------------------------------------+
| Context: Vol Spend: ₹1,42,850 | Savings Rate: 28.4% | Score: 68 Risk | Anomaly Flags: 3 Unresolved|
|                                                                                                   |
| Bot: I have finalized your financial risk summary review. I found three specific actionable       |
|      savings vectors to reduce default probability and optimize cash management:                  |
|                                                                                                   |
|      🎯 PROPOSED STRUCTURAL SAVINGS TARGETS FOR UPCOMING MONTHLY TARGET CYCLES:                   |
|      +-----------------------+-------------------------+---------------------------------------+  |
|      | Target Spend Category | Target Reduction Matrix | Projected Capital Impact Result      |  |
|      +-----------------------+-------------------------+---------------------------------------+  |
|      | Food & Dining         | Trim ₹6,500.00 (Zomato) | Increments savings rate by +4.5%      |  |
|      | Entertainment/Subs    | Cut ₹2,200.00 (Passive) | Reallocates emergency cash shield     |  |
|      | Unresolved Anomalies  | Audit ₹6,400.00 Item    | Protects system account integrity     |  |
|      +-----------------------+-------------------------+---------------------------------------+  |
|                                                                                                   |
| User: Tell me more about why the target optimization matrix selected the Zomato spend line.       |
|                                                                                                   |
| 📥 [ Ask a question regarding credit profile rehabilitation or spend optimizations...       ]  |
+---------------------------------------------------------------------------------------------------+
```

---

## 4. State Management, Visual Language & Interactive Flows

### 4.1 Component Interaction Matrix

| Dashboard Component | Action Initiated | Direct Inter-Component System Effect | Caching Strategy Used |
| :--- | :--- | :--- | :--- |
| **Job Market: Sidebar Filters** | Modified dropdown item | Recalculates metrics, rebuilds spaCy skill vector array frequencies, updates Plotly bar counts. | `st.cache_data` (TTL 3600 seconds) |
| **Job Market: Resume Text Area** | Inserted custom text | Re-initializes TF-IDF matrix transformations, computes Cosine Similarity array scores instantly. | Session Volatile Memory |
| **Finance: File Upload Drop** | Ingested PDF file | Clears previous execution memory states, runs pdfplumber parsing stream, runs extraction pipelines. | Dynamic Session State Storage |
| **Finance: Fraud Flag Row** | Pressed '[ Dismiss ]' | Modifies item's boolean state value to true, triggers instantaneous data layout item refresh. | Session Component Array Mutation |

### 4.2 Streamlit Layout Best Practices & Theme Settings
- **Visual Typography:** Primary UI elements use native text size hierarchies. Sub-elements and mathematical context blocks utilize italics and subtle text formatting for clean separation.
- **Alert Components (`st.alert` styling):** - `Low Risk` / `Exceptional Match` -> Rendered via clean green validation states.
  - `Moderate Risk` / `Borderline Fit` -> Rendered via amber warning indicator configurations.
  - `High Risk Threat Matrix` -> Rendered via explicit red structural alert containers.
- **Progress Tracking Matrix:** Multi-second backend workloads (e.g., spaCy tokenization parses, Isolation Forest scoring executions, PDF extraction tasks) require explicit deployment inside `st.spinner()` blocks to ensure a continuous runtime responsiveness layout.

---
*End of Layout Specification Document*
