# 🤖 AI Data Analyst

### Natural Language → SQL → Analytics Platform

> Ask questions about your database in plain English.
> The platform understands the schema, generates SQL, validates it through a deterministic security layer, executes it against a read-only database, visualizes the results, and turns the data into business insights.

<p align="center">

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit\&logoColor=white)](https://ai-dataa-analyst.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-Database-D71F00)](https://www.sqlalchemy.org/)
[![SQLGlot](https://img.shields.io/badge/SQLGlot-SQL%20AST-orange)](https://sqlglot.com/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/SehajAnalyst/ai-data-analyst)

</p>

---

## 🚀 Live Demo

### Try the application

**[🔗 Launch AI Data Analyst](https://ai-dataa-analyst.streamlit.app/)**

Ask questions such as:

> **"Show the average salary by department."**

or:

> **"Which department has the highest number of employees?"**

The platform handles the SQL generation, validation, execution, visualization, and analysis automatically.

---

# 🌟 Why This Project Stands Out

This project is **not simply an LLM wrapper that generates SQL**.

The core design treats LLM-generated SQL as **untrusted input** and places a deterministic validation layer between the LLM and the database.

### 🔐 1. LLM + Deterministic Security Boundary

```text
User Question
      │
      ▼
   LLM / SQL Generator
      │
      │  Untrusted SQL
      ▼
┌───────────────────────┐
│   SQL Validator       │
│                       │
│ • Syntax validation   │
│ • AST analysis        │
│ • Statement checks   │
│ • Injection checks    │
│ • Schema validation  │
│ • Access policies    │
│ • LIMIT enforcement  │
└───────────────────────┘
      │
      │ Only validated SQL
      ▼
 Read-Only Database
```

This separation is one of the most important architectural decisions in the project.

The LLM **does not get direct authority to execute SQL**.

---

### 🛡️ 2. AST-Based SQL Security

SQL is parsed using **SQLGlot** and validated structurally rather than relying only on fragile string matching.

The validator distinguishes between:

```sql
SELECT * FROM drop_log;
```

and:

```sql
DROP TABLE employees;
```

This allows the security layer to reason about the **SQL structure**, rather than simply searching for dangerous words.

---

### 🚫 3. Statement-Type Enforcement

Only permitted read-only statements are allowed.

The validator rejects:

```text
INSERT   ❌
UPDATE   ❌
DELETE   ❌
DROP     ❌
```

while allowing:

```text
SELECT       ✅
CTE + SELECT ✅
```

Example:

```text
INSERT INTO employees ...
→ REJECTED

DROP TABLE employees
→ REJECTED
```

---

### 🔒 4. Multiple-Statement Protection

Stacked SQL statements are rejected before execution.

For example:

```sql
SELECT * FROM employees;
DROP TABLE employees;
```

is rejected because the request contains multiple statements.

---

### 🗄️ 5. Read-Only Database Boundary

The application uses a read-only database execution path for analytics queries.

This creates another layer of defense even if an unsafe query somehow passes an earlier validation layer.

---

### 🧠 6. Schema-Aware SQL Generation

The application introspects the connected database and provides structural schema information to the SQL-generation pipeline.

The system understands:

* Tables
* Columns
* Data types
* Primary keys
* Foreign keys
* Table relationships

This reduces hallucinated tables and columns and improves SQL generation accuracy.

---

### 🔄 7. Context-Aware Conversational Analytics

The application isn't limited to isolated questions.

Users can ask follow-up questions such as:

```text
User:
Show total sales by country.

AI:
[Results + chart]

User:
Which country had the highest sales?

AI:
[Follow-up analysis]
```

The application maintains conversational context so users can explore data naturally.

---

### 📊 8. Automatic Visualization

The platform analyzes query results and selects an appropriate visualization based on the structure of the returned data.

It supports analytical views such as:

* Bar charts
* Pie charts
* Line charts
* Scatter plots
* Tabular results

The visualization layer also avoids treating identifier columns such as `sale_id` and `product_id` as meaningful analytical metrics.

---

### 💡 9. AI-Powered Business Insights

The platform doesn't stop at:

> "Here are your SQL results."

It can interpret the resulting dataset and generate business-oriented insights such as:

* Key trends
* Highest/lowest performing categories
* Comparisons
* Important metrics
* Potential patterns
* Follow-up analytical questions

This transforms the application from a **Text-to-SQL tool** into an **AI analytics assistant**.

---

### 📄 10. Persistent Conversations & PDF Export

Users can:

* Create new conversations
* Rename conversations
* Delete conversations
* Continue previous conversations
* Ask context-aware follow-ups
* Export conversations to PDF

This makes the application behave more like a persistent analytics workspace rather than a one-shot query interface.

---

# 🏗️ System Architecture

```text
                         ┌─────────────────┐
                         │      User       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   Streamlit UI  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Prompt Builder  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    Groq LLM     │
                         └────────┬────────┘
                                  │
                           Generated SQL
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │    SQL Validator        │
                    │                         │
                    │  • SQL Parsing          │
                    │  • AST Security         │
                    │  • Statement Type       │
                    │  • Injection Checks      │
                    │  • Schema Validation    │
                    │  • Access Policies       │
                    │  • LIMIT Enforcement    │
                    └────────────┬────────────┘
                                 │
                          Validated SQL
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Read-Only DB Engine    │
                    │       SQLAlchemy        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ Query Results │
                         └───────┬───────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
             ┌─────────────┐         ┌────────────────┐
             │ Visualization│         │ AI Insights    │
             └──────┬──────┘         └───────┬────────┘
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                      ┌─────────────────────┐
                      │ Conversation History│
                      └──────────┬──────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │  PDF Export   │
                         └───────────────┘
```

---

# ✨ Core Features

| Feature                     | Description                                        |
| --------------------------- | -------------------------------------------------- |
| 💬 Natural Language Queries | Ask database questions using plain English         |
| 🧠 Text-to-SQL              | LLM converts questions into SQL                    |
| 🔍 Schema Introspection     | Automatically discovers database structure         |
| 🛡️ SQL Security Layer      | Deterministic validation before execution          |
| 🌳 AST Validation           | Structural SQL analysis using SQLGlot              |
| 🚫 Statement Protection     | Blocks write/destructive SQL statements            |
| 🔒 Read-Only Execution      | Analytics queries execute through a read-only path |
| 🧪 Injection Protection     | Detects and rejects suspicious SQL patterns        |
| 📏 LIMIT Enforcement        | Prevents unnecessarily large result sets           |
| 📊 Automatic Charts         | Selects visualizations based on result structure   |
| 💡 AI Insights              | Converts query results into business insights      |
| 🔄 Follow-Up Questions      | Supports conversational analytics                  |
| 💾 Persistent History       | Stores and manages conversations                   |
| ✏️ Conversation Rename      | Organize analytics sessions                        |
| 🗑️ Conversation Delete     | Remove unwanted sessions                           |
| 📄 PDF Export               | Export analytical conversations                    |
| ⚡ Interactive UI            | Fast Streamlit-based experience                    |

---

# 🔐 Security Model

The security architecture follows a **defense-in-depth** approach.

### Validation Pipeline

```text
Raw LLM SQL
     │
     ▼
Empty Input Check
     │
     ▼
Comment Detection
     │
     ▼
SQL Parsing
     │
     ▼
Multiple Statement Check
     │
     ▼
Statement Type Check
     │
     ▼
Forbidden AST Node Check
     │
     ▼
Injection Pattern Check
     │
     ▼
CTE / Alias Resolution
     │
     ▼
Table Validation
     │
     ▼
Column Validation
     │
     ▼
Access Policy
     │
     ▼
LIMIT Enforcement
     │
     ▼
Validated SQL
     │
     ▼
Read-Only Execution
```

### Security Test Results

| Test Case           | Expected | Result |
| ------------------- | -------: | -----: |
| Normal `SELECT`     |    Allow |      ✅ |
| CTE `SELECT`        |    Allow |      ✅ |
| `INSERT`            |    Block |      ✅ |
| `UPDATE`            |    Block |      ✅ |
| `DELETE`            |    Block |      ✅ |
| `DROP`              |    Block |      ✅ |
| Multiple statements |    Block |      ✅ |

Example:

```text
SELECT * FROM employees
→ VALID = True

WITH x AS (...) SELECT * FROM x
→ VALID = True

INSERT INTO employees ...
→ VALID = False

UPDATE employees SET ...
→ VALID = False

DELETE FROM employees
→ VALID = False

DROP TABLE employees
→ VALID = False

SELECT * FROM employees; DROP TABLE employees
→ VALID = False
```

---

# 🔄 Example Workflow

### User

> Show the average salary by department.

### System

```text
Natural Language
       ↓
Schema Context
       ↓
LLM SQL Generation
       ↓
SQL Validation
       ↓
Read-Only Execution
       ↓
Result DataFrame
       ↓
Chart Selection
       ↓
AI Insight Generation
```

### Output

The user receives:

* Generated SQL
* Query results
* Visualization
* Key business insights
* Follow-up analytical questions

---

# 🧰 Technology Stack

### Application

* Python
* Streamlit
* SQLAlchemy

### AI

* Groq LLM
* Natural Language → SQL generation
* AI-powered result analysis

### SQL & Security

* SQLGlot
* SQL Abstract Syntax Trees
* Deterministic validation rules
* Schema-aware validation

### Data

* SQLite
* Pandas
* NumPy

### Visualization

* Matplotlib
* Streamlit Charts

### Reporting

* ReportLab

### Development

* Git
* GitHub
* Pytest

---

# 📂 Project Structure

```text
ai-data-analyst/
│
├── app/
│   └── ...
│
├── config/
│   └── security_rules.yaml
│
├── core/
│   ├── nl2sql/
│   │   ├── sql_generator.py
│   │   ├── sql_validator.py
│   │   └── security_rules_loader.py
│   │
│   ├── schema/
│   │   ├── schema_introspector.py
│   │   └── schema_context_builder.py
│   │
│   ├── visualization/
│   │   └── chart_selector.py
│   │
│   └── ...
│
├── db/
│   └── connectors/
│       ├── connection_manager.py
│       └── sqlite_connector.py
│
├── exceptions/
├── logging_setup/
├── ml_plugins/
├── scripts/
├── tests/
├── utils/
│
├── requirements.txt
└── README.md
```

---

# 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/SehajAnalyst/ai-data-analyst.git
cd ai-data-analyst
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it

**Windows**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file and provide the required API credentials.

**Never commit `.env` or API keys to GitHub.**

### 6. Run the application

```bash
streamlit run app/main.py
```

---

# 🧪 Testing

Run the test suite with:

```bash
pytest
```

The project also includes direct validation checks for dangerous SQL statements and multiple-statement payloads.

---

# 📈 Future Roadmap

### Database

* PostgreSQL support
* MySQL support
* Additional SQL dialects

### AI

* Text-to-SQL evaluation benchmark
* SQL generation accuracy tracking
* Improved self-correction loops
* Query-plan-aware generation

### Security

* Role-based database access
* Fine-grained table/column permissions
* Query cost controls
* Rate limiting
* Enhanced audit logging

### Analytics

* Custom dashboards
* Advanced visualizations
* Query performance analysis
* Scheduled reports

### Platform

* Multi-user support
* Authentication
* Cloud-native deployment
* Production observability

---

# 📌 Resume Highlights

* Built an end-to-end **AI-powered Natural Language-to-SQL analytics platform** that converts plain-English questions into validated SQL and business insights.
* Implemented a **deterministic AST-based SQL security layer** using SQLGlot to validate LLM-generated queries before database execution.
* Designed a **read-only database execution boundary** with schema-aware validation, statement restrictions, injection checks, and result-size controls.
* Developed **conversational analytics** with persistent history, context-aware follow-up questions, conversation management, and PDF export.
* Built automatic **result visualization and AI-powered business insight generation** to transform SQL query results into actionable analytics.

---

# 🎯 What This Project Demonstrates

This project demonstrates practical experience with:

```text
LLM Applications
       │
       ├── Prompt Engineering
       ├── Text-to-SQL
       └── AI Insight Generation
       │
       ▼
Data Engineering
       │
       ├── Schema Introspection
       ├── SQLAlchemy
       └── Database Execution
       │
       ▼
AI Security
       │
       ├── AST Validation
       ├── Injection Protection
       ├── Read-Only Execution
       └── Deterministic Guardrails
       │
       ▼
Analytics Engineering
       │
       ├── DataFrames
       ├── Visualization
       └── Business Insights
       │
       ▼
Application Engineering
       │
       ├── Streamlit
       ├── Persistence
       ├── Error Handling
       └── PDF Reporting
```

The key engineering principle is:

> **The LLM generates the SQL, but the LLM does not decide whether that SQL is safe to execute.**

That responsibility belongs to the deterministic validation and execution layers.

---

# 👨‍💻 Author

**Sehaj Oberoi**

AI / Data Engineering • Machine Learning • LLM Applications • Text-to-SQL

**GitHub:** [SehajAnalyst](https://github.com/SehajAnalyst)

**Live Project:** [AI Data Analyst](https://ai-dataa-analyst.streamlit.app/)

---

## ⭐ Support

If you found this project interesting or useful, consider giving the repository a ⭐ on GitHub.
