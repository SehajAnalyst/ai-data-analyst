# 🤖 AI Data Analyst – Natural Language to SQL Analytics Platform

An AI-powered analytics platform that enables users to interact with SQL databases using plain English. The application converts natural language questions into SQL queries, executes them securely, visualizes the results, and generates AI-powered business insights.

---

## 🚀 Live Application

Experience the deployed AI Data Analyst Platform:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-dataa-analyst.streamlit.app/)

🔗 **Live Demo:** [AI Data Analyst Platform](https://ai-dataa-analyst.streamlit.app/)

## 🚀 Features

- 💬 Ask questions in plain English
- 🧠 AI-powered Natural Language → SQL generation
- ✅ Secure SQL validation before execution
- 🗄️ Supports SQLite databases
- 📊 Automatic table and chart generation
- 📈 AI-generated business insights
- 💾 Persistent conversation history
- ✏️ Rename conversations
- 🗑️ Delete conversations
- 🆕 New Chat functionality
- 🧠 Context-aware follow-up questions
- 📄 Export conversations to PDF
- ⚡ Fast interactive Streamlit interface

---

## 🏗️ System Architecture

```
User Question
      │
      ▼
Prompt Builder
      │
      ▼
Groq LLM
      │
      ▼
SQL Generator
      │
      ▼
SQL Validator
      │
      ▼
Database Execution
      │
      ▼
Result Analysis
      │
      ▼
Charts + Business Insights
      │
      ▼
Conversation History
      │
      ▼
PDF Export
```

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python
- SQLAlchemy

### Database
- SQLite

### AI / LLM
- Groq LLM

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib
- Streamlit Charts

### Version Control
- Git
- GitHub

---

## 📂 Project Structure

```
app/
config/
core/
db/
exceptions/
logging_setup/
ml_plugins/
scripts/
tests/
utils/
```

---

## 🔒 Security Features

- SQL Validation Pipeline
- Read-only database execution
- Query sanitization
- Schema-aware SQL generation
- Safe execution engine
- Error handling and logging

---

## 💡 Example Questions

- List all customers
- Show total sales by country
- Which genre has the highest revenue?
- Top 10 selling artists
- Show invoices from 2025
- Average invoice amount by customer

---

## 📷 Screenshots

### Home Page
(Add Screenshot)

### SQL Generation
(Add Screenshot)

### Charts & Insights
(Add Screenshot)

### Conversation History
(Add Screenshot)

### PDF Export
(Add Screenshot)

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/SehajAnalyst/ai-data-analyst.git
```

Move into the project

```bash
cd ai-data-analyst
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/main.py
```

---

## 📌 Future Improvements

- PostgreSQL support
- MySQL support
- User authentication
- Dashboard customization
- Multi-user conversation history
- Cloud deployment
- Vector database integration
- RAG-based enterprise analytics

---

## 📄 Resume Highlights

- Developed an AI-powered Natural Language to SQL analytics platform.
- Implemented secure SQL validation and execution pipelines.
- Built interactive dashboards with AI-generated insights.
- Designed persistent conversation management with rename, delete, context-aware follow-up queries, and PDF export.
- Optimized session handling and database persistence for a seamless conversational analytics experience.

---

## 👨‍💻 Author

**Sehaj Oberoi**

GitHub: https://github.com/SehajAnalyst

LinkedIn: *(Add your LinkedIn URL)*

---

## ⭐ If you found this project useful, consider giving it a star!
