# 🚀 JobFit-AI: Your AI-Powered Career Companion 🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.1-brightgreen)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Enabled-orange)](https://python.langchain.com/docs/langgraph)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini_1.5_Pro-blueviolet)](https://ai.google.dev/gemini-api)

## ✨ Overview

**JobFit-AI** is a powerful, open-source web application designed to revolutionize your job search!  It combines cutting-edge AI with insightful job market data to provide you with:

*   **Personalized Resume Analysis:**  Get in-depth feedback on your resume, tailored to the specific job description you provide.  Go beyond simple keyword matching and understand how well your skills and experience *truly* align with the target role.
*   **Indian Job Market Insights:**  Explore interactive visualizations of job posting trends, top companies, in-demand skills, and more, specifically for the Indian market.  Make data-driven decisions about your career path.

JobFit-AI is built with a modern tech stack, including Streamlit, LangChain, LangGraph, Google Gemini 1.5 Pro, SpaCy, and more.

## 🌟 Features

*   **Resume Upload:**  Supports PDF and DOCX formats.
*   **Job Description Input:**  Paste the job description directly or provide a URL (supports LinkedIn and Indeed).
*   **AI-Powered Analysis:**
    *   **Section Extraction:** Identifies key sections in your resume (profile, skills, experience, education, etc.).
    *   **Detailed Feedback:**  Provides specific, actionable recommendations for improving your resume's content, structure, and wording.
    *   **Skill Gap Analysis:**  Highlights missing skills mentioned in the job description.
    *   **Relevance Score:**  Calculates an overall match score (out of 100) to quantify how well your resume aligns with the job.
    * **Job Role Recommendations:** Suggest job roles that are a great fit for your resume.
    * **Interview Preparation Tips:** Tips on preparing for the interviews.
    * **Recommended Learning Sources:** Provide direct links for learning sources.
*   **Job Market Dashboard (Indian Market):**
    *   **Interactive Visualizations:**  Explore trends in job postings, top job titles, locations, companies, work types, and sectors.
    *   **Filtering:**  Narrow down the data by date range, job title, location, work type, and sector.
    *   **Data-Driven Insights:**  Make informed decisions based on real-world job market data.
* **API Integrations:** SerpAPI and Firecrawl for robust URL parsing.

## 🛠️ Tech Stack

*   **Frontend:**  [Streamlit](https://streamlit.io/) -  A fast and intuitive way to build interactive web apps in Python.
*   **NLP & AI:**
    *   [Google Gemini 1.5 Pro](https://ai.google.dev/gemini-api) - A powerful large language model for in-depth text analysis and feedback generation.
    *   [LangChain](https://www.langchain.com/) - A framework for developing applications powered by language models.
    *   [LangGraph](https://python.langchain.com/docs/langgraph) -  For building stateful, multi-actor applications with LLMs (used for the resume analysis workflow).
    *   [SpaCy](https://spacy.io/) -  For advanced natural language processing tasks (NER, POS tagging).
*   **Web Scraping:**
    *   [Firecrawl](https://firecrawl.dev/) -  For extracting job descriptions from Indeed URLs.
    *   [SerpAPI](https://serpapi.com/) - For extracting job descriptions from LinkedIn URLs and as a fallback for Indeed.
* **Data processing/visualization**: [Pandas](https://pandas.pydata.org/), [Plotly](https://plotly.com/python/)
*   **Other:** `pdfminer.six`, `python-docx`, `beautifulsoup4`, `requests`, `nest_asyncio`, `python-dotenv`.

## 🚀 Getting Started

### Prerequisites

*   Python 3.9 or higher
*   API Keys for:
    *   Google Gemini
    *   Firecrawl
    *   SerpAPI

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/JobFit-AI.git  # Replace your-username
    cd JobFit-AI
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download SpaCy Model**
    ```bash
     python -m spacy download en_core_web_lg
    ```

5.  **Set up API keys:**

    *   Create a `.env` file in the root directory of the project.
    *   Add your API keys to the `.env` file:

        ```
        GOOGLE_API_KEY=your_google_api_key
        FIRECRAWL_API_KEY=your_firecrawl_api_key
        SERPAPI_KEY=your_serpapi_key
        ```

        *Alternatively, you can set these API keys directly in the Streamlit UI (sidebar), but using a `.env` file is generally better for security and organization.*

### Running the Application

```bash
streamlit run app.py
```

This will start the Streamlit application, and you can access it in your web browser (usually at http://localhost:8501).

## 🔥 Try out the app now: [JobFit-AI Live Demo](https://get-jobfit-ai.streamlit.app/)

💻 Usage

Upload Your Resume: On the "Upload" page, upload your resume in PDF or DOCX format.

Provide Job Description: Either paste the job description text directly or enter the URL of the job posting (LinkedIn and Indeed URLs are supported).

Analyze: Click the "Analyze Resume" button. The analysis may take up to a minute.

View Results: The "Results" page will display detailed feedback, a relevance score, and personalized recommendations.

Explore Job Market: Navigate to the "Dashboard" page to explore interactive visualizations of the Indian job market data. Use the filters in the sidebar to customize the data displayed.

