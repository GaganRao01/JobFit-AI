
import nest_asyncio

import streamlit as st
import os
import json
import tempfile
import base64
import time
import requests
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
from docx import Document
from firecrawl import FirecrawlApp

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, Optional, Literal

import spacy
from spacy import displacy
import asyncio  # To deal with event loop in certain environments

# SerpAPI imports
from serpapi import GoogleSearch

nest_asyncio.apply()

st.set_page_config(
    page_title="JobFit-AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


load_dotenv()

# --- Configuration and Initialization ---

# Initialize session state  
if "resume_analysis" not in st.session_state:
    st.session_state.resume_analysis = None
if "resume_file_path" not in st.session_state:
    st.session_state.resume_file_path = None
if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "job_url" not in st.session_state:
    st.session_state.job_url = ""
if "current_page" not in st.session_state:
    st.session_state.current_page = "upload"
if "relevance_score" not in st.session_state:
    st.session_state.relevance_score = None
if "google_api_key" not in st.session_state:  
    st.session_state.google_api_key = ""  
if "firecrawl_api_key" not in st.session_state:
    st.session_state.firecrawl_api_key = ""  
if "serpapi_key" not in st.session_state:
    st.session_state.serpapi_key = ""  



# Load spaCy model (large model for better accuracy)
@st.cache_resource()
def load_spacy_model():
    try:
        return spacy.load("en_core_web_lg")
    except OSError:
        st.warning("Downloading spaCy model 'en_core_web_lg'. This may take a few minutes...")
        try:
            spacy.cli.download("en_core_web_lg")  
        except Exception as e:
            st.error(
                f"Error downloading spaCy model: {e}. Please ensure you have an internet connection and sufficient permissions."
            )
            st.stop()
        return spacy.load("en_core_web_lg")  


nlp = load_spacy_model()


# Define state types for LangGraph
class ResumeAnalysisState(TypedDict):
    resume_text: str
    job_description: str
    extracted_sections: Optional[Dict[str, Any]]
    analysis_result: Optional[Dict[str, Any]]
    error: Optional[str]
    relevance_score: Optional[Dict[str, Any]]


# --- Helper Functions ---

def extract_resume_text(file_path):
    """Extracts text from PDF or DOCX resumes."""
    if not os.path.exists(file_path):
        return None, "File not found."

    ext = file_path.lower().split(".")[-1]
    try:
        if ext == "pdf":
            return extract_text(file_path), None
        elif ext == "docx":
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs]), None
        else:
            return None, "Unsupported file format. Please upload a PDF or DOCX file."
    except Exception as e:
        st.error(f"Error reading resume file: {e}") 
        return None, f"Error reading file: {e}"


from bs4 import BeautifulSoup  

class Colors:  # For colored output 
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

def scrape_indeed_with_firecrawl_html(url, api_key):
    """
    Attempts to scrape Indeed job postings using Firecrawl's HTML format
    and Beautiful Soup.
    """
    print(f"{Colors.YELLOW}Trying Firecrawl HTML for Indeed: {url}{Colors.RESET}")
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload = {
        "url": url,
        "formats": ["html"]
    }
    try:
        response = requests.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers=headers,
            json=payload,
            timeout=120  # Increased timeout
        )
        response.raise_for_status()
        result = response.json()

        if result.get('success'):
            html_content = result['data']['html']
            soup = BeautifulSoup(html_content, 'html.parser')

            # Indeed-specific parsing 
            description_element = soup.find('div', class_='jobsearch-jobDescriptionText')  
            if description_element:
                print(f"{Colors.GREEN}Indeed description found with Firecrawl.{Colors.RESET}")
                return description_element.get_text(separator='\n', strip=True), None 
            else:
                
                description_element = soup.find('div', id='jobDescriptionText')
                if description_element:
                    print(f"{Colors.GREEN}Indeed description found with Firecrawl.{Colors.RESET}")
                    return description_element.get_text(separator='\n', strip=True), None  
                print(f"{Colors.YELLOW}Indeed description NOT found with Firecrawl.{Colors.RESET}")
                return None, "Indeed description NOT found with Firecrawl." 
        else:
            error_message = f"Firecrawl error for Indeed: {result.get('error')}"
            print(f"{Colors.RED}{error_message}{Colors.RESET}")
            return None, error_message  

    except requests.exceptions.RequestException as e:
        error_message = f"Request error for Indeed (Firecrawl): {e}"
        print(f"{Colors.RED}{error_message}{Colors.RESET}")
        return None, error_message 
    except Exception as e:
        error_message = f"Other error on Indeed: {e}"
        print(f"{Colors.RED}{error_message}{Colors.RESET}")
        return None, error_message   

def get_job_data_from_serpapi(job_url, serp_api_key):
    """
    Uses SerpApi's Google Jobs Listing engine to get job data.  Takes SerpAPI key.
    """
    print(f"{Colors.YELLOW}Trying SerpApi for: {job_url}{Colors.RESET}")
    try:
        params = {
            "engine": "google_jobs_listing",
            "q": job_url,
            "api_key": serp_api_key 
        }
        search = GoogleSearch(params) 
        results = search.get_dict()
       
        if 'description' in results:

            print(f"{Colors.GREEN}Job data found with SerpApi.{Colors.RESET}")
            return results.get('description', ""), None 
        else:
            error_message = "Job data NOT found with SerpApi."
            print(f"{Colors.YELLOW}{error_message}{Colors.RESET}")
            return None, error_message 

    except Exception as e:
        error_message = f"Error using SerpApi: {e}"
        print(f"{Colors.RED}{error_message}{Colors.RESET}")
        return None, error_message 
def extract_job_description_from_url(job_url):
    """
    Main function to extract the job description, trying different methods.
    """
    # Check for API keys in session state.
    if not st.session_state.firecrawl_api_key:
        return None, "Firecrawl API key is not set."
    if not st.session_state.serpapi_key:
        return None, "SerpAPI key is not set."

    # Check the URL and decide on strategy
    if "linkedin.com/jobs/view" in job_url:
        # LinkedIn: Go straight to SerpApi
        print(f"{Colors.CYAN}Detected LinkedIn URL. Using SerpApi...{Colors.RESET}")
        return get_job_data_from_serpapi(job_url, st.session_state.serpapi_key) 


    elif "indeed.com" in job_url:  
        
        print(f"{Colors.CYAN}Detected Indeed URL. Trying Firecrawl...{Colors.RESET}")
        description, error = scrape_indeed_with_firecrawl_html(job_url, st.session_state.firecrawl_api_key)
        if description:
            return description, None 

        print(f"{Colors.YELLOW}Falling back to SerpApi for Indeed...{Colors.RESET}")
        return get_job_data_from_serpapi(job_url, st.session_state.serpapi_key) 


    else:
        print(f"{Colors.YELLOW}Unsupported URL: {job_url}{Colors.RESET}")
        return None, f"Unsupported URL: {job_url}"

    print(f"{Colors.RED}Could not extract job description automatically.{Colors.RESET}")
    return None, "Could not extract job description automatically."




def get_pdf_download_link(analysis_text, filename="JobFit-AI_Resume_Analysis.txt"):
    """Generates a link to download the analysis as plain text."""
    try:
        b64 = base64.b64encode(analysis_text.encode()).decode()
        return f'<a href="data:text/plain;base64,{b64}" download="{filename}">Download Analysis Report (TXT)</a>'
    except Exception as e:
        st.error(f"Error while generating download link {e}")
        return f"Download link generation failed: {e}"


# --- LangChain Components ---

@st.cache_resource()
def get_llm():
     
    try:
        if not st.session_state.google_api_key:
            st.error("Google API key is not set. Please enter it in the sidebar.")
            st.stop() 
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-002",
            temperature=0.7,
            google_api_key=st.session_state.google_api_key,  
            convert_system_message_to_human=True,
        )
    except Exception as e:
        st.error(f"Error Initializing LLM: {e}")
        st.stop()

# Named Entity Recognition (NER)
def extract_entities(text):
    """Extracts named entities using spaCy."""
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    return entities


def extract_sections(state: ResumeAnalysisState) -> ResumeAnalysisState:
    """Extract key sections from the resume text using spaCy and LLM."""
    try:
        doc = nlp(state["resume_text"])

        extracted_sections = {
            "profile_summary": "",
            "skills": [],
            "publications": [],
            "internships": [],
            "experience": [],
            "projects": [],
            "certifications": [],
            "education": [],
        }

        # Profile Summary ( LLM for better summarization)
        summary_prompt = ChatPromptTemplate.from_template(
            """
            Summarize the following text in 2-3 sentences, focusing on key professional highlights:
            {resume_text}
            """
        )
        llm = get_llm()
        summary_chain = summary_prompt | llm
        summary = summary_chain.invoke({"resume_text": state["resume_text"]}).content
        extracted_sections["profile_summary"] = summary

        # Skills ( spaCy's NER and pattern matching)
        skill_entities = extract_entities(state["resume_text"]).get(
            "SKILL", []
        )  

        for sent in doc.sents:
            if "skill" in sent.text.lower():
                for token in sent:
                    if token.dep_ in ("dobj", "pobj", "conj") and token.pos_ in (
                        "NOUN",
                        "PROPN",
                    ):
                        extracted_sections["skills"].append(token.text)
        extracted_sections["skills"] = list(
            set(extracted_sections["skills"] + skill_entities)
        )

        # Basic keyword matching for other sections
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if "publication" in sent_text and isinstance(extracted_sections["publications"], list):
                extracted_sections["publications"].append(sent.text)
            if "internship" in sent_text and isinstance(extracted_sections["internships"], list):
                extracted_sections["internships"].append(sent.text)
            if "experience" in sent_text or "worked as" in sent_text and isinstance(extracted_sections["experience"], list):
                extracted_sections["experience"].append(sent.text)
            if "project" in sent_text and isinstance(extracted_sections["projects"], list):
                extracted_sections["projects"].append(sent.text)
            if "certification" in sent_text or "certified in" in sent_text and isinstance(extracted_sections["certifications"], list):
                extracted_sections["certifications"].append(sent.text)
            if "education" in sent_text or "degree" in sent_text and isinstance(extracted_sections["education"], list):
                extracted_sections["education"].append(sent.text)


        return {**state, "extracted_sections": extracted_sections, "error": None}
    except Exception as e:
        return {**state, "error": f"Section extraction error: {str(e)}"}
    


# Defining analysis node 
def analyze_resume(state: ResumeAnalysisState) -> ResumeAnalysisState:
    """Analyze the resume against the job description, heavily using the LLM."""
    try:
        if "error" in state and state["error"]:
            return state

        analysis_prompt = ChatPromptTemplate.from_template(
            """
        You are a highly skilled AI resume analyst.  Your task is to deeply analyze a candidate's resume in relation to a specific job description.  Provide *detailed* and *personalized* feedback, addressing the user directly (using "you" and "your"). Go beyond simple keyword matching; consider the context and nuance of both the resume and the job description.

        **Resume Text:**
        ```text
        {resume_text}
        ```

        **Job Description:**
        ```text
        {job_description}
        ```

        **Instructions:**

        1.  **Profile Summary:**  Assess how well the candidate's profile, as presented in their resume, aligns with the job requirements. Offer specific suggestions for improvement.

        2.  **Skills Assessment:**
           *   **Current Skills:** Identify skills present in the resume that are relevant to the job. For *each* skill, explain *why* it's relevant, referencing specific requirements from the job description.
           *   **Missing Skills:** Identify *critical* skills mentioned in the job description that are *not* evident in the resume. Explain why each missing skill is important and suggest how the candidate might acquire it.

        3.  **Experience Evaluation:** Analyze the candidate's work experience and projects.  Are they relevant to the job?  Do they demonstrate the required skills and competencies? Suggest how the candidate could better highlight relevant experience or gain additional experience if needed.

        4.  **Education and Certifications:** Assess the candidate's education and certifications. Are they sufficient for the role? Suggest additional qualifications that might be beneficial.

        5.  **Job Role Recommendations:** Based on the resume, suggest 3-5 job roles (including the target role, if applicable) that the candidate would be well-suited for.  Explain *why* each role is a good fit.

        6.  **Personalized Recommendations:**  Provide *very specific*, actionable recommendations for:
           *   **Resume Improvement:**  Suggest concrete changes to the resume's content, structure, and wording to better target the job description.
           *   **Skill Development:** Recommend specific courses, books, projects, or activities to address skill gaps.  Include links where possible.
           *   **Interview Preparation:**  Offer tailored advice on how the candidate can best prepare for an interview for this type of role, anticipating likely questions.
           *.  **Recommended Videos** Provide actual youtube links
        7. **Overall Match Score (Estimate):** Estimate an overall match score (out of 100) representing how well the resume aligns with the job description.

        **Output Format:**
        Return a JSON object with the following structure (use the keys exactly as shown):

        ```json
        {{
          "profile_summary": "...",
          "current_skills": [{{ "skill": "...", "relevance": "..." }}, ...],
          "skills_to_add": [{{ "skill": "...", "reason": "..." }}, ...],
          "experience_evaluation": "...",
          "education_evaluation": "...",
          "predicted_job_roles": [{{ "role": "...", "suitability": "..." }}, ...],
          "resume_tips": [{{ "tip": "...", "explanation": "..." }}, ...],
          "interview_tips": [{{ "tip": "...", "explanation": "..." }}, ...],
          "recommended_courses": [{{ "course": "...", "benefit": "...", "link": "..." }}, ...],
          "recommended_videos": [{{ "topic": "...", "reason": "...", "link": "..."}}, ...],
          "overall_match_score": 0  // A numerical score (0-100)
        }}
        ```
        """
        )

        llm = get_llm()  
        chain = analysis_prompt | llm | JsonOutputParser()
        analysis_result = chain.invoke(
            {
                "resume_text": state["resume_text"],  
                "job_description": state["job_description"],
            }
        )

        return {**state, "analysis_result": analysis_result, "error": None}
    except Exception as e:
        return {**state, "error": f"Analysis error: {str(e)}"}


# Define error handling node
def handle_error(state: ResumeAnalysisState) -> Literal["end"]:
    """Handle errors in the workflow"""
    st.error(f"An error occurred: {state.get('error', 'Unknown error')}")  # Show error on Streamlit
    return "end"


# Define routing function
def should_continue(state: ResumeAnalysisState) -> Literal["analyze", "error"]:
    """Determine if we should continue to analysis or handle an error"""
    if "error" in state and state["error"]:
        return "error"
    return "analyze"


def calculate_relevance_score(state: ResumeAnalysisState) -> ResumeAnalysisState:
    """Calculates a relevance score based on skills, experience, and education."""

    if state.get("error") or not state.get("analysis_result"):
        return {**state, "relevance_score": {"overall_match_score": 0}}

    analysis_result = state["analysis_result"]
    if analysis_result is None: 
        return {**state, "relevance_score": {"overall_match_score": 0}}

    # Use the LLM-provided score if available; otherwise, calculate a basic score.
    overall_score = analysis_result.get("overall_match_score", 0)
    if overall_score == 0:  # If overall score somehow got set to 0.
        skill_match_score = 0
        if analysis_result.get("current_skills"):
            skill_match_score = len(analysis_result["current_skills"]) * 4  # Example weighting

        experience_weighting = 0
        if analysis_result.get("predicted_job_roles"):
            experience_weighting = (
                len(analysis_result["predicted_job_roles"]) * 3
            )  # Example weighting

        education_score = 0

        extracted_education = state.get("extracted_sections", {}).get("education")
        if extracted_education and isinstance(extracted_education, list):
             education_score = len(extracted_education) * 2

        overall_score = min(
            100, skill_match_score + experience_weighting + education_score
        )
        relevance_scores = {
            "overall_match_score": overall_score,
            "skill_match_score": skill_match_score,
            "experience_weighting": experience_weighting,
            "education_score": education_score,
        }

    else:
        relevance_scores = {
            "overall_match_score": overall_score,
            "skill_match_score": 0,  
            "experience_weighting": 0,
            "education_score": 0,
        }

    return {**state, "relevance_score": relevance_scores}


# --- LangGraph Workflow ---


@st.cache_resource()
def create_workflow():
    workflow = StateGraph(ResumeAnalysisState)
    workflow.add_node("extract_sections", extract_sections)
    workflow.add_node("analyze_resume", analyze_resume)
    workflow.add_node("calculate_relevance_score", calculate_relevance_score)  # Add scoring
    workflow.add_node("handle_error", handle_error)

   
    workflow.set_entry_point("extract_sections")

    # Define edges using conditional logic
    workflow.add_conditional_edges(
        "extract_sections",  # Source node
        should_continue,  # Routing function
        {
            "analyze": "analyze_resume",  
            "error": "handle_error",  
        },
    )

    workflow.add_edge(
        "analyze_resume", "calculate_relevance_score"
    )  
    workflow.add_edge("calculate_relevance_score", END)
    workflow.add_edge("handle_error", END)

    
    return workflow.compile()

# --- Streamlit UI Components ---

def display_analysis_results(analysis, relevance_score):
    """Displays the analysis results in a user-friendly format."""
    # Profile Summary
    if "profile_summary" in analysis and analysis["profile_summary"]:
        st.write("### 📋 Profile Summary")
        st.markdown(
            f"<div style='background-color:#2c2f33;padding:15px;border-radius:10px;'>{analysis['profile_summary']}</div>",
            unsafe_allow_html=True,
        )

    # Overall Match Score
    st.write("### 🎯 Overall Match Score")
    score = relevance_score.get("overall_match_score", 0)  
    st.markdown(f"<h4 style='margin-bottom:0px;'>{score}/100</h4>", unsafe_allow_html=True)
    st.progress(score / 100)

   
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("### 💡 Detailed Analysis & Feedback")

    
    if "experience_evaluation" in analysis and analysis["experience_evaluation"]:
        st.markdown("#### Experience Evaluation")
        st.markdown(analysis["experience_evaluation"])

    if "education_evaluation" in analysis and analysis["education_evaluation"]:
        st.markdown("#### Education Evaluation")
        st.markdown(analysis["education_evaluation"])

    # Current Skills and Skills to Add in columns
    col1, col2 = st.columns(2)

    with col1:
        st.write("### 🔍 Current Skills")
        if "current_skills" in analysis:
            for skill in analysis["current_skills"]:
                with st.expander(f"✓ {skill['skill']}"):
                    st.write(skill["relevance"])

    with col2:
        st.write("### 🔧 Skills to Add")
        if "skills_to_add" in analysis:
            for skill in analysis["skills_to_add"]:
                with st.expander(f"+ {skill['skill']}"):
                    st.write(skill["reason"])

    # Suitable Job Roles
    st.write("### 👔 Suitable Job Roles")
    if "predicted_job_roles" in analysis:
        for role in analysis["predicted_job_roles"]:
            with st.expander(f"• {role['role']}"):
                st.write(role["suitability"])

    # Recommended Courses
    st.write("### 📚 Recommended Courses & Certifications")
    if "recommended_courses" in analysis:
        for course in analysis["recommended_courses"]:
            with st.expander(f"• {course['course']}"):
                st.write(course["benefit"])
                if "link" in course and course["link"]:
                    st.markdown(
                        f"[Click here to access the course]({course['link']})",
                        unsafe_allow_html=True,
                    )

    # Resume Tips
    st.write("### 📝 Resume Improvement Tips")
    if "resume_tips" in analysis:
        for i, tip in enumerate(analysis["resume_tips"]):
            with st.expander(f"{i+1}. {tip['tip']}"):
                st.write(tip["explanation"])

    # Interview Tips
    st.write("### 🗣️ Interview Tips")
    if "interview_tips" in analysis:
        for i, tip in enumerate(analysis["interview_tips"]):
            with st.expander(f"{i+1}. {tip['tip']}"):
                st.write(tip["explanation"])

    # Recommended Videos
    st.write("### 🎥 Recommended Videos")
    if "recommended_videos" in analysis:
        for video in analysis["recommended_videos"]:
            with st.expander(f"• {video['topic']}"):
                st.write(video["reason"])
                if "link" in video and video["link"]:
                    st.markdown(
                        f"[Watch the video]({video['link']})",
                        unsafe_allow_html=True
                    )


def process_resume():
    """Process the resume and job description, triggering the LangGraph workflow."""
    if st.session_state.resume_file_path and st.session_state.job_description:
        with st.spinner("Analyzing your resume... This may take up to 60 seconds"):
            # Extract resume text
            resume_text, extract_error = extract_resume_text(
                st.session_state.resume_file_path
            )
            if extract_error:
                return  

            # Prepare the initial state for the LangGraph workflow
            initial_state = {
                "resume_text": resume_text,
                "job_description": st.session_state.job_description,
                "extracted_sections": None,
                "analysis_result": None,
                "error": None,
                "relevance_score": None,
            }

            # Create and run the LangGraph workflow
            try:
                resume_analysis_chain = create_workflow()  # This is cached
                final_state = resume_analysis_chain.invoke(initial_state)

                if final_state.get("error"):
                    return 

                # Store results in session state
                st.session_state.resume_analysis = final_state["analysis_result"]
                st.session_state.relevance_score = final_state["relevance_score"]
                st.success("Analysis complete!")

                # Switch to results page and re-run
                st.session_state.current_page = "results"
                st.rerun()

            except Exception as e:
                st.error(f"An unexpected error occurred during processing: {e}")
                return



def upload_page():
    """UI for uploading the resume and entering/pasting the job description."""
    # Centered Title and Subtitle
    st.markdown(
        "<h1 style='text-align: center; color: #4F46E5;'>JobFit-AI</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; margin-bottom: 2rem;'>Your Personalized AI Powered Resume Analyzer</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True) 

    st.write("### Upload Your Resume and Job Description")

    col1, col2 = st.columns(2)

    with col1:
        resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
        if resume_file:
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, resume_file.name)
            with open(temp_path, "wb") as f:
                f.write(resume_file.getbuffer())
            st.session_state.resume_file_path = temp_path
            st.success(f"Resume uploaded: {resume_file.name}")

    with col2:
        job_input_method = st.radio(
            "Job Description Input Method", ["Enter Job URL", "Paste Job Description"]
        )

        if job_input_method == "Enter Job URL":
            job_url = st.text_input(
                "Job Posting URL", placeholder="https://www.example.com/job/12345"
            )
            if job_url and job_url != st.session_state.job_url:
                with st.spinner("Extracting job description..."):
                    st.session_state.job_url = job_url
                    job_desc, error = extract_job_description_from_url(job_url) 
                    if job_desc:
                        st.session_state.job_description = job_desc  
                        st.success("Job description extracted successfully!")
                    else:
                        st.error(f"Couldn't extract job description: {error}") 
                        st.warning("Please copy and paste the job description manually.")
                        st.session_state.job_description = ""  
            if st.session_state.job_description: 
                st.session_state.job_description = st.text_area(
                    "Extracted Job Description (you can edit)",
                    st.session_state.job_description,
                    height=200,
                )
        else: 
            st.session_state.job_description = st.text_area(
                "Job Description",
                st.session_state.job_description,
                height=200,
                placeholder="Paste the job description here...",
            )
    st.markdown("<br>", unsafe_allow_html=True)  

   
    if st.button(
        "Analyze Resume",
        type="primary",
        disabled=not (
            st.session_state.resume_file_path and st.session_state.job_description
        ),
    ):
        process_resume()



def results_page():
    """Displays the analysis results and provides navigation options."""
    
    if st.button("Start New Analysis", type="primary"):
        # Clear relevant session state variables
        st.session_state.resume_analysis = None
        st.session_state.resume_file_path = None
        st.session_state.job_description = ""
        st.session_state.job_url = ""
        st.session_state.relevance_score = None
        st.session_state.current_page = "upload"
        st.rerun()  

    
    if st.session_state.resume_analysis and st.session_state.relevance_score:
        
        tab1, tab2, tab3 = st.tabs(["Analysis Results", "Job Description", "Raw Resume Data"])

        with tab1:
            display_analysis_results(
                st.session_state.resume_analysis, st.session_state.relevance_score
            )

            
            st.markdown("---")
            analysis_text = json.dumps(
                st.session_state.resume_analysis, indent=2
            ) 
            download_link = get_pdf_download_link(analysis_text)
            st.markdown(download_link, unsafe_allow_html=True)

        with tab2:
            st.write("### Job Description")
            st.write(st.session_state.job_description)

        with tab3:
            st.write("### Resume Data")
            if st.session_state.resume_file_path:
                st.write(
                    f"Resume File: {os.path.basename(st.session_state.resume_file_path)}"
                )
                resume_text, _ = extract_resume_text(st.session_state.resume_file_path)
                if resume_text:
                    st.text(resume_text)

    else:
        st.write(
            "No analysis data available. Please upload a resume and job description on the previous page."
        )

    
    with st.expander("See Example Analysis"):
        st.write(
            """
        ### Example Analysis Results

        This is a preview of what your personalized resume analysis will look like.  Upload your resume and a job description to get started!

        **Features include:**
        - Match score assessment
        - Skill gap analysis
        - Recommended courses with direct links
        - Interview preparation tips
        - Resume improvement suggestions
        - Job role recommendations
        - Recommended videos with YouTube links
        """
        )

        st.image("https://via.placeholder.com/800x400.png?text=JobFit-AI+Example+Analysis", use_container_width=True)

def dashboard_page():
    """Placeholder for now.  We'll import and call the dashboard function here."""
    import dashboard
    dashboard.show_dashboard()


def main():
    """Main function to run the Streamlit app."""
    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        # API Key Inputs
        st.session_state.google_api_key = st.text_input(
            "Google Gemini API Key",
            value=st.session_state.google_api_key,
            type="password"
        )
        st.session_state.firecrawl_api_key = st.text_input(
            "Firecrawl API Key",
            value=st.session_state.firecrawl_api_key,
            type="password"
        )
        st.session_state.serpapi_key = st.text_input(
            "SerpAPI Key",
            value=st.session_state.serpapi_key,
            type="password"
        )
        st.markdown("---")

        st.markdown("### Advanced Options")
        st.selectbox("AI Model", ["gemini-1.5-pro-002"], disabled=True,
            help="Select the AI model for analysis") 
        st.markdown("---")

        st.markdown("### Features")
        st.checkbox("Enable Cover Letter Generation", value=False, disabled=True,
                   help="Coming soon")  
        st.checkbox("Generate Interview Questions", value=False, disabled=True,
                   help="Coming soon")  
        st.markdown("---")

        #Dashboard Button
        if st.button("Dashboard", key="dashboard_button", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()

        # Home Button 
        if st.button("Home", key="home_button", use_container_width=True):
            st.session_state.current_page = "upload"
            st.rerun()

        st.markdown("---") 
        #About Section
        st.markdown("### About")
        st.markdown("JobFit-AI uses AI to analyze resumes and provide personalized career recommendations.")
        st.markdown("Powered by LangChain, LangGraph, Google Gemini, Firecrawl, and SerpAPI")
        st.markdown("Version 1.0.0")


    # --- Main Content Area: Page Navigation ---
    if st.session_state.current_page == "upload":
        upload_page()
    elif st.session_state.current_page == "results":
        results_page()
    elif st.session_state.current_page == "dashboard":
        dashboard_page()

    # --- Footer  ---
    st.markdown(
        """
        <div style="text-align: center; margin-top: 2rem;">
            <p>JobFit-AI © 2025 | Powered by LangChain, LangGraph, Google Gemini, Firecrawl, and SerpAPI | <a href="#">Privacy Policy</a></p>
            <p>Created by <a href="https://www.linkedin.com/in/gagan-rao">gagan</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )



if __name__ == "__main__":
    main()
