import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np
import os
from groq import Groq
from dotenv import load_dotenv
import tempfile

# SET PAGE CONFIG MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Resume Ranking System", layout="wide")

# Initialize the SentenceTransformer model (cache it)
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# Load environment variables
load_dotenv()

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    groq_api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=groq_api_key)

client = get_groq_client()

# Helper functions (same as your original code)
def normalize_text(data):
    """Improved text normalization that handles nested structures and prioritizes key fields."""
    if isinstance(data, str):
        return data
    elif isinstance(data, list):
        return " ".join(normalize_text(item) for item in data)
    elif isinstance(data, dict):
        if "description" in data:
            return normalize_text(data["description"])
        elif "jobTitle" in data or "Job Title" in data:
            return normalize_text(data.get("jobTitle") or data.get("Job Title"))
        else:
            return " ".join(normalize_text(v) for v in data.values())
    else:
        return str(data)

def get_skills(resume):
    """Case-insensitive skill extraction"""
    for key in ["Skills", "skills", "SKILLS"]:
        if key in resume:
            return resume[key]
    return []

def get_experience(resume):
    """Case-insensitive experience extraction"""
    for key in ["Experience", "experience", "EXPERIENCE"]:
        if key in resume:
            return resume[key]
    return {}

def clamp_score(score):
    """Ensure score is between 0 and 1"""
    return max(0.0, min(1.0, float(score)))

def match_resume_to_job(resume_text, job_description, model):
    """Compute similarity between resume text and job description"""
    embeddings = model.encode([resume_text, job_description], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

def compute_skill_match(skills, job_description):
    """Improved with synonyms, partial matches, and mandatory skills"""
    mandatory_skills = {"python", "sql", "machine learning", "aws", "gcp", "azure"}
    synonyms = {
        "ml": "machine learning", "ai": "artificial intelligence",
        "nlp": "natural language processing", "pytorch": "torch",
        "tensorflow": "tf", "dl": "deep learning", "spark": "apache spark",
    }
    
    job_desc_lower = job_description.lower()
    skill_matches = 0
    
    for skill in skills:
        skill_lower = skill.lower()
        skill_lower = synonyms.get(skill_lower, skill_lower)
        if any(skill_word in job_desc_lower for skill_word in skill_lower.split()):
            skill_matches += 1
    
    missing_mandatory = max(0, len(mandatory_skills - {s.lower() for s in skills}))
    penalty = min(0.5, 0.1 * missing_mandatory)
    
    return max(0, (skill_matches / max(1, len(mandatory_skills))) - penalty)

def rank_candidates(resumes, job_description, model):
    """Final ranking with all fixes applied"""
    scores = []
    for resume in resumes:
        skills = get_skills(resume)
        experience = get_experience(resume)
        education = resume.get("Education") or resume.get("education") or {}
        
        skills_text = normalize_text(skills)
        experience_text = normalize_text(experience)
        education_text = normalize_text(education)
        
        skill_score = clamp_score(match_resume_to_job(skills_text, job_description, model))
        experience_score = clamp_score(match_resume_to_job(experience_text, job_description, model))
        education_score = clamp_score(match_resume_to_job(education_text, job_description, model))
        keyword_score = clamp_score(compute_skill_match(skills, job_description))
        
        total_score = (
            0.4 * skill_score +
            0.4 * experience_score + 
            0.1 * education_score +
            0.1 * keyword_score
        )
        
        name = (
            resume.get("Name") or
            resume.get("contactInformation", {}).get("name") or
            resume.get("Contact Information", {}).get("Name") or
            f"Unknown Candidate {len(scores)+1}"
        )
        
        scores.append({
            "Name": name,
            "Skill Score": skill_score,
            "Experience Score": experience_score,
            "Education Score": education_score,
            "Keyword Match Score": keyword_score,
            "Total Score": clamp_score(total_score)
        })
    
    return sorted(scores, key=lambda x: x["Total Score"], reverse=True)

def generate_plot(ranked_candidates, top_n=10):
    """Generate and return a matplotlib plot"""
    candidates = ranked_candidates[:top_n]
    if not candidates:
        return None

    names = [candidate['Name'] for candidate in candidates]
    categories = ['Skill', 'Experience', 'Education', 'Keyword Match']
    scores = {
        'Skill': [candidate['Skill Score'] for candidate in candidates],
        'Experience': [candidate['Experience Score'] for candidate in candidates],
        'Education': [candidate['Education Score'] for candidate in candidates],
        'Keyword Match': [candidate['Keyword Match Score'] for candidate in candidates],
        'Total': [candidate['Total Score'] for candidate in candidates]
    }

    plt.figure(figsize=(12, 8))
    bar_width = 0.15
    index = np.arange(len(names))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (category, color) in enumerate(zip(categories, colors)):
        plt.bar(index + i*bar_width, scores[category], bar_width, 
                label=category, color=color)

    plt.plot(index + 1.5*bar_width, scores['Total'], 
             color='black', marker='o', linestyle='-', 
             linewidth=2, markersize=8, label='Total Score')

    plt.xticks(index + 1.5*bar_width, names, rotation=45, ha='right')
    plt.xlabel("Candidates")
    plt.ylabel("Scores")
    plt.title("Candidate Comparison Scores")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    
    return plt

def generate_feedback(ranked_candidates, job_description):
    """Generate feedback using Groq API"""
    feedback_reports = []
    
    for candidate in ranked_candidates:
        prompt = f"""
        Job Description: {job_description[:500]}... [truncated]
        
        Candidate Score: {candidate['Total Score']:.2f}/1.0
        - Skills Match: {candidate['Skill Score']:.2f}
        - Experience Match: {candidate['Experience Score']:.2f}
        - Education Match: {candidate['Education Score']:.2f}
        
        Generate structured feedback with:
        Candidate Name: {candidate['Name']}
        Analysis:
        1. THREE key strengths (bullet points)
        2. THREE main gaps (bullet points)
        3. TWO specific upskilling recommendations (bullet points)

        Format exactly like this example:
        
        Candidate Name:
        
        Analysis:
        ✓ Strengths:
        - Strength 1
        - Strength 2
        - Strength 3
        
        ✗ Missing Qualifications:
        - Gap 1
        - Gap 2
        - Gap 3
        
        ↗ Recommendations:
        - Recommendation 1
        - Recommendation 2
        """
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a strict career advisor. Use only the template provided."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        
        feedback_reports.append({
            "name": candidate["Name"],
            "score": candidate["Total Score"],
            "feedback": response.choices[0].message.content
        })
    
    return feedback_reports

# Streamlit UI
def main():
    # st.set_page_config(page_title="Resume Ranking System", layout="wide")
    st.title("📄 Resume Ranking System")
    st.markdown("Upload resumes in JSON format and a job description to rank candidates")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Input Parameters")
        uploaded_files = st.file_uploader("Upload Resume JSON Files", 
                                         type=["json"], 
                                         accept_multiple_files=True)
        job_description = st.text_area("Job Description", height=200,
                                      placeholder="Paste the job description here...")
        
        if st.button("Rank Candidates"):
            if not uploaded_files or not job_description:
                st.error("Please upload resume files and enter a job description")
                st.stop()
    
    # Main content area
    if uploaded_files and job_description:
        # Process uploaded files
        resumes = []
        for uploaded_file in uploaded_files:
            try:
                resume_data = json.load(uploaded_file)
                resumes.append(resume_data)
            except json.JSONDecodeError:
                st.error(f"Error decoding {uploaded_file.name}. Please ensure it's valid JSON.")
                st.stop()
        
        # Rank candidates
        with st.spinner("Ranking candidates..."):
            ranked_candidates = rank_candidates(resumes, job_description, model)
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["Score Cards", "Visualization", "Feedback Reports"])
        
        with tab1:
            st.subheader("Candidate Scores")
            for i, candidate in enumerate(ranked_candidates, 1):
                with st.expander(f"{i}. {candidate['Name']} (Score: {candidate['Total Score']:.2f})"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Skill Score", f"{candidate['Skill Score']:.2f}")
                    with col2:
                        st.metric("Experience Score", f"{candidate['Experience Score']:.2f}")
                    with col3:
                        st.metric("Education Score", f"{candidate['Education Score']:.2f}")
                    with col4:
                        st.metric("Keyword Match", f"{candidate['Keyword Match Score']:.2f}")
        
        with tab2:
            st.subheader("Score Visualization")
            plot = generate_plot(ranked_candidates)
            if plot:
                st.pyplot(plot)
                
                # Download button for the plot
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    plot.savefig(tmpfile.name, dpi=300, bbox_inches='tight')
                    with open(tmpfile.name, "rb") as f:
                        st.download_button(
                            label="Download Visualization",
                            data=f,
                            file_name="candidate_scores.png",
                            mime="image/png"
                        )
        
        with tab3:
            st.subheader("Candidate Feedback")
            if st.button("Generate Feedback Reports"):
                with st.spinner("Generating feedback..."):
                    feedback_reports = generate_feedback(ranked_candidates, job_description)
                
                for report in feedback_reports:
                    with st.expander(f"{report['name']} (Score: {report['score']:.2f})"):
                        st.markdown(report['feedback'])

if __name__ == "__main__":
    main()