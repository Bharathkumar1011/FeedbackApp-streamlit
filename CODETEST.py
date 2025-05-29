import streamlit as st # Streamlit for web app
import json
import re # Regular expressions for text processing
from sentence_transformers import SentenceTransformer, util 
import matplotlib.pyplot as plt
import numpy as np # For plotting
import os # For file handling
from groq import Groq # Importing Groq for model inference
from dotenv import load_dotenv # Load environment variables from .env file
null = None





# Job description for matching  
job_description = """Looking for candidate with 3+ years of experience.  
Must have strong skills in Python (Scikit-learn, TensorFlow, PyTorch), SQL (Spark/Hadoop/ETL), and cloud platforms (AWS/GCP/Azure).  
Experience in NLP, LLMs, GenAI, statistical analysis, and data visualization (Tableau/Power BI) is preferred.  
Remote or based in Mumbai, Bangalore, or Pune."""  


def normalize_text(data):
    """Improved text normalization that handles nested structures and prioritizes key fields."""
    if isinstance(data, str):
        return data
    elif isinstance(data, list):
        return " ".join(normalize_text(item) for item in data)
    elif isinstance(data, dict):
        # Prioritize description/role fields
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
    """
    Compute similarity between resume text and job description using a sentence-transformer model.
    """
    embeddings = model.encode([resume_text, job_description], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

# Compute skill match with improved handling of synonyms, partial matches, and mandatory skills
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
        # Check both full and partial matches
        if any(skill_word in job_desc_lower for skill_word in skill_lower.split()):
            skill_matches += 1
    
    # Penalty capped at 50% for missing mandatory skills
    missing_mandatory = max(0, len(mandatory_skills - {s.lower() for s in skills}))
    penalty = min(0.5, 0.1 * missing_mandatory)
    
    return max(0, (skill_matches / max(1, len(mandatory_skills))) - penalty)

# Main function to rank candidates based on resumes and job description
def rank_candidates(resumes, job_description, model):
    """Final ranking with all fixes applied"""
    scores = []
    for resume in resumes:
        # Extract data with case-insensitive handling
        skills = get_skills(resume)
        experience = get_experience(resume)
        education = resume.get("Education") or resume.get("education") or {}
        
        # Normalize text
        skills_text = normalize_text(skills)
        experience_text = normalize_text(experience)
        education_text = normalize_text(education)
        
        # Compute scores
        skill_score = clamp_score(match_resume_to_job(skills_text, job_description, model))
        experience_score = clamp_score(match_resume_to_job(experience_text, job_description, model))
        education_score = clamp_score(match_resume_to_job(education_text, job_description, model))
        keyword_score = clamp_score(compute_skill_match(skills, job_description))
        
        # Weighted scoring (adjusted weights)
        total_score = (
            0.4 * skill_score +
            0.4 * experience_score + 
            0.1 * education_score +
            0.1 * keyword_score
        )
        
        # Get candidate name
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


def plot_scores(ranked_candidates, top_n=10):
    """
    Visualize the scores of candidates using a grouped bar chart with better formatting.
    
    Args:
        ranked_candidates: List of candidate dictionaries with scores
        top_n: Number of top candidates to display (default: 10)
    """
    # Take only top N candidates
    candidates = ranked_candidates[:top_n]
    if not candidates:
        print("No candidates to plot")
        return

    # Prepare data
    names = [candidate['Name'] for candidate in candidates]
    categories = ['Skill', 'Experience', 'Education', 'Keyword Match']
    scores = {
        'Skill': [candidate['Skill Score'] for candidate in candidates],
        'Experience': [candidate['Experience Score'] for candidate in candidates],
        'Education': [candidate['Education Score'] for candidate in candidates],
        'Keyword Match': [candidate['Keyword Match Score'] for candidate in candidates],
        'Total': [candidate['Total Score'] for candidate in candidates]
    }

    # Plot setup
    plt.figure(figsize=(12, 8))
    bar_width = 0.15
    index = np.arange(len(names))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Create bars for each category
    for i, (category, color) in enumerate(zip(categories, colors)):
        plt.bar(index + i*bar_width, scores[category], bar_width, 
                label=category, color=color)

    # Add total scores as a line plot
    plt.plot(index + 1.5*bar_width, scores['Total'], 
             color='black', marker='o', linestyle='-', 
             linewidth=2, markersize=8, label='Total Score')

    # Configure plot details
    plt.xticks(index + 1.5*bar_width, names, rotation=45, ha='right')
    plt.xlabel("Candidates", fontsize=12)
    plt.ylabel("Scores", fontsize=12)
    plt.title("Candidate Comparison Scores", fontsize=14, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid and adjust layout
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.1)  # Leave room for legend
    plt.tight_layout()
    
    # Save plot instead of showing
    plot_path = "candidate_scores_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {os.path.abspath(plot_path)}")


# Load SentenceTransformer model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

#loding input data (.json file)
f1 = "Feedback Input/resume_data_2_pdf1.json"
f2 = "Feedback Input/resume_data_5_pdf2.json"
f3 = "Feedback Input/resume_data_3_pdf3.json"
f4 = "Feedback Input/resume_data_4_pdf4.json"
f5 = "Feedback Input/resume_data_1_pdf5.json"

# List of resume files
files = [f1, f2, f3, f4, f5]


# Load resumes from JSON file
docs = []
for file_path in files:
    with open(file_path, 'r') as f:  # Opens the file
        docs.append(json.load(f))    # Loads JSON content
# If the resume ranker expects a list of dictionaries, wrap the dictionary in a list
resume_data_list = docs

# Define resumes(INPUT_DATA)
resumes = resume_data_list

# Rank candidates
ranked_candidates = rank_candidates(resumes, job_description, model)

# Display ranked candidates
for candidate in ranked_candidates:
    print(f"Name: {candidate['Name']}, Total Score: {candidate['Total Score']:.2f}")


# Display ranked candidates with detailed scores
for i, candidate in enumerate(ranked_candidates, 1):
    print(f"\nCandidate #{i}: {candidate['Name']}")
    print("-" * 40)
    print(f"Total Score: {candidate['Total Score']:.2f}/1.00")
    print(f"• Skill Score: {candidate['Skill Score']:.2f}")
    print(f"• Experience Score: {candidate['Experience Score']:.2f}")
    print(f"• Education Score: {candidate['Education Score']:.2f}")
    print(f"• Keyword Match: {candidate['Keyword Match Score']:.2f}")


# Plot the scores
plot_scores(ranked_candidates)


# Load environment variables from .env file
load_dotenv()  

# Securely access the API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=groq_api_key)


#generate feedback for each candidate
def generate_feedback(ranked_candidates, job_description):
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
            temperature=0.5,  # Lower for more structured output
        )
        
        feedback_reports.append({
            "name": candidate["Name"],
            "score": candidate["Total Score"],
            "feedback": response.choices[0].message.content
        })
    
    return feedback_reports


# Generate feedback for ranked candidates
feedback = generate_feedback(ranked_candidates, job_description) 

# Display feedback for each candidate
print(feedback[0]['feedback'])

# Display feedback for each candidate
print(feedback)
