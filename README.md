# Career Compass

Career Compass helps NUS students compare their curriculum and experience with the skills employers request in job descriptions.

## Features

- Load NUS modules using their module codes.
- Compare syllabus content with a target job description.
- View alignment scores, skill gaps, and covered requirements.
- Upload a resume in PDF, DOCX, or TXT format.
- Compare up to four future modules for a target career.

## Run locally

Clone the repository:

```bash
git clone https://github.com/k4b-04/career-compass.git
cd career-compass
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

The first run may take longer because the embedding model is downloaded from Hugging Face.

## Technology

Career Compass is built with Python and Streamlit. Streamlit provides the interactive web dashboard without requiring a separate frontend.

The app uses Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` model to understand the meaning of syllabus topics, job requirements, and resume evidence. It turns each text section into a numerical representation, then uses cosine similarity from scikit-learn to compare how closely two pieces of text are related.

The raw similarity scores are converted into a simpler 0–100 Career Compass Alignment Index. This makes the results easier to understand while keeping the underlying comparisons consistent.

NUSMods provides module names and descriptions through its API. Pandas displays the alignment tables, while `pypdf` and `python-docx` extract text from uploaded resumes. Streamlit caching prevents the language model and repeated NUSMods requests from being unnecessarily reloaded.
