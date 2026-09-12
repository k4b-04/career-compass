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

- Python
- Streamlit
- Hugging Face Transformers / Sentence Transformers
- scikit-learn
- NUSMods API
