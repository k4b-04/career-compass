from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from html import escape
from io import BytesIO
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import streamlit as st


LOGGER = logging.getLogger("career_compass")


# Optional imports allow the app to show actionable dependency errors.
try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer

    DEPENDENCY_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - exercised only in misconfigured environments
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]
    cosine_similarity = None  # type: ignore[assignment]
    SentenceTransformer = None  # type: ignore[assignment,misc]
    DEPENDENCY_ERROR = str(exc)

try:
    from pypdf import PdfReader
except ImportError:  # Optional: only required when a PDF resume is uploaded.
    PdfReader = None  # type: ignore[assignment,misc]

try:
    from docx import Document
except ImportError:  # Optional: only required when a DOCX resume is uploaded.
    Document = None  # type: ignore[assignment,misc]


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUSMODS_API_BASE = "https://api.nusmods.com/v2"
NUSMODS_CACHE_TTL_SECONDS = 60 * 60 * 24
DEFAULT_ALIGNMENT_INDEX_THRESHOLD = 60.0
CALIBRATION_MIDPOINT = 0.32
CALIBRATION_SLOPE = 10.0
MIN_CHUNK_CHARACTERS = 3
MAX_DISPLAY_TEXT_LENGTH = 76
MAX_COMPARISON_MODULES = 4

DASHBOARD_TABLE_STYLES = [
    {
        "selector": "th",
        "props": [
            ("background-color", "#eef2ff"),
            ("color", "#1e293b"),
            ("font-weight", "700"),
            ("border", "1px solid #d9e2ef"),
            ("font-size", "0.82rem"),
        ],
    },
    {
        "selector": "td",
        "props": [
            ("border", "1px solid rgba(217, 226, 239, 0.72)"),
            ("font-weight", "600"),
            ("color", "#1e293b"),
        ],
    },
]

# Explicit skill aliases complement sentence-level semantic similarity.
SKILL_PATTERNS: dict[str, tuple[str, ...]] = {
    "python": (r"\bpython\b",),
    "pandas": (r"\bpandas\b",),
    "numpy": (r"\bnumpy\b",),
    "sql": (r"\bsql\b", r"structured query language"),
    "tableau": (r"\btableau\b",),
    "power bi": (r"\bpower\s*bi\b",),
    "aws": (r"\baws\b", r"amazon web services"),
    "cloud": (r"\bcloud\b",),
    "s3": (r"\bs3\b",),
    "athena": (r"\bathena\b",),
    "redshift": (r"\bredshift\b",),
    "docker": (r"\bdocker\b",),
    "git": (r"\bgit\b",),
    "dbt": (r"\bdbt\b",),
    "data cleaning": (r"data\s+clean(?:ing|ed)",),
    "data visualization": (r"data\s+visuali[sz](?:ation|e|ed|ing)",),
    "data analysis": (r"data\s+anal(?:ysis|ytics|yst)",),
    "statistics": (r"\bstatistic(?:s|al)?\b",),
    "hypothesis testing": (r"hypothesis\s+test(?:ing)?",),
    "a/b testing": (r"a\s*/\s*b\s+test(?:ing)?|a-b\s+test(?:ing)?|ab\s+test(?:ing)?",),
    "regression": (r"(?:linear\s+)?regression",),
    "forecasting": (r"\bforecast(?:ing)?\b",),
    "relational databases": (r"relational\s+(?:database|data\s+base)s?",),
    "academic writing": (r"academic\s+writing",),
    "communication": (r"\bcommunicat(?:e|es|ed|ing|ion)\b",),
}

JOB_CONTEXT_PATTERNS = (
    r"^(?:we|our company|the company)\b",
    r"^(?:key\s+)?(?:responsibilities|requirements|qualifications)\s*:?$",
    r"^(?:nice\s+to\s+have|preferred qualifications|about the role)\s*:?$",
    r"^job description\s*:?$",
)

RESUME_SECTION_NAMES = {
    "experience": "Experience",
    "work experience": "Experience",
    "professional experience": "Experience",
    "internship": "Internship",
    "internships": "Internship",
    "internship experience": "Internship",
    "projects": "Project",
    "project": "Project",
    "selected projects": "Project",
    "academic projects": "Project",
    "course projects": "Project",
    "research experience": "Research",
    "research projects": "Research",
}

RESUME_STOP_SECTION_NAMES = {
    "education",
    "skills",
    "technical skills",
    "certifications",
    "awards",
    "leadership",
    "activities",
    "volunteering",
    "interests",
    "summary",
    "profile",
    "professional certifications",
    "technical skills and professional certifications",
}

MAX_RESUME_BYTES = 5_000_000

RESUME_ACTION_PATTERN = re.compile(
    r"^(?:built|created|developed|designed|implemented|analyzed|automated|"
    r"deployed|led|managed|improved|reduced|increased|evaluated|researched|"
    r"used|applied|worked|collaborated|conducted|architected|engineered|"
    r"validated|integrated|leveraged|provided|utili[sz]ed|devised|constructed|"
    r"acted|coordinated|trained|optimized|refactored|launched|delivered|"
    r"achieved|mentored|streamlined|centralized|established|spearheaded)\b",
    re.IGNORECASE,
)

RESUME_DATE_PATTERN = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s+20\d{2}\b|\b20\d{2}\b",
    re.IGNORECASE,
)


def is_resume_content_line(line: str) -> bool:
    """Return whether a paragraph looks like entry evidence, not a title.

    Resume DOCX files often use ordinary paragraphs instead of actual list
    bullets. Sentence punctuation and action-led wording are useful signals
    across DOCX, PDF, and TXT uploads, while still allowing short metadata
    lines such as project subtitles to remain part of the entry label.
    """

    return bool(
        RESUME_ACTION_PATTERN.match(line)
        or re.search(r"[.!?]\s*$", line)
        or len(re.findall(r"[A-Za-z0-9]+", line)) >= 14
    )


def is_resume_entry_heading(line: str) -> bool:
    """Return whether a non-bullet paragraph likely starts a new entry."""

    words = re.findall(r"[A-Za-z0-9]+", line)
    if not words or len(words) > 32:
        return False

    # Date and role signals identify new resume entries.
    has_date = bool(RESUME_DATE_PATTERN.search(line))
    has_role_signal = bool(
        re.search(
            r"\b(?:intern|developer|analyst|engineer|manager|consultant|"
            r"researcher|assistant|lead|founder|coordinator|in[- ]charge|"
            r"committee|volunteer|officer)\b",
            line,
            re.IGNORECASE,
        )
    )
    return has_date and (has_role_signal or "|" in line or "\t" in line)


def raw_to_alignment_index(raw_score: float) -> float:
    """Map the raw hybrid score to a smoother, user-facing 0–100 index.

    This is a documented heuristic calibration, not a probability. The
    midpoint and slope can later be fitted from labelled curriculum/job pairs.
    """

    logit = CALIBRATION_SLOPE * (float(raw_score) - CALIBRATION_MIDPOINT)
    logit = max(-60.0, min(60.0, logit))
    return 100.0 / (1.0 + math.exp(-logit))


def alignment_index_to_raw(index: float) -> float:
    """Convert a 0–100 user threshold back to the raw score space."""

    probability = max(0.000001, min(0.999999, float(index) / 100.0))
    return CALIBRATION_MIDPOINT + (
        math.log(probability / (1.0 - probability)) / CALIBRATION_SLOPE
    )


def raw_scores_to_alignment_index(raw_scores: Any) -> Any:
    """Vectorized version of ``raw_to_alignment_index`` for the matrix view."""

    if np is None:
        return raw_scores

    logits = CALIBRATION_SLOPE * (
        np.asarray(raw_scores, dtype=float) - CALIBRATION_MIDPOINT
    )
    logits = np.clip(logits, -60.0, 60.0)
    return 100.0 / (1.0 + np.exp(-logits))


@dataclass(frozen=True)
class ResumeEvidence:
    """One internship, project, or related resume entry extracted for scoring."""

    category: str
    label: str
    text: str


DEFAULT_SYLLABUS = """Course: Applied Data Science and Research Communication

Module 1 — Python for Data Science
- Python syntax, functions, object-oriented programming, NumPy, and Pandas
- Reproducible notebooks, virtual environments, Git, and basic testing

Module 2 — Data Cleaning and Exploratory Analysis
- Data cleaning: missing values, duplicate records, outliers, and data types
- Exploratory data analysis with descriptive statistics and data visualization

Module 3 — SQL and Relational Data
- Relational database concepts, SELECT statements, joins, aggregations, and subqueries
- Query optimization, data validation, and translating business questions into SQL

Module 4 — Statistical Learning
- Probability, sampling, hypothesis testing, confidence intervals, and correlation
- Linear regression, regularization, model evaluation, and interpreting coefficients

Module 5 — Communicating Evidence
- Academic writing, literature review, citation practices, and research ethics
- Presenting analytical findings for technical and non-technical audiences

Module 6 — Capstone Project
- End-to-end analysis of a real dataset with a written report and oral presentation
"""


DEFAULT_JOB_DESCRIPTION = """Senior Data Analyst — Growth Analytics

We are looking for a Senior Data Analyst to turn product and customer data into clear decisions for cross-functional teams.

Key responsibilities and requirements:
- Write advanced SQL queries against large relational data sets and build reliable reporting tables.
- Use Python with Pandas for data cleaning, feature preparation, and repeatable analysis.
- Build executive-ready dashboards and exploratory reports in Tableau.
- Work with AWS Cloud services such as S3, Athena, and Redshift to access and transform data.
- Package reproducible analytics workflows with Docker and document how they are deployed.
- Design and analyze A/B testing experiments, including power, statistical significance, and business impact.
- Translate ambiguous business questions into measurable metrics and communicate recommendations to stakeholders.
- Apply regression, forecasting, and statistical modeling to explain changes in key performance indicators.
- Maintain high standards for data quality, documentation, and clear written communication.

Nice to have: experience with dbt, Git, and mentoring junior analysts.
"""


def default_nus_academic_year() -> str:
    """Infer the current NUS academic year using the August start of term."""

    now = datetime.now()
    start_year = now.year if now.month >= 8 else now.year - 1
    return f"{start_year}-{start_year + 1}"


def normalize_nus_module_code(raw_code: str) -> str:
    """Normalize codes such as ``cs 2040s`` and validate their shape."""

    module_code = re.sub(r"[\s-]+", "", raw_code or "").upper()
    if not re.fullmatch(r"[A-Z]{2,6}\d{3,5}[A-Z]*", module_code):
        raise ValueError(
            "Enter a NUS module code, for example CS1010A, CS2040S, BT1101, or MA1521."
        )
    return module_code


def parse_nus_module_codes(raw_codes: str) -> tuple[list[str], list[str]]:
    """Parse a comma-separated module list while preserving input order.

    Valid codes are de-duplicated before lookup. Invalid entries are returned
    separately so one typo does not prevent the remaining modules from loading.
    """

    entries = [
        entry.strip()
        for entry in (raw_codes or "").split(",")
        if entry.strip()
    ]
    if not entries:
        raise ValueError(
            "Enter at least one NUS module code, for example BT1101 or "
            "BT1101, CS1010A, CS2030."
        )

    normalized_codes: list[str] = []
    invalid_entries: list[str] = []
    seen_codes: set[str] = set()
    for entry in entries:
        try:
            normalized_code = normalize_nus_module_code(entry)
        except ValueError:
            invalid_entries.append(entry)
            continue

        if normalized_code not in seen_codes:
            normalized_codes.append(normalized_code)
            seen_codes.add(normalized_code)

    if not normalized_codes:
        raise ValueError(
            "No valid NUS module codes were found. Use entries such as "
            "BT1101, CS1010A, CS2030, or CS2040."
        )

    return normalized_codes, invalid_entries


def get_module_value(module: dict[str, Any], *keys: str) -> Any:
    """Read current camelCase fields and older PascalCase fields safely."""

    for key in keys:
        value = module.get(key)
        if value not in (None, ""):
            return value
    return None


@st.cache_data(ttl=NUSMODS_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_nusmods_module(
    academic_year: str,
    module_code: str,
) -> dict[str, Any]:
    """Fetch and cache a single NUSMods module record for 24 hours."""

    if not re.fullmatch(r"\d{4}-\d{4}", academic_year):
        raise ValueError("Academic year must use the format YYYY-YYYY.")

    endpoint = (
        f"{NUSMODS_API_BASE}/{quote(academic_year, safe='')}/modules/"
        f"{quote(module_code, safe='')}.json"
    )
    request = Request(
        endpoint,
        headers={
            "Accept": "application/json",
            "User-Agent": "CareerCompass-NUS-Student-Hackathon/1.0",
        },
    )

    try:
        with urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            raise ValueError(
                f"NUSMods could not find {module_code} for academic year {academic_year}."
            ) from exc
        raise RuntimeError(f"NUSMods returned an HTTP {exc.code} error.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            "Could not reach NUSMods. Check your internet connection and try again."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("NUSMods returned an invalid JSON response.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("NUSMods returned an unexpected response format.")
    return payload


def module_to_alignment_text(
    module: dict[str, Any],
    academic_year: str,
    requested_code: str,
) -> str:
    """Convert NUSMods module data into editable text for Career Compass."""

    module_code = get_module_value(module, "moduleCode", "ModuleCode") or requested_code
    title = get_module_value(module, "title", "ModuleTitle") or "Untitled module"
    description = get_module_value(module, "description", "ModuleDescription")

    if not description:
        raise ValueError(
            f"{module_code} was found, but NUSMods has no module description "
            f"for {academic_year}."
        )

    return (
        f"NUS Module: {module_code} — {title}\n"
        f"Academic Year: {academic_year}\n"
        f"Module Description: {description}"
    )


@dataclass(frozen=True)
class AlignmentResult:
    """All values needed to render one completed alignment audit."""

    syllabus_chunks: tuple[str, ...]
    job_chunks: tuple[str, ...]
    similarity_matrix: Any
    best_matches: tuple[float, ...]
    overall_score: float
    overall_alignment_index: float
    raw_threshold: float
    threshold_index: float
    resume_evidence: tuple[ResumeEvidence, ...] = ()
    resume_scores: tuple[float, ...] = ()
    resume_best_requirements: tuple[tuple[str, ...], ...] = ()

    @property
    def gap_indexes(self) -> tuple[int, ...]:
        """Return job requirements whose best syllabus match is below threshold."""

        return tuple(
            index
            for index, score in enumerate(self.best_matches)
            if score < self.raw_threshold
        )


def configure_page() -> None:
    """Configure the page and apply the Career Compass visual system."""

    st.set_page_config(
        page_title="Career Compass | NUS Career Alignment",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            :root {
                --dcma-ink: #0f172a;
                --dcma-ink-soft: #243452;
                --dcma-muted: #64748b;
                --dcma-border: #d9e2ef;
                --dcma-surface: rgba(255, 255, 255, 0.88);
                --dcma-page: #f4f7fb;
                --dcma-slate: #0b1220;
                --dcma-slate-soft: #131f35;
                --dcma-indigo: #4f46e5;
                --dcma-indigo-deep: #312e81;
                --dcma-cyan: #0891b2;
                --dcma-green: #047857;
                --dcma-red: #dc2626;
            }

            .stApp {
                background: var(--dcma-page);
            }
            [data-testid="stAppViewContainer"] .main {
                background:
                    radial-gradient(circle at 92% 0%, rgba(79, 70, 229, 0.10), transparent 30rem),
                    radial-gradient(circle at 5% 42%, rgba(8, 145, 178, 0.05), transparent 26rem),
                    var(--dcma-page);
            }
            [data-testid="stHeader"] {
                background: transparent;
            }
            .block-container {
                max-width: 1680px;
                padding: 2.35rem 3.2rem 4rem;
            }
            h1, h2, h3 {
                color: var(--dcma-ink);
                letter-spacing: -0.025em;
            }
            h2, h3 {
                margin-top: 0.2rem;
            }
            [data-testid="stCaptionContainer"] p {
                color: var(--dcma-muted);
            }

            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, #0a1222 0%, #0d172a 55%, #101d33 100%);
                border-right: 1px solid #243b61;
                box-shadow: 10px 0 30px rgba(15, 23, 42, 0.08);
            }
            [data-testid="stSidebar"] > div:first-child {
                padding: 1.35rem 1.1rem 1.5rem;
            }
            [data-testid="stSidebar"] * {
                color: #dbe7f5;
            }
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: #f8fbff;
            }
            [data-testid="stSidebar"] [data-baseweb="input"],
            [data-testid="stSidebar"] [data-baseweb="textarea"],
            [data-testid="stSidebar"] [data-baseweb="base-input"] {
                background: #ffffff !important;
                border: 1px solid #b7c9e2;
                border-radius: 11px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
            }
            [data-testid="stSidebar"] input,
            [data-testid="stSidebar"] textarea {
                background: #ffffff !important;
                color: #10213f !important;
                -webkit-text-fill-color: #10213f !important;
            }
            [data-testid="stSidebar"] input::placeholder,
            [data-testid="stSidebar"] textarea::placeholder {
                color: #64748b !important;
                -webkit-text-fill-color: #64748b !important;
            }
            [data-testid="stSidebar"] [data-testid="stForm"] {
                padding: 0.25rem 0 0;
            }
            [data-testid="stSidebar"] hr {
                border-color: #263d60;
            }
            [data-testid="stSidebar"] small,
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
                color: #9fb1c8;
            }
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] code {
                border: 1px solid #35537c;
                border-radius: 5px;
                padding: 0.12rem 0.3rem;
                background: #1a3152;
                color: #c9e5ff !important;
                -webkit-text-fill-color: #c9e5ff !important;
            }
            .sidebar-brand {
                display: flex;
                align-items: center;
                gap: 0.7rem;
                margin-bottom: 1.15rem;
            }
            .sidebar-brand-mark {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 2.2rem;
                height: 2.2rem;
                border: 1px solid #4b83df;
                border-radius: 11px;
                background: linear-gradient(135deg, var(--dcma-indigo), var(--dcma-cyan));
                color: #ffffff;
                font-size: 0.77rem;
                font-weight: 800;
                letter-spacing: 0.04em;
                box-shadow: 0 6px 16px rgba(79, 70, 229, 0.32);
            }
            .sidebar-brand-copy strong {
                display: block;
                color: #f8fbff;
                font-size: 0.92rem;
            }
            .sidebar-brand-copy span {
                display: block;
                margin-top: 0.1rem;
                color: #8fa5c0;
                font-size: 0.71rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
                border: 1px dashed #41638e;
                border-radius: 11px;
                background: #13223a;
            }
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
                color: #9fb1c8;
            }

            .dcma-hero {
                position: relative;
                overflow: hidden;
                border: 1px solid rgba(165, 180, 252, 0.34);
                border-radius: 24px;
                padding: 2.2rem 2.35rem 2rem;
                margin-bottom: 1.55rem;
                background:
                    radial-gradient(circle at 92% 8%, rgba(34, 211, 238, 0.24), transparent 18rem),
                    linear-gradient(120deg, #111936 0%, #26376d 56%, #075985 100%);
                box-shadow: 0 20px 45px rgba(30, 41, 92, 0.20);
                isolation: isolate;
            }
            .dcma-hero::after {
                position: absolute;
                width: 24rem;
                height: 24rem;
                right: -9rem;
                top: -12rem;
                border-radius: 50%;
                border: 1px solid rgba(255, 255, 255, 0.12);
                background: rgba(255, 255, 255, 0.04);
                content: "";
                z-index: -1;
            }
            .dcma-kicker {
                position: relative;
                z-index: 1;
                color: #a5b4fc;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                margin-bottom: 0.55rem;
            }
            .dcma-hero h1 {
                position: relative;
                z-index: 1;
                margin: 0;
                font-size: clamp(2rem, 3.5vw, 3.25rem);
                line-height: 1.06;
                color: #ffffff;
            }
            .dcma-subtitle {
                position: relative;
                z-index: 1;
                max-width: 900px;
                margin-top: 0.9rem;
                color: #d8e4f6;
                font-size: 1rem;
                line-height: 1.5;
            }
            .dcma-hero-tags {
                position: relative;
                z-index: 1;
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-top: 1.25rem;
            }
            .dcma-hero-tag {
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 999px;
                padding: 0.36rem 0.72rem;
                background: rgba(255, 255, 255, 0.10);
                color: #e5edff;
                font-size: 0.76rem;
                font-weight: 600;
                backdrop-filter: blur(8px);
            }

            [data-testid="stMetric"] {
                position: relative;
                min-height: 116px;
                border: 1px solid var(--dcma-border);
                border-radius: 18px;
                padding: 1.05rem 1.2rem;
                background: var(--dcma-surface);
                box-shadow: 0 12px 28px rgba(30, 41, 92, 0.07);
                backdrop-filter: blur(14px);
                transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
            }
            [data-testid="stMetric"]:hover {
                border-color: #bdc7f7;
                box-shadow: 0 16px 34px rgba(49, 46, 129, 0.12);
                transform: translateY(-2px);
            }
            [data-testid="stMetricLabel"] p {
                color: var(--dcma-muted);
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.01em;
            }
            [data-testid="stMetricValue"] {
                color: var(--dcma-ink);
                font-size: 1.95rem;
                font-variant-numeric: tabular-nums;
            }
            [data-testid="stDataFrame"] {
                overflow: hidden;
                border: 1px solid var(--dcma-border);
                border-radius: 16px;
                background: rgba(255, 255, 255, 0.82);
                box-shadow: 0 12px 28px rgba(30, 41, 92, 0.07);
            }
            [data-testid="stExpander"] {
                overflow: hidden;
                border: 1px solid var(--dcma-border);
                border-radius: 16px;
                background: rgba(255, 255, 255, 0.72);
                box-shadow: 0 8px 22px rgba(30, 41, 92, 0.045);
                transition: border-color 160ms ease, box-shadow 160ms ease;
            }
            [data-testid="stExpander"]:hover {
                border-color: #c5cff5;
                box-shadow: 0 12px 28px rgba(49, 46, 129, 0.08);
            }
            [data-testid="stExpander"] summary {
                color: var(--dcma-ink-soft);
                font-weight: 700;
            }
            [data-testid="stDataFrame"] iframe {
                border-radius: 15px;
            }
            [data-testid="stTabs"] {
                margin-top: -0.75rem;
            }
            [data-testid="stTabs"] [role="tablist"] {
                width: fit-content;
                max-width: 100%;
                margin: 0 auto 1.45rem;
                padding: 0;
                gap: 0.65rem;
                border: 0;
                background: transparent;
                box-shadow: none;
            }
            [data-testid="stTabs"] button[role="tab"] {
                flex: 0 0 auto;
                min-width: 12.5rem;
                min-height: 3rem;
                border-radius: 999px;
                border: 1px solid #d5deec !important;
                background: #e9eef7 !important;
                color: var(--dcma-ink-soft) !important;
                font-size: 0.96rem;
                font-weight: 750;
                line-height: 1.15;
                padding: 0.88rem 1.55rem;
                white-space: nowrap;
                transition: color 150ms ease, background 150ms ease, box-shadow 150ms ease,
                    border-color 150ms ease, transform 150ms ease;
            }
            [data-testid="stTabs"] button[role="tab"]:hover {
                border-color: #aebce0 !important;
                background: #f2f5fb !important;
                box-shadow: 0 5px 14px rgba(30, 41, 92, 0.10);
                transform: translateY(-1px);
            }
            [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
                border-color: #c8d2e4 !important;
                color: var(--dcma-indigo-deep) !important;
                background: #ffffff !important;
                box-shadow: 0 3px 10px rgba(30, 41, 92, 0.13);
            }
            [data-testid="stTabs"] button[role="tab"] > div,
            [data-testid="stTabs"] button[role="tab"] > div > p {
                color: inherit !important;
                white-space: nowrap;
            }
            [data-testid="stTabs"] button[role="tab"]::after,
            [data-testid="stTabs"] button[role="tab"] > div::after {
                background: transparent !important;
                height: 0 !important;
            }
            .comparison-intro {
                display: flex;
                align-items: flex-start;
                gap: 0.9rem;
                border: 1px solid #d8def7;
                border-radius: 16px;
                margin: 0.35rem 0 1.2rem;
                padding: 1rem 1.1rem;
                background: linear-gradient(120deg, #f8faff, #f0fdfa);
                color: var(--dcma-ink-soft);
                box-shadow: 0 10px 24px rgba(30, 41, 92, 0.055);
            }
            .comparison-intro-icon {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 2.1rem;
                height: 2.1rem;
                border-radius: 11px;
                background: linear-gradient(135deg, #e0e7ff, #cffafe);
                color: var(--dcma-indigo-deep);
                font-size: 1.05rem;
            }
            .comparison-intro strong {
                color: var(--dcma-ink);
            }
            .comparison-intro p {
                margin: 0.18rem 0 0;
                color: var(--dcma-muted);
                font-size: 0.87rem;
                line-height: 1.45;
            }
            .comparison-winner {
                border: 1px solid #a7e3c7;
                border-left: 5px solid #059669;
                border-radius: 16px;
                margin: 1.1rem 0;
                padding: 1rem 1.15rem;
                background: linear-gradient(120deg, #ecfdf5, #f7fffb);
                box-shadow: 0 12px 26px rgba(5, 150, 105, 0.09);
            }
            .comparison-winner-label {
                color: #047857;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .comparison-winner-title {
                margin-top: 0.18rem;
                color: var(--dcma-ink);
                font-size: 1.2rem;
                font-weight: 800;
            }
            .comparison-winner-score {
                margin-top: 0.25rem;
                color: #047857;
                font-size: 0.9rem;
                font-weight: 700;
            }
            .comparison-card {
                border: 1px solid var(--dcma-border);
                border-left: 4px solid #6366f1;
                border-radius: 14px;
                margin: 0.65rem 0;
                padding: 0.85rem 1rem;
                background: rgba(255, 255, 255, 0.84);
                box-shadow: 0 8px 20px rgba(30, 41, 92, 0.055);
            }
            .comparison-card-title {
                color: var(--dcma-ink);
                font-weight: 750;
            }
            .comparison-card-meta,
            .comparison-card-match {
                color: var(--dcma-muted);
                font-size: 0.82rem;
                line-height: 1.45;
            }
            .comparison-card-meta {
                margin-top: 0.18rem;
            }
            .comparison-card-match {
                margin-top: 0.45rem;
            }
            [data-testid="stAlert"] {
                border-radius: 13px;
                box-shadow: 0 7px 18px rgba(30, 41, 92, 0.05);
            }
            .audit-status {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                border: 1px solid #b8e3d0;
                border-radius: 14px;
                margin-bottom: 1.55rem;
                padding: 0.84rem 1.05rem;
                background: linear-gradient(90deg, #ecfdf5, #f4fffa);
                color: #05603a;
                font-size: 0.91rem;
                font-weight: 600;
                box-shadow: 0 8px 20px rgba(4, 120, 87, 0.06);
            }
            .audit-status-icon {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 1.4rem;
                height: 1.4rem;
                border-radius: 50%;
                background: #c9f3dc;
                color: #05603a;
                font-weight: 800;
            }
            .section-caption {
                margin: -0.55rem 0 1rem;
                color: var(--dcma-muted);
                font-size: 0.88rem;
            }

            .gap-card {
                position: relative;
                border: 1px solid #f6caca;
                border-left: 4px solid #ef4444;
                border-radius: 13px;
                margin: 0.62rem 0;
                padding: 0.9rem 1rem;
                background: linear-gradient(105deg, #fff4f4, #fffafa);
                color: var(--dcma-ink-soft);
                box-shadow: 0 7px 17px rgba(220, 38, 38, 0.05);
                transition: transform 150ms ease, box-shadow 150ms ease, border-color 150ms ease;
            }
            .gap-card:hover {
                border-color: #f09a9a;
                box-shadow: 0 12px 24px rgba(220, 38, 38, 0.09);
                transform: translateX(2px);
            }
            .nusmods-card {
                border: 1px solid #2c4b75;
                border-left: 3px solid #38bdf8;
                border-radius: 11px;
                margin: 0.16rem 0 0.28rem;
                padding: 0.54rem 0.65rem;
                background: linear-gradient(135deg, #13223a, #172c4c);
                box-shadow: 0 6px 14px rgba(0, 0, 0, 0.14);
                transition: transform 150ms ease, border-color 150ms ease, box-shadow 150ms ease;
            }
            .nusmods-card:hover {
                border-color: #4f8df7;
                box-shadow: 0 10px 20px rgba(5, 24, 55, 0.22);
                transform: translateY(-1px);
            }
            .nusmods-card-code {
                color: #8cc5ff;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.06em;
            }
            .nusmods-card-title {
                color: #f5f9ff;
                font-size: 0.82rem;
                font-weight: 700;
                margin-top: 0.15rem;
            }
            .covered-card {
                border: 1px solid #b4e3cf;
                border-left: 4px solid #10b981;
                border-radius: 13px;
                margin: 0.62rem 0;
                padding: 0.9rem 1rem;
                background: linear-gradient(105deg, #effcf5, #f8fffb);
                color: var(--dcma-ink-soft);
                box-shadow: 0 7px 17px rgba(5, 150, 105, 0.05);
                transition: transform 150ms ease, box-shadow 150ms ease, border-color 150ms ease;
            }
            .covered-card:hover {
                border-color: #71cda8;
                box-shadow: 0 12px 24px rgba(5, 150, 105, 0.09);
                transform: translateX(2px);
            }
            .empty-state {
                display: flex;
                align-items: flex-start;
                gap: 1rem;
                border: 1px solid #d8def7;
                border-radius: 18px;
                margin: 1rem 0 1.35rem;
                padding: 1.25rem 1.3rem;
                background: rgba(255, 255, 255, 0.84);
                box-shadow: 0 12px 28px rgba(30, 41, 92, 0.07);
                backdrop-filter: blur(14px);
                transition: transform 160ms ease, box-shadow 160ms ease;
            }
            .empty-state:hover {
                box-shadow: 0 16px 34px rgba(49, 46, 129, 0.11);
                transform: translateY(-2px);
            }
            .empty-state-icon {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 2.35rem;
                height: 2.35rem;
                border-radius: 12px;
                background: linear-gradient(135deg, #e0e7ff, #cffafe);
                color: var(--dcma-indigo-deep);
                font-size: 1.2rem;
                box-shadow: inset 0 0 0 1px rgba(79, 70, 229, 0.08);
            }
            .empty-state h3 {
                margin: 0;
                font-size: 1.05rem;
            }
            .empty-state p {
                margin: 0.25rem 0 0;
                color: var(--dcma-muted);
            }
            .method-card {
                min-height: 112px;
                border: 1px solid rgba(217, 226, 239, 0.96);
                border-radius: 16px;
                padding: 1.05rem 1.08rem;
                background: rgba(255, 255, 255, 0.76);
                box-shadow: 0 8px 20px rgba(30, 41, 92, 0.04);
                transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
            }
            .method-card:hover {
                border-color: #b8c1f2;
                box-shadow: 0 14px 28px rgba(49, 46, 129, 0.09);
                transform: translateY(-3px);
            }
            .method-card strong {
                color: var(--dcma-ink);
            }
            .method-card p {
                margin: 0.35rem 0 0;
                color: var(--dcma-muted);
                font-size: 0.84rem;
                line-height: 1.45;
            }
            .stButton > button,
            .stFormSubmitButton button {
                border-radius: 10px;
                font-weight: 650;
                transition: border-color 150ms ease, box-shadow 150ms ease, transform 150ms ease;
            }
            .stButton > button:hover,
            .stFormSubmitButton button:hover {
                border-color: #7aa9ed;
                box-shadow: 0 5px 14px rgba(37, 99, 235, 0.16);
                transform: translateY(-1px);
            }
            [data-testid="stSidebar"] .stButton > button[kind="primary"],
            [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"],
            [data-testid="stSidebar"] .stFormSubmitButton button[data-testid="stBaseButton-primaryFormSubmit"] {
                border-color: #4f8df7;
                background: linear-gradient(135deg, var(--dcma-indigo), var(--dcma-cyan));
            }
            [data-testid="stSidebar"] .stButton > button[kind="primary"],
            [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"],
            [data-testid="stSidebar"] .stButton > button[kind="primary"] p,
            [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] p,
            [data-testid="stSidebar"] .stFormSubmitButton button[data-testid="stBaseButton-primaryFormSubmit"],
            [data-testid="stSidebar"] .stFormSubmitButton button[data-testid="stBaseButton-primaryFormSubmit"] * {
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
            }
            [data-testid="stSidebar"] .stButton > button[kind="secondary"],
            [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"] {
                border-color: #41638e;
                background: #ffffff !important;
                color: #17345f !important;
                -webkit-text-fill-color: #17345f !important;
            }
            [data-testid="stSidebar"] .stButton > button[kind="secondary"] *,
            [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"] *,
            [data-testid="stSidebar"] .stButton > button[kind="secondary"] p,
            [data-testid="stSidebar"] .stButton > button[kind="secondary"] span,
            [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"] p,
            [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"] span {
                color: #0b3b78 !important;
                -webkit-text-fill-color: #0b3b78 !important;
            }
            [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-secondaryFormSubmit"],
            [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-secondaryFormSubmit"] * {
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
                opacity: 1 !important;
            }
            [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-secondaryFormSubmit"] {
                border-color: #5b9cf6 !important;
                background: linear-gradient(135deg, #1d4ed8, #075985) !important;
                color: #ffffff !important;
            }
            [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
                border-color: #93c5fd !important;
                background: linear-gradient(135deg, #2563eb, #0e7490) !important;
            }
            .stFormSubmitButton button[data-testid="stBaseButton-secondaryFormSubmit"] {
                border: 1px solid #5b9cf6 !important;
                background: #1d4ed8 !important;
                background-color: #1d4ed8 !important;
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
                opacity: 1 !important;
            }
            .stFormSubmitButton button[data-testid="stBaseButton-secondaryFormSubmit"] * {
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
                opacity: 1 !important;
            }
            .stFormSubmitButton button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
                background: #2563eb !important;
                background-color: #2563eb !important;
            }
            @media (max-width: 900px) {
                .block-container {
                    padding: 1.5rem 1rem 3rem;
                }
                .dcma-hero {
                    padding: 1.35rem 1.2rem;
                }
                [data-testid="stTabs"] [role="tablist"] {
                    width: 100%;
                    gap: 0.45rem;
                }
                [data-testid="stTabs"] button[role="tab"] {
                    min-width: 0;
                    flex: 1;
                    padding-inline: 0.55rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def preprocess_text(text: str) -> list[str]:
    """Split syllabus or job text into meaningful sentence/bullet chunks.

    Newline-delimited bullets are preserved as individual requirements. Plain
    prose is then split on sentence boundaries. Short fragments made only of
    punctuation or stop words are excluded, while compact technical terms
    such as ``SQL`` and ``AWS`` are intentionally retained.
    """

    if not isinstance(text, str):
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    candidates: list[str] = []

    for line in normalized.split("\n"):
        clean_line = re.sub(r"^[\s\-*•▪‣◦]+", "", line).strip()
        if not clean_line:
            continue

        # Split prose only at likely sentence boundaries.
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean_line)
        candidates.extend(piece.strip() for piece in pieces if piece.strip())

    stop_words = {
        "a",
        "an",
        "and",
        "for",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
    }
    chunks: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        cleaned = re.sub(r"\s+", " ", candidate).strip(" \t-–—")
        if len(cleaned) < MIN_CHUNK_CHARACTERS:
            continue

        words = re.findall(r"[A-Za-z0-9]+", cleaned.lower())
        if not words or all(word in stop_words for word in words):
            continue

        dedupe_key = cleaned.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        chunks.append(cleaned)

    return chunks


def preprocess_job_requirements(text: str) -> list[str]:
    """Extract actual job requirements instead of titles and boilerplate.

    When a job posting contains bullets, only bullet lines are treated as
    requirements. This removes role titles, introductions, and section labels
    such as ``Key responsibilities and requirements:`` from the gap report.
    For prose-only postings, sentence splitting is used with a small set of
    generic context filters.
    """

    if not isinstance(text, str):
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    bullet_candidates: list[str] = []
    prose_candidates: list[str] = []
    bullet_pattern = re.compile(r"^\s*(?:[-*•▪‣◦]|\d+[.)])\s*")

    for line in normalized.split("\n"):
        if not line.strip():
            continue

        bullet_match = bullet_pattern.match(line)
        clean_line = bullet_pattern.sub("", line).strip()
        if not clean_line:
            continue

        if bullet_match:
            # Keep bullets whole so tools stay paired with their actions.
            bullet_candidates.append(clean_line)
        else:
            pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean_line)
            prose_candidates.extend(piece.strip() for piece in pieces if piece.strip())

    candidates = bullet_candidates if bullet_candidates else prose_candidates
    stop_words = {
        "a",
        "an",
        "and",
        "for",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
    }
    chunks: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        cleaned = re.sub(r"\s+", " ", candidate).strip(" \t-–—")
        if len(cleaned) < MIN_CHUNK_CHARACTERS:
            continue

        if not bullet_candidates:
            lower_cleaned = cleaned.casefold()
            if any(re.search(pattern, lower_cleaned) for pattern in JOB_CONTEXT_PATTERNS):
                continue

            # Exclude short role titles that are not requirements.
            if (
                len(cleaned.split()) <= 10
                and re.search(r"\b(?:analyst|engineer|scientist|manager|developer)\b", lower_cleaned)
                and not re.search(
                    r"\b(?:write|use|build|work|design|apply|maintain|develop|analy[sz]e|experience)\b",
                    lower_cleaned,
                )
            ):
                continue

        words = re.findall(r"[A-Za-z0-9]+", cleaned.lower())
        if not words or all(word in stop_words for word in words):
            continue

        dedupe_key = cleaned.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        chunks.append(cleaned)

    return chunks


def identify_resume_section(line: str) -> str | None:
    """Identify supported resume section headings."""

    normalized = re.sub(r"[^a-z ]", "", line.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if normalized in RESUME_SECTION_NAMES:
        return RESUME_SECTION_NAMES[normalized]
    if normalized in RESUME_STOP_SECTION_NAMES:
        return ""
    return None


def extract_resume_evidence(text: str) -> list[ResumeEvidence]:
    """Extract internship, experience, project, and research entries.

    Resume layouts vary considerably, so this parser uses section headings and
    bullet/action lines rather than relying on one rigid document template.
    Each entry is kept as one evidence item for scoring.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    bullet_pattern = re.compile(r"^\s*(?:[-*•▪‣◦]|\d+[.)])\s*")
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.replace("\r", "\n").split("\n")
        if line.strip()
    ]
    evidence: list[ResumeEvidence] = []
    current_category: str | None = None
    current_label: str | None = None
    current_content: list[str] = []

    def flush_entry() -> None:
        nonlocal current_label, current_content
        if current_category and current_content:
            label = current_label or f"{current_category} item {len(evidence) + 1}"
            evidence.append(
                ResumeEvidence(
                    category=current_category,
                    label=label,
                    text=f"{label}: {' '.join(current_content)}",
                )
            )
        current_label = None
        current_content = []

    for raw_line in lines:
        section = identify_resume_section(raw_line)
        if section is not None:
            flush_entry()
            current_category = section or None
            continue

        if current_category is None:
            continue

        is_bullet = bool(bullet_pattern.match(raw_line))
        clean_line = bullet_pattern.sub("", raw_line).strip()
        if not clean_line:
            continue

        starts_new_entry = is_resume_entry_heading(clean_line)
        if is_bullet or (not starts_new_entry and is_resume_content_line(clean_line)):
            if current_label is None:
                current_label = f"{current_category} item {len(evidence) + 1}"
            current_content.append(clean_line)
            continue

        # A dated role/project heading starts the next evidence item.
        if current_content and starts_new_entry:
            flush_entry()
            current_label = clean_line
        elif current_label is None:
            current_label = clean_line
        elif current_content:
            current_content.append(clean_line)
        else:
            current_label = f"{current_label} | {clean_line}"

    flush_entry()
    return evidence


def extract_resume_text(uploaded_file: Any) -> str:
    """Extract text from an uploaded TXT, PDF, or DOCX resume."""

    if uploaded_file is None:
        return ""

    file_name = str(getattr(uploaded_file, "name", "resume")).lower()
    try:
        file_bytes = uploaded_file.getvalue()
    except Exception as exc:
        raise RuntimeError(
            "The uploaded resume could not be read. Please upload the file again."
        ) from exc

    if not isinstance(file_bytes, (bytes, bytearray)):
        raise RuntimeError("The uploaded resume has an unsupported file representation.")
    if not file_bytes:
        raise ValueError("The uploaded resume is empty.")
    if len(file_bytes) > MAX_RESUME_BYTES:
        raise ValueError("Resume files must be 5 MB or smaller.")

    if file_name.endswith(".txt"):
        extracted = file_bytes.decode("utf-8-sig", errors="replace")
    elif file_name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError(
                "PDF resume support requires pypdf. Install it with: pip install pypdf"
            )
        try:
            reader = PdfReader(BytesIO(file_bytes))
            extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise RuntimeError(
                "The PDF resume could not be read. Try exporting it as a text-based PDF."
            ) from exc
    elif file_name.endswith(".docx"):
        if Document is None:
            raise RuntimeError(
                "DOCX resume support requires python-docx. "
                "Install it with: pip install python-docx"
            )
        try:
            document = Document(BytesIO(file_bytes))
            extracted = "\n".join(
                paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
            )
        except Exception as exc:
            raise RuntimeError("The DOCX resume could not be read.") from exc
    else:
        raise ValueError("Upload a TXT, PDF, or DOCX resume.")

    normalized = "\n".join(
        re.sub(r"\s+", " ", line).strip()
        for line in extracted.splitlines()
        if line.strip()
    )
    if not normalized:
        raise ValueError("The uploaded resume did not contain extractable text.")
    return normalized


def extract_skill_terms(text: str) -> set[str]:
    """Return canonical skills explicitly mentioned in a text chunk."""

    normalized = text.casefold()
    return {
        skill
        for skill, patterns in SKILL_PATTERNS.items()
        if any(re.search(pattern, normalized) for pattern in patterns)
    }


def explicit_skill_matrix(
    job_chunks: Sequence[str],
    syllabus_chunks: Sequence[str],
) -> Any:
    """Score explicit job-skill coverage for every job/module pair."""

    if np is None:
        return None

    job_skills = [extract_skill_terms(chunk) for chunk in job_chunks]
    syllabus_skills = [extract_skill_terms(chunk) for chunk in syllabus_chunks]
    matrix = np.zeros((len(job_chunks), len(syllabus_chunks)), dtype=float)

    for job_index, required_skills in enumerate(job_skills):
        if not required_skills:
            continue
        for module_index, module_skills in enumerate(syllabus_skills):
            matrix[job_index, module_index] = len(
                required_skills.intersection(module_skills)
            ) / len(required_skills)

    return matrix


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> Any:
    """Load and cache the embedding model for the lifetime of the app process."""

    if SentenceTransformer is None:
        raise RuntimeError(
            "The sentence-transformers package is unavailable. Install the app dependencies "
            "with: pip install streamlit sentence-transformers scikit-learn pandas numpy"
        )

    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as exc:  # pragma: no cover - depends on local model/network state
        raise RuntimeError(
            f"Unable to load '{MODEL_NAME}'. Confirm that the machine can access Hugging Face "
            "the first time the app runs, then try again."
        ) from exc


def encode_chunks(model: Any, chunks: Sequence[str]) -> Any:
    """Create normalized sentence embeddings in a memory-conscious batch."""

    if not chunks:
        raise ValueError("At least one meaningful text chunk is required.")

    try:
        return model.encode(
            list(chunks),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "The embedding model could not encode the supplied text. "
            "Try shorter input or restart the Streamlit app."
        ) from exc


def run_alignment(
    syllabus_text: str,
    job_text: str,
    threshold_index: float,
    syllabus_chunks_override: Sequence[str] | None = None,
    resume_text: str | None = None,
) -> AlignmentResult:
    """Preprocess, embed, compare, and summarize one alignment audit.

    Imported NUSMods modules can be passed as overrides so each module remains
    one atomic syllabus item even when its description contains paragraphs.
    """

    module_syllabus_chunks = [
        chunk
        for chunk in (syllabus_chunks_override or ())
        if isinstance(chunk, str) and chunk.strip()
    ]
    manual_syllabus_chunks = preprocess_text(syllabus_text)
    syllabus_chunks = module_syllabus_chunks + manual_syllabus_chunks
    job_chunks = preprocess_job_requirements(job_text)

    if not syllabus_chunks:
        raise ValueError(
            "No meaningful syllabus modules were found. Add sentences or bullet-point topics."
        )
    if not job_chunks:
        raise ValueError(
            "No meaningful job requirements were found. Add sentences or bullet-point requirements."
        )
    if not 0.0 <= threshold_index <= 100.0:
        raise ValueError("Minimum alignment index must be between 0 and 100.")
    if cosine_similarity is None or np is None:
        raise RuntimeError(
            "The analytics dependencies are unavailable. Install scikit-learn and numpy."
        )

    raw_threshold = alignment_index_to_raw(threshold_index)

    model = load_embedding_model()
    job_embeddings = encode_chunks(model, job_chunks)

    # Compare each visible module by its strongest internal sentence.
    embedding_chunks: list[str] = []
    embedding_groups: list[list[int]] = []
    for module_chunk in module_syllabus_chunks:
        internal_chunks = preprocess_text(module_chunk) or [module_chunk]
        group_indexes = list(
            range(len(embedding_chunks), len(embedding_chunks) + len(internal_chunks))
        )
        embedding_chunks.extend(internal_chunks)
        embedding_groups.append(group_indexes)

    for manual_chunk in manual_syllabus_chunks:
        embedding_groups.append([len(embedding_chunks)])
        embedding_chunks.append(manual_chunk)

    syllabus_embeddings = encode_chunks(model, embedding_chunks)
    sentence_similarity = cosine_similarity(job_embeddings, syllabus_embeddings)
    sentence_similarity = np.clip(
        np.asarray(sentence_similarity, dtype=float),
        0.0,
        1.0,
    )

    # Aggregate sentence scores back to visible module columns.
    semantic_matrix = np.column_stack(
        [np.max(sentence_similarity[:, indexes], axis=1) for indexes in embedding_groups]
    )
    skill_matrix = explicit_skill_matrix(job_chunks, syllabus_chunks)

    # Explicit skill evidence complements, but does not dominate, semantics.
    matrix = np.clip(0.75 * semantic_matrix + 0.25 * skill_matrix, 0.0, 1.0)
    best_matches = np.max(matrix, axis=1)
    overall_score = float(np.mean(best_matches))
    overall_alignment_index = raw_to_alignment_index(overall_score)

    resume_evidence = tuple(extract_resume_evidence(resume_text or ""))
    resume_scores: tuple[float, ...] = ()
    resume_best_requirements: tuple[tuple[str, ...], ...] = ()

    if resume_evidence:
        resume_texts = [evidence.text for evidence in resume_evidence]
        resume_embeddings = encode_chunks(model, resume_texts)
        resume_semantic = cosine_similarity(job_embeddings, resume_embeddings)
        resume_semantic = np.clip(
            np.asarray(resume_semantic, dtype=float),
            0.0,
            1.0,
        )
        resume_skill = explicit_skill_matrix(job_chunks, resume_texts)
        resume_matrix = np.clip(
            0.75 * resume_semantic + 0.25 * resume_skill,
            0.0,
            1.0,
        )

        # Average the top three requirement matches to limit generic matches.
        top_k = min(3, len(job_chunks))
        resume_scores = tuple(
            float(score)
            for score in np.mean(
                np.sort(resume_matrix, axis=0)[-top_k:, :],
                axis=0,
            )
        )
        resume_best_requirements = tuple(
            tuple(
                job_chunks[int(job_index)]
                for job_index in np.argsort(resume_matrix[:, evidence_index])[::-1][
                    : min(2, len(job_chunks))
                ]
            )
            for evidence_index in range(len(resume_evidence))
        )

    return AlignmentResult(
        syllabus_chunks=tuple(syllabus_chunks),
        job_chunks=tuple(job_chunks),
        similarity_matrix=matrix,
        best_matches=tuple(float(score) for score in best_matches),
        overall_score=overall_score,
        overall_alignment_index=overall_alignment_index,
        raw_threshold=raw_threshold,
        threshold_index=float(threshold_index),
        resume_evidence=resume_evidence,
        resume_scores=resume_scores,
        resume_best_requirements=resume_best_requirements,
    )


def shorten(text: str, max_length: int = MAX_DISPLAY_TEXT_LENGTH) -> str:
    """Shorten table labels without changing the underlying gap text."""

    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 1].rstrip()}…"


def style_dashboard_dataframe(
    frame: Any,
    format_spec: str | dict[str, str] | None = None,
    gradient_subset: Sequence[str] | None = None,
    apply_gradient: bool = True,
) -> Any:
    """Apply consistent, high-contrast presentation styling to a dataframe.

    This helper only affects rendering. It deliberately does not mutate the
    input frame or alter any values used by the alignment calculations.
    """

    styler = frame.style
    if format_spec is not None:
        styler = styler.format(format_spec)

    if apply_gradient:
        gradient_kwargs: dict[str, Any] = {
            "cmap": "YlGnBu",
            "vmin": 0.0,
            "vmax": 100.0,
            "text_color_threshold": 0.45,
        }
        if gradient_subset is not None:
            gradient_kwargs["subset"] = list(gradient_subset)
        styler = styler.background_gradient(**gradient_kwargs)

    return styler.set_table_styles(DASHBOARD_TABLE_STYLES)


def render_matrix(result: AlignmentResult) -> None:
    """Render the job-by-syllabus hybrid alignment matrix as a heatmap table."""

    if pd is None:
        st.error("Pandas is unavailable, so the alignment matrix cannot be rendered.")
        return

    requirement_labels = [
        f"R{index + 1}: {shorten(requirement)}"
        for index, requirement in enumerate(result.job_chunks)
    ]
    module_labels = [
        f"M{index + 1}: {shorten(module, 50)}"
        for index, module in enumerate(result.syllabus_chunks)
    ]
    display_matrix = raw_scores_to_alignment_index(result.similarity_matrix)
    matrix_frame = pd.DataFrame(
        display_matrix,
        index=requirement_labels,
        columns=module_labels,
    )

    styled_frame = style_dashboard_dataframe(
        matrix_frame,
        format_spec="{:.0f}",
    )
    st.dataframe(
        styled_frame,
        use_container_width=True,
        height=min(720, 160 + 42 * len(result.job_chunks)),
    )

    with st.expander("View full requirement and module labels"):
        label_frame = pd.DataFrame(
            {
                "Requirement": list(result.job_chunks),
                "Best syllabus match": [
                    result.syllabus_chunks[int(np.argmax(row))]
                    for row in result.similarity_matrix
                ],
                "Career Compass index": [
                    f"{raw_to_alignment_index(score):.0f}/100"
                    for score in result.best_matches
                ],
                "Raw model score": [f"{score:.3f}" for score in result.best_matches],
            }
        )
        st.dataframe(
            style_dashboard_dataframe(label_frame, apply_gradient=False),
            use_container_width=True,
            hide_index=True,
        )


def render_met_requirements(result: AlignmentResult) -> None:
    """Show requirements whose best curriculum match clears the threshold."""

    met_indexes = tuple(
        index
        for index, score in enumerate(result.best_matches)
        if score >= result.raw_threshold
    )
    st.subheader("Market Requirements Already Met")
    st.caption(
        f"These requirements meet or exceed the {result.threshold_index:.0f}/100 "
        "minimum alignment index."
    )

    if not met_indexes:
        st.info(
            "No requirements currently clear the selected threshold. Lower the threshold or "
            "add more relevant modules, projects, or experience to see covered requirements."
        )
        return

    st.success(f"{len(met_indexes)} job requirement(s) currently meet the threshold.")
    for index in met_indexes:
        requirement = escape(result.job_chunks[index])
        raw_score = result.best_matches[index]
        alignment_index = raw_to_alignment_index(raw_score)
        module_index = int(np.argmax(result.similarity_matrix[index]))
        module_label = result.syllabus_chunks[module_index].splitlines()[0]
        st.markdown(
            f'<div class="covered-card"><strong>{requirement}</strong><br>'
            f'<small>Career Compass alignment index: {alignment_index:.0f}/100 '
            f'· matched by: {escape(shorten(module_label, 110))}</small></div>',
            unsafe_allow_html=True,
        )


def render_gaps(result: AlignmentResult) -> None:
    """Render exact requirement strings that fall below the selected threshold."""

    st.subheader("Critical Skill Gaps to Bridge via Micro-Credentials")
    st.caption(
        f"Requirements below {result.threshold_index:.0f}/100 are flagged for targeted "
        f"learning or applied projects (raw model cutoff: {result.raw_threshold:.3f})."
    )

    if not result.gap_indexes:
        st.success(
            "No critical gaps were detected at this threshold. The supplied syllabus covers "
            "all identified job requirements at or above the selected similarity level."
        )
        return

    st.warning(
        f"{len(result.gap_indexes)} of {len(result.job_chunks)} job requirements are below "
        f"the {result.threshold_index:.0f}/100 threshold."
    )
    for index in result.gap_indexes:
        requirement = result.job_chunks[index]
        best_score = result.best_matches[index]
        safe_requirement = escape(requirement)
        alignment_index = raw_to_alignment_index(best_score)
        st.markdown(
            f'<div class="gap-card"><strong>{safe_requirement}</strong><br>'
            f'<small>Career Compass alignment index: {alignment_index:.0f}/100 '
            f'· raw model score: {best_score:.3f}</small></div>',
            unsafe_allow_html=True,
        )


def render_coverage_summary(result: AlignmentResult) -> None:
    """Show a compact count of covered versus flagged requirements."""

    gap_count = len(result.gap_indexes)
    covered_count = len(result.job_chunks) - gap_count
    summary = pd.DataFrame(
        {
            "Status": ["Covered / at threshold", "Below threshold"],
            "Requirements": [covered_count, gap_count],
        }
    )
    st.dataframe(
        style_dashboard_dataframe(summary, apply_gradient=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Requirements": st.column_config.NumberColumn(format="%d"),
        },
        )


def render_resume_alignment(result: AlignmentResult) -> None:
    """Render alignment scores for extracted resume evidence."""

    st.divider()
    st.subheader("Resume Evidence Alignment")
    st.caption(
        "Internships, projects, and research entries are scored against the target job. "
        "Each evidence score is the average of its three strongest requirement matches."
    )

    if not result.resume_evidence:
        st.warning(
            "No internship, project, or research entries were detected. Use clear resume "
            "section headings such as 'Experience' or 'Projects' and try again."
        )
        return

    if pd is None:
        st.error("Pandas is unavailable, so resume alignment cannot be rendered.")
        return

    rows: list[dict[str, Any]] = []
    for evidence, raw_score, matched_requirements in zip(
        result.resume_evidence,
        result.resume_scores,
        result.resume_best_requirements,
    ):
        rows.append(
            {
                "Type": evidence.category,
                "Entry": evidence.label,
                "Evidence": shorten(evidence.text, 280),
                "Alignment Index": raw_to_alignment_index(raw_score),
                "Raw model score": raw_score,
                "Top matching job requirements": "; ".join(
                    shorten(requirement, 105) for requirement in matched_requirements
                ),
            }
        )

    resume_frame = pd.DataFrame(rows)
    styled_frame = style_dashboard_dataframe(
        resume_frame,
        format_spec={
            "Alignment Index": "{:.0f}",
            "Raw model score": "{:.3f}",
        },
        gradient_subset=["Alignment Index"],
    )
    st.dataframe(
        styled_frame,
        use_container_width=True,
        hide_index=True,
        height=min(540, 150 + 44 * len(rows)),
    )


def render_nusmods_lookup() -> None:
    """Render NUSMods lookup controls and add fetched modules to session state."""

    st.subheader("NUSMods Module Lookup")
    st.caption(
        "Enter one or more comma-separated codes to add their NUSMods descriptions "
        "as separate curriculum modules."
    )

    st.session_state.setdefault("nusmods_academic_year", default_nus_academic_year())
    st.session_state.setdefault("nusmods_module_code", "")
    st.session_state.setdefault("nusmods_modules", {})

    # Migrate older sessions without duplicating imported modules as manual text.
    if "manual_syllabus_input" not in st.session_state:
        previous_syllabus = st.session_state.get("syllabus_input", "")
        if (
            isinstance(previous_syllabus, str)
            and previous_syllabus.strip()
            and not previous_syllabus.lstrip().startswith("NUS Module:")
        ):
            st.session_state["manual_syllabus_input"] = previous_syllabus
        else:
            st.session_state["manual_syllabus_input"] = DEFAULT_SYLLABUS

    with st.form("nusmods_lookup_form", clear_on_submit=False):
        academic_year = st.text_input(
            "Academic year",
            key="nusmods_academic_year",
            max_chars=9,
        )
        module_codes = st.text_input(
            "NUS Module Code(s)",
            placeholder="e.g. BT1101, CS1010A, CS2030, CS2040",
            key="nusmods_module_code",
            max_chars=300,
        )
        lookup_clicked = st.form_submit_button(
            "Load from NUSMods",
            use_container_width=True,
        )

    if lookup_clicked:
        try:
            normalized_codes, invalid_entries = parse_nus_module_codes(module_codes)
            academic_year_value = academic_year.strip()
            loaded_codes: list[str] = []
            failed_lookups: list[str] = []
            was_empty_before_batch = not st.session_state["nusmods_modules"]

            for normalized_code in normalized_codes:
                try:
                    module = fetch_nusmods_module(
                        academic_year_value,
                        normalized_code,
                    )
                    imported_text = module_to_alignment_text(
                        module,
                        academic_year_value,
                        normalized_code,
                    )

                    # Remove the demo syllabus only on the first successful import.
                    if (
                        was_empty_before_batch
                        and not loaded_codes
                        and st.session_state.get("manual_syllabus_input", "").strip()
                        == DEFAULT_SYLLABUS.strip()
                    ):
                        st.session_state["manual_syllabus_input"] = ""

                    title = get_module_value(module, "title", "ModuleTitle") or "Untitled module"
                    module_record = {
                        "academic_year": academic_year_value,
                        "module_code": normalized_code,
                        "module": module,
                        "alignment_text": imported_text,
                        "title": str(title),
                    }
                    # Insert at the front so the newest batch entry appears first.
                    existing_modules = st.session_state["nusmods_modules"]
                    existing_modules.pop(normalized_code, None)
                    st.session_state["nusmods_modules"] = {
                        normalized_code: module_record,
                        **existing_modules,
                    }
                    loaded_codes.append(normalized_code)
                except (ValueError, RuntimeError) as exc:
                    failed_lookups.append(f"{normalized_code}: {exc}")
                except Exception:
                    LOGGER.exception("Unexpected NUSMods lookup failure for %s", normalized_code)
                    failed_lookups.append(
                        f"{normalized_code}: an unexpected lookup error occurred"
                    )

            if invalid_entries:
                st.warning(
                    "Skipped invalid module code(s): "
                    + ", ".join(invalid_entries)
                    + "."
                )
            if loaded_codes:
                st.success(
                    f"Added {len(loaded_codes)} module(s): {', '.join(loaded_codes)}."
                )
            if failed_lookups:
                st.error("Some modules could not be loaded:")
                for failure in failed_lookups:
                    st.caption(f"• {failure}")
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
        except Exception:
            LOGGER.exception("Unexpected NUSMods batch lookup failure")
            st.error("NUSMods could not complete the module lookup. Please try again.")


def render_loaded_nusmods_modules() -> None:
    """Render loaded modules as removable visual cards."""

    modules = st.session_state.get("nusmods_modules", {})
    if not modules:
        st.caption("No NUSMods modules loaded yet.")
        return

    st.markdown("**Loaded NUSMods Modules**")
    st.caption(
        "Each module stays together as one syllabus item during the alignment audit."
    )

    for module_code, record in list(modules.items()):
        title = escape(str(record.get("title", "Untitled module")))
        academic_year = escape(str(record.get("academic_year", "")))

        with st.container():
            info_col, remove_col = st.columns([5, 1])
            with info_col:
                st.markdown(
                    f'<div class="nusmods-card">'
                    f'<div class="nusmods-card-code">{escape(module_code)} · AY {academic_year}</div>'
                    f'<div class="nusmods-card-title">{title}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with remove_col:
                if st.button("✕", key=f"remove_nusmods_{module_code}", use_container_width=True):
                    del st.session_state["nusmods_modules"][module_code]
                    st.rerun()


def render_sidebar() -> tuple[str, str, float, bool, tuple[str, ...], Any]:
    """Render the input form and return its values plus submit state."""

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-mark">CC</div>
                <div class="sidebar-brand-copy">
                    <strong>Career Compass</strong>
                    <span>NUS curriculum audit workspace</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.header("Audit Inputs")
        st.caption("Paste a curriculum and a target role to identify actionable alignment gaps.")
        render_nusmods_lookup()
        render_loaded_nusmods_modules()
        resume_file = st.file_uploader(
            "Optional resume / CV",
            type=["pdf", "docx", "txt"],
            help="Upload a text-based resume. Career Compass will extract relevant experience, internship, and project entries.",
        )
        if resume_file is not None:
            st.caption(f"Resume ready: {resume_file.name}")
        st.divider()

        st.session_state.setdefault("manual_syllabus_input", DEFAULT_SYLLABUS)
        st.session_state.setdefault("job_input", DEFAULT_JOB_DESCRIPTION)

        # Keep text inputs live for Mod Comparison; audit runs on submit.
        syllabus_text = st.text_area(
            "Manual syllabus or additional context",
            key="manual_syllabus_input",
            height=320,
            help="Use this for extra course context. Loaded NUSMods modules are included separately as atomic items.",
        )
        job_text = st.text_area(
            "Target job description",
            key="job_input",
            height=320,
            help="Include responsibilities, required skills, and preferred qualifications.",
        )
        with st.form("alignment_controls", clear_on_submit=False):
            threshold = st.slider(
                "Minimum Alignment Index",
                min_value=0.0,
                max_value=100.0,
                value=DEFAULT_ALIGNMENT_INDEX_THRESHOLD,
                key="alignment_threshold_input",
                step=1.0,
                format="%.0f",
                help="Requirements below this calibrated 0–100 score are flagged as critical skill gaps.",
            )
            st.caption(
                "Displayed scores range from 0–100. The default cutoff is 60/100."
            )
            submitted = st.form_submit_button(
                "Run Alignment Audit",
                type="primary",
                use_container_width=True,
            )

        st.divider()
        st.caption(f"Embedding model: `{MODEL_NAME}`")
        st.caption(
            "Scores combine sentence embeddings with explicit skill evidence and are calibrated "
            "to a 0–100 Career Compass Alignment Index."
        )

    loaded_module_texts = tuple(
        record["alignment_text"]
        for record in st.session_state.get("nusmods_modules", {}).values()
    )
    return syllabus_text, job_text, threshold, submitted, loaded_module_texts, resume_file


def render_header() -> None:
    """Render the application header and social-good framing."""

    st.markdown(
        """
        <div class="dcma-hero">
            <div class="dcma-kicker">Student Hackathon • Social Good</div>
            <h1>Career Compass</h1>
            <div class="dcma-subtitle">
                Connect what NUS students learn with the roles they want—turning uncertain
                module choices into evidence-based pathways to future employment.
            </div>
            <div class="dcma-hero-tags">
                <span class="dcma-hero-tag">NUSMods-ready</span>
                <span class="dcma-hero-tag">Semantic + skill evidence</span>
                <span class="dcma-hero-tag">Actionable micro-credentials</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    """Explain how to start before the first audit is submitted."""

    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">✦</div>
            <div>
                <h3>Your alignment workspace is ready</h3>
                <p>Review the sample inputs in the sidebar, add NUS modules or upload a resume,
                then run an audit to turn curriculum evidence into a focused action plan.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    method_col_1, method_col_2, method_col_3 = st.columns(3)
    with method_col_1:
        st.markdown(
            """
            <div class="method-card">
                <strong>01 · Build your evidence base</strong>
                <p>Load NUSMods modules, paste extra syllabus context, or include a resume.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with method_col_2:
        st.markdown(
            """
            <div class="method-card">
                <strong>02 · Compare with the market</strong>
                <p>Career Compass maps job requirements to the strongest matching curriculum evidence.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with method_col_3:
        st.markdown(
            """
            <div class="method-card">
                <strong>03 · Prioritise next steps</strong>
                <p>See what is covered and where a micro-credential or applied project can help.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div class="section-caption" style="margin-top: 1.35rem;">
            Scores are calibrated to a 0–100 Career Compass Alignment Index so they are easier to
            interpret than raw cosine similarity alone.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mod_comparison(job_text: str, threshold_index: float) -> None:
    """Rank up to four future modules against the active target job."""

    st.subheader("Mod Comparison")
    st.markdown(
        """
        <div class="comparison-intro">
            <div class="comparison-intro-icon">↗</div>
            <div>
                <strong>Choose your next best-fit modules</strong>
                <p>
                    Enter up to four NUS module codes to see which options align most closely
                    with the target role in your sidebar. Use Career Compass Audit to assess the full
                    impact on your current curriculum and resume.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not isinstance(job_text, str) or not job_text.strip():
        st.warning(
            "Add a target job description in the sidebar before comparing future modules."
        )
        return

    academic_year = str(
        st.session_state.get("nusmods_academic_year", default_nus_academic_year())
    ).strip()
    st.info(
        f"Using the sidebar target job description · NUSMods academic year: "
        f"{academic_year or 'not set'}"
    )

    for index in range(MAX_COMPARISON_MODULES):
        st.session_state.setdefault(f"comparison_module_code_{index + 1}", "")

    with st.form("mod_comparison_form", clear_on_submit=False):
        input_columns = st.columns(2)
        candidate_inputs: list[str] = []
        for index in range(MAX_COMPARISON_MODULES):
            with input_columns[index % 2]:
                candidate_inputs.append(
                    st.text_input(
                        f"Candidate module {index + 1}",
                        placeholder="e.g. BT1101",
                        key=f"comparison_module_code_{index + 1}",
                        max_chars=80,
                    )
                )

        compare_clicked = st.form_submit_button(
            "Compare Modules",
            type="primary",
            use_container_width=True,
        )

    if compare_clicked:
        # Clear stale recommendations before starting a new comparison.
        st.session_state["mod_comparison_results"] = []
        st.session_state["mod_comparison_failures"] = []
        st.session_state["mod_comparison_invalid"] = []

        entered_values = [value for value in candidate_inputs if value.strip()]
        if not entered_values:
            st.error("Enter at least one NUS module code to compare.")
            return

        try:
            candidate_codes, invalid_entries = parse_nus_module_codes(
                ", ".join(entered_values)
            )
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
            return

        if len(candidate_codes) > MAX_COMPARISON_MODULES:
            st.error(
                f"Compare up to {MAX_COMPARISON_MODULES} unique module codes at a time."
            )
            return

        comparison_rows: list[dict[str, Any]] = []
        failed_lookups: list[str] = []

        with st.spinner("Fetching modules and calculating fit scores…"):
            for module_code in candidate_codes:
                try:
                    module = fetch_nusmods_module(academic_year, module_code)
                    imported_text = module_to_alignment_text(
                        module,
                        academic_year,
                        module_code,
                    )
                    candidate_result = run_alignment(
                        syllabus_text="",
                        job_text=job_text,
                        threshold_index=threshold_index,
                        syllabus_chunks_override=(imported_text,),
                    )
                    top_match_indexes = sorted(
                        range(len(candidate_result.job_chunks)),
                        key=lambda index: candidate_result.best_matches[index],
                        reverse=True,
                    )[:3]
                    top_matches = tuple(
                        (
                            candidate_result.job_chunks[index],
                            raw_to_alignment_index(candidate_result.best_matches[index]),
                        )
                        for index in top_match_indexes
                    )
                    title = (
                        get_module_value(module, "title", "ModuleTitle")
                        or "Untitled module"
                    )
                    comparison_rows.append(
                        {
                            "module_code": module_code,
                            "title": str(title),
                            "academic_year": academic_year,
                            "alignment_index": float(
                                candidate_result.overall_alignment_index
                            ),
                            "raw_score": float(candidate_result.overall_score),
                            "top_matches": top_matches,
                        }
                    )
                except (ValueError, RuntimeError) as exc:
                    failed_lookups.append(f"{module_code}: {exc}")
                except Exception:
                    LOGGER.exception(
                        "Unexpected module comparison failure for %s", module_code
                    )
                    failed_lookups.append(
                        f"{module_code}: an unexpected comparison error occurred"
                    )

        st.session_state["mod_comparison_results"] = comparison_rows
        st.session_state["mod_comparison_failures"] = failed_lookups
        st.session_state["mod_comparison_invalid"] = invalid_entries

    invalid_entries = st.session_state.get("mod_comparison_invalid", [])
    if invalid_entries:
        st.warning("Skipped invalid module code(s): " + ", ".join(invalid_entries) + ".")

    failed_lookups = st.session_state.get("mod_comparison_failures", [])
    if failed_lookups:
        st.error("Some candidate modules could not be compared:")
        for failure in failed_lookups:
            st.caption(f"• {failure}")

    comparison_rows = st.session_state.get("mod_comparison_results", [])
    if not comparison_rows:
        st.caption(
            "Add between one and four module codes above. The comparison uses the target job "
            "description currently shown in the sidebar."
        )
        return

    ranked_rows = sorted(
        comparison_rows,
        key=lambda row: row["alignment_index"],
        reverse=True,
    )
    winner = ranked_rows[0]
    winner_code = escape(str(winner["module_code"]))
    winner_title = escape(str(winner["title"]))
    winner_score = float(winner["alignment_index"])
    st.markdown(
        f'<div class="comparison-winner">'
        f'<div class="comparison-winner-label">Best fit for this target role</div>'
        f'<div class="comparison-winner-title">{winner_code} · {winner_title}</div>'
        f'<div class="comparison-winner-score">Career Compass alignment index: '
        f'{winner_score:.0f}/100</div></div>',
        unsafe_allow_html=True,
    )

    st.subheader("Ranked module options")
    if pd is not None:
        ranking_frame = pd.DataFrame(
            [
                {
                    "Rank": rank,
                    "Module": row["module_code"],
                    "Module name": row["title"],
                    "Alignment Index": row["alignment_index"],
                    "Raw model score": row["raw_score"],
                }
                for rank, row in enumerate(ranked_rows, start=1)
            ]
        )
        st.dataframe(
            style_dashboard_dataframe(
                ranking_frame,
                format_spec={
                    "Alignment Index": "{:.0f}",
                    "Raw model score": "{:.3f}",
                },
                gradient_subset=["Alignment Index"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    for rank, row in enumerate(ranked_rows, start=1):
        top_matches = row.get("top_matches", ())
        match_summary = "; ".join(
            f"{shorten(requirement, 105)} ({float(score):.0f}/100)"
            for requirement, score in top_matches
        )
        st.markdown(
            f'<div class="comparison-card">'
            f'<div class="comparison-card-title">#{rank} · '
            f'{escape(str(row["module_code"]))} — {escape(str(row["title"]))}</div>'
            f'<div class="comparison-card-meta">'
            f'Career Compass alignment index: {float(row["alignment_index"]):.0f}/100 · '
            f'Academic Year: {escape(str(row["academic_year"]))}</div>'
            f'<div class="comparison-card-match"><strong>Strongest job matches:</strong> '
            f'{escape(match_summary) if match_summary else "No strong requirement matches found."}'
            f'</div></div>',
            unsafe_allow_html=True,
        )


def render_audit_workspace(
    syllabus_text: str,
    job_text: str,
    threshold: float,
    submitted: bool,
    loaded_module_texts: tuple[str, ...],
    resume_file: Any,
) -> None:
    """Render the Career Compass audit workflow inside its dedicated tab."""

    if submitted:
        resume_text: str | None = None
        if resume_file is not None:
            try:
                resume_text = extract_resume_text(resume_file)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
                return

        with st.spinner("Loading the cached language model and auditing alignment…"):
            try:
                result = run_alignment(
                    syllabus_text,
                    job_text,
                    threshold,
                    syllabus_chunks_override=loaded_module_texts,
                    resume_text=resume_text,
                )
            except ValueError as exc:
                st.error(str(exc))
                return
            except RuntimeError as exc:
                st.error(str(exc))
                return
            except Exception:
                LOGGER.exception("Unexpected alignment processing error")
                st.error(
                    "The audit could not be completed because of an unexpected processing error. "
                    "Check the input text and try again."
                )
                return

        st.session_state["alignment_result"] = result
        st.session_state["alignment_threshold_index"] = threshold
        st.session_state["resume_audited_filename"] = (
            resume_file.name if resume_file is not None else None
        )

    result = st.session_state.get("alignment_result")
    if not isinstance(result, AlignmentResult):
        render_empty_state()
        return

    st.markdown(
        f'<div class="audit-status"><span class="audit-status-icon">✓</span>'
        f'<span>Audit complete · compared {len(result.job_chunks)} job requirements '
        f'against {len(result.syllabus_chunks)} syllabus modules/topics.</span></div>',
        unsafe_allow_html=True,
    )

    st.subheader("Alignment Overview")
    metric_col, covered_col, gaps_col, threshold_col = st.columns(4)
    gap_count = len(result.gap_indexes)
    covered_count = len(result.job_chunks) - gap_count
    with metric_col:
        st.metric(
            "Overall Market Alignment Index",
            f"{result.overall_alignment_index:.0f}/100",
            help="Average of the strongest calibrated hybrid match for each job requirement.",
        )
    with covered_col:
        st.metric("Requirements Covered", f"{covered_count}/{len(result.job_chunks)}")
    with gaps_col:
        st.metric("Critical Gaps", gap_count, delta=None)
    with threshold_col:
        st.metric("Minimum Alignment Index", f"{result.threshold_index:.0f}/100")

    st.divider()
    st.subheader("Alignment Matrix")
    st.caption(
        "Each cell combines 75% best sentence-level semantic similarity with 25% explicit skill "
        "evidence, then maps the result to a calibrated 0–100 index. Green indicates stronger "
        "alignment; yellow/red indicates weaker alignment."
    )
    render_matrix(result)

    left_col, right_col = st.columns([1.35, 1])
    with left_col:
        render_met_requirements(result)
        render_gaps(result)
    with right_col:
        st.subheader("Coverage Snapshot")
        render_coverage_summary(result)
        st.caption(
            "A flagged requirement is a prioritization signal—not a judgment of a learner's "
            "ability or the full value of a course."
        )

    if (
        resume_file is not None
        and st.session_state.get("resume_audited_filename") == resume_file.name
    ):
        render_resume_alignment(result)


def main() -> None:
    """Application entry point."""

    configure_page()
    audit_tab, comparison_tab = st.tabs(["Career Compass Audit", "Mod Comparison"])
    (
        syllabus_text,
        job_text,
        threshold,
        submitted,
        loaded_module_texts,
        resume_file,
    ) = render_sidebar()

    if DEPENDENCY_ERROR is not None:
        st.error(
            "The app cannot start because one or more analytics dependencies are missing. "
            "Install `streamlit`, `sentence-transformers`, `scikit-learn`, `pandas`, and `numpy`, "
            "then restart Streamlit."
        )
        with st.expander("Technical detail"):
            st.code(DEPENDENCY_ERROR)
        return

    with audit_tab:
        render_header()
        render_audit_workspace(
            syllabus_text,
            job_text,
            threshold,
            submitted,
            loaded_module_texts,
            resume_file,
        )
    with comparison_tab:
        render_header()
        render_mod_comparison(job_text, threshold)


def run_app_safely() -> None:
    """Render the application without exposing raw tracebacks to end users."""

    try:
        main()
    except Exception:
        LOGGER.exception("Unhandled Career Compass application error")
        st.error(
            "Career Compass could not render this page because of an unexpected application error. "
            "Please refresh the app and try again."
        )


if __name__ == "__main__":
    run_app_safely()
