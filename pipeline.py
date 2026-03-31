"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           OPEN CANCER KNOWLEDGE GRAPH PIPELINE  v1.0                        ║
║                                                                              ║
║  Mission: Map every cross-domain connection in cancer research that          ║
║           siloed databases have never made.                                  ║
║                                                                              ║
║  Sources : PubMed (abstracts + MeSH) · ClinicalTrials.gov · PubChem         ║
║  LLM     : qwen2.5:7b via Ollama (extraction + Q&A)                         ║
║  Embeds  : nomic-embed-text via Ollama                                       ║
║  Output  : NDJSON knowledge records + SQLite graph index                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Setup
-----
pip install requests tqdm

Ollama models (one-time):
  ollama pull qwen2.5:7b
  ollama pull nomic-embed-text

Run examples
------------
  # Test with 100 docs
  python cancer_pipeline.py --source pubmed --limit 100

  # Full run, all sources (leave overnight)
  python cancer_pipeline.py --source pubmed   --limit 50000 --workers 3
  python cancer_pipeline.py --source trials   --limit 20000 --workers 2
  python cancer_pipeline.py --source pubchem  --limit 10000 --workers 2

  # Cross-reference pass (run AFTER all sources complete)
  python cancer_pipeline.py --crossref

What makes this different
-------------------------
Every record extracts the SAME structured schema regardless of source type.
That means a PubMed paper about a compound, a trial testing that compound,
and a PubChem entry for that compound all share: cancer_types, pathways,
compounds, genes - so you can JOIN them later by those fields.

The embedding is computed over a rich semantic string (title + summary +
pathways + compounds) so cosine similarity finds conceptual twins across
sources, decades, and vocabulary differences.

The Q&A pairs are designed specifically for fine-tuning a cancer research
assistant - questions a researcher would actually ask.
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode

import requests
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE  = os.getenv("OLLAMA_BASE",  "http://localhost:11434")
LLM_MODEL    = os.getenv("LLM_MODEL",   "qwen2.5:14b")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "nomic-embed-text")

OUTPUT_DIR   = Path("cancer_output")
DB_PATH      = Path("cancer_pipeline.db")
LOG_PATH     = Path("cancer_pipeline.log")

OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Cancer-domain constants
# ─────────────────────────────────────────────────────────────────────────────

# All major cancer types - used to filter PubMed queries and tag records
CANCER_TYPES = [
    "glioblastoma", "glioma", "astrocytoma", "medulloblastoma",          # brain
    "non-small cell lung cancer", "small cell lung cancer",               # lung
    "breast cancer", "triple-negative breast cancer", "HER2",            # breast
    "colorectal cancer", "colon cancer", "rectal cancer",                 # colorectal
    "pancreatic cancer", "pancreatic ductal adenocarcinoma",              # pancreas
    "hepatocellular carcinoma", "cholangiocarcinoma",                     # liver
    "acute myeloid leukemia", "acute lymphoblastic leukemia",             # leukemia
    "diffuse large B-cell lymphoma", "follicular lymphoma",               # lymphoma
    "multiple myeloma",                                                   # myeloma
    "melanoma", "uveal melanoma",                                         # skin
    "ovarian cancer", "cervical cancer", "endometrial cancer",            # gynecologic
    "prostate cancer", "bladder cancer", "renal cell carcinoma",          # urologic
    "gastric cancer", "esophageal cancer",                                # GI upper
    "thyroid cancer", "adrenocortical carcinoma",                         # endocrine
    "osteosarcoma", "Ewing sarcoma", "rhabdomyosarcoma",                  # sarcoma
    "mesothelioma", "neuroblastoma", "retinoblastoma",                    # rare
]

# Key biological pathways - the connective tissue between papers
PATHWAYS = [
    "apoptosis", "autophagy", "cell cycle arrest", "DNA damage response",
    "angiogenesis", "tumor microenvironment", "immune evasion",
    "epithelial-mesenchymal transition", "metastasis",
    "PI3K/AKT/mTOR", "RAS/MAPK/ERK", "Wnt/beta-catenin", "Notch",
    "Hedgehog", "JAK/STAT", "NF-kB", "p53", "VEGF", "HIF-1alpha",
    "PD-1/PD-L1", "CTLA-4", "CAR-T", "tumor suppressor",
    "oncogene amplification", "synthetic lethality",
    "KRAS", "BRCA1", "BRCA2", "EGFR", "ALK", "ROS1", "BRAF", "IDH1", "IDH2",
]

# PubMed search queries - each fetches a targeted slice
PUBMED_QUERIES = [
    # Mechanism-focused
    "cancer apoptosis resistance mechanism[tiab]",
    "tumor microenvironment immunotherapy[tiab]",
    "synthetic lethality cancer[tiab]",
    "cancer metabolism warburg[tiab]",
    "epigenetic cancer therapy[tiab]",
    "cancer stem cell resistance[tiab]",
    # Compound-focused
    "natural compound anticancer activity[tiab]",
    "drug repurposing cancer[tiab]",
    "cancer combination therapy synergy[tiab]",
    "nanoparticle drug delivery cancer[tiab]",
    # Cross-cancer patterns
    "pan-cancer genomic analysis[tiab]",
    "cancer driver mutation[tiab]",
    "liquid biopsy circulating tumor DNA[tiab]",
    # Underexplored areas
    "traditional medicine cancer[tiab]",
    "microbiome cancer treatment[tiab]",
    "circadian rhythm cancer[tiab]",
    "cancer cachexia mechanism[tiab]",
]

# ─────────────────────────────────────────────────────────────────────────────
# Database - state + graph index
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS processed (
            doc_id      TEXT PRIMARY KEY,
            source      TEXT NOT NULL,
            done_at     TEXT NOT NULL,
            record_path TEXT
        );

        -- Graph index: compound/gene/pathway → doc_id
        -- This is what lets you find "all papers mentioning KRAS + apoptosis"
        CREATE TABLE IF NOT EXISTS entity_index (
            entity_type TEXT NOT NULL,   -- compound | gene | pathway | cancer_type
            entity_name TEXT NOT NULL,
            doc_id      TEXT NOT NULL,
            source      TEXT NOT NULL,
            UNIQUE(entity_type, entity_name, doc_id)
        );

        CREATE INDEX IF NOT EXISTS idx_entity ON entity_index(entity_type, entity_name);
        CREATE INDEX IF NOT EXISTS idx_doc    ON entity_index(doc_id);

        -- Cross-reference candidates (populated by --crossref pass)
        CREATE TABLE IF NOT EXISTS crossref (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id_a     TEXT NOT NULL,
            doc_id_b     TEXT NOT NULL,
            source_a     TEXT NOT NULL,
            source_b     TEXT NOT NULL,
            shared_keys  TEXT NOT NULL,   -- JSON list of what they share
            confidence   REAL NOT NULL,   -- 0-1
            surfaced_at  TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn


def is_done(conn: sqlite3.Connection, doc_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM processed WHERE doc_id=?", (doc_id,)
    ).fetchone() is not None


def mark_done(conn: sqlite3.Connection, doc_id: str, source: str, path: str):
    conn.execute(
        "INSERT OR IGNORE INTO processed (doc_id,source,done_at,record_path) VALUES(?,?,?,?)",
        (doc_id, source, datetime.now(timezone.utc).isoformat(), path),
    )
    conn.commit()


def to_str(x) -> str:
    """Safely convert any entity value to string."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("name", "") or x.get("term", "") or str(x)
    return str(x) if x else ""


def index_entities(conn: sqlite3.Connection, doc_id: str, source: str, record: dict):
    """Index all entities from a record for fast cross-referencing."""
    rows = []
    for ct in record.get("cancer_types", []):
        v = to_str(ct).lower().strip()
        if v: rows.append(("cancer_type", v, doc_id, source))
    for p in record.get("pathways_mentioned", []):
        v = to_str(p).lower().strip()
        if v: rows.append(("pathway", v, doc_id, source))
    for c in record.get("compounds", []):
        v = to_str(c).lower().strip()
        if v: rows.append(("compound", v, doc_id, source))
    for g in record.get("genes_proteins", []):
        v = to_str(g).upper().strip()
        if v: rows.append(("gene", v, doc_id, source))

    conn.executemany(
        "INSERT OR IGNORE INTO entity_index (entity_type,entity_name,doc_id,source) VALUES(?,?,?,?)",
        rows,
    )
    conn.commit()

# ─────────────────────────────────────────────────────────────────────────────
# Ollama helpers
# ─────────────────────────────────────────────────────────────────────────────

def ollama_generate(prompt: str, system: str, temperature: float = 0.05,
                    max_tokens: int = 1500, force_json: bool = False) -> str:
    # force_json kept for API compatibility but not used - schema too complex
    payload = {
        "model":   LLM_MODEL,
        "prompt":  prompt,
        "system":  system,
        "stream":  False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    # Note: not using format=json - complex nested schema breaks it
    for attempt in range(4):
        try:
            r = requests.post(
                f"{OLLAMA_BASE}/api/generate", json=payload, timeout=180
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"Ollama generate attempt {attempt+1} failed ({e}), retry in {wait}s")
            time.sleep(wait)
    return ""


def ollama_embed(text: str) -> list[float]:
    # Truncate to avoid OOM; nomic-embed-text context = 8192 tokens
    payload = {"model": EMBED_MODEL, "prompt": text[:4000]}
    for attempt in range(4):
        try:
            r = requests.post(
                f"{OLLAMA_BASE}/api/embeddings", json=payload, timeout=60
            )
            r.raise_for_status()
            return r.json().get("embedding", [])
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"Ollama embed attempt {attempt+1} failed ({e}), retry in {wait}s")
            time.sleep(wait)
    return []


def parse_json_response(raw: str) -> dict | list | None:
    """Robustly extract JSON from LLM output that may have markdown fences."""
    raw = re.sub(r"^```[a-z]*\n?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find first {...} or [...]
        for pattern in [r"\{.*\}", r"\[.*\]"]:
            m = re.search(pattern, raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Pass 1 - Cancer-specific structured extraction
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """
You extract cancer research data. Return ONLY a JSON object, no text before or after.

Example output format:
{"title":"...","summary":"...","document_type":"research_paper","cancer_types":[],"pathways_mentioned":[],"compounds":[],"genes_proteins":[],"mechanism_of_action":"...","experimental_result":{"effect":"...","model":"...","outcome":"positive","followed_up":false},"similar_terms":[],"potential_connections":[],"year":null,"study_phase":"preclinical","data_quality":"high"}

Rules:
- document_type: research_paper, clinical_trial, compound_profile, or review
- outcome: positive, negative, mixed, or inconclusive
- followed_up: false if this finding was never built upon (= research gap)
- potential_connections: hypotheses like "Compound X inhibits KRAS - untested in pancreatic cancer"
- study_phase: preclinical, phase_1, phase_2, phase_3, approved, or unknown
- data_quality: high, medium, or low
- Return ONLY the JSON. No markdown. No explanation.
""".strip()


def extract_cancer_record(text: str) -> dict:
    prompt = (
        f"Document to extract from:\n\n{text[:4000]}\n\n"
        "Return the JSON extraction:"
    )
    raw = ollama_generate(prompt, system=EXTRACTION_SYSTEM, temperature=0.05, max_tokens=2048, force_json=True)
    result = parse_json_response(raw)
    if isinstance(result, dict):
        return result
    log.warning(f"Extraction failed to parse. Raw (first 200 chars): {repr(raw[:200])}")
    return {}

# ─────────────────────────────────────────────────────────────────────────────
# Pass 2 - Research Q&A generation (fine-tuning data)
# ─────────────────────────────────────────────────────────────────────────────

QA_SYSTEM = """
You are generating training data for a cancer research AI assistant.

Given a biomedical document, produce exactly 5 question-answer pairs in English.
Design questions that a cancer researcher, oncologist, or drug developer would ask.

Question types to include (one each):
1. Factual: "What compound/gene/pathway is described?"
2. Mechanistic: "How does [X] produce its anti-cancer effect?"
3. Clinical: "What stage or cancer type is this most relevant to?"
4. Gap-spotting: "What follow-up experiment would validate this finding?"
5. Connection: "What other cancer type or pathway might this relate to?"

Return ONLY a JSON array:
[
  {"question": "...", "answer": "...", "type": "factual|mechanistic|clinical|gap|connection"},
  ...
]

Each answer: 2-4 sentences. English only. No preamble, no markdown fences.
""".strip()


def generate_cancer_qa(text: str, context_summary: str = "") -> list[dict]:
    hint = f"Key context: {context_summary}\n\n" if context_summary else ""
    prompt = f"{hint}Document:\n\n{text[:3500]}\n\nReturn the JSON array of 5 Q&A pairs:"
    raw = ollama_generate(prompt, system=QA_SYSTEM, temperature=0.35, max_tokens=1200)
    result = parse_json_response(raw)
    if isinstance(result, list):
        return result[:5]
    return []

# ─────────────────────────────────────────────────────────────────────────────
# Pass 3 - Semantic embedding
# ─────────────────────────────────────────────────────────────────────────────

def safe_join(lst) -> str:
    """Join a list safely - handles mixed str/dict/None items."""
    if not lst:
        return ""
    return ", ".join(str(x) if not isinstance(x, str) else x
                     for x in lst if x)


def build_embed_string(record: dict) -> str:
    """
    Construct a rich semantic string for embedding.
    We include pathways + compounds + genes so that cross-source similarity
    works on biology, not just surface text.
    """
    parts = [
        str(record.get("title", "") or ""),
        str(record.get("summary", "") or ""),
        "Cancer types: " + safe_join(record.get("cancer_types", [])),
        "Pathways: "     + safe_join(record.get("pathways_mentioned", [])),
        "Compounds: "    + safe_join(record.get("compounds", [])),
        "Genes: "        + safe_join(record.get("genes_proteins", [])),
        "Similar terms: "+ safe_join(record.get("similar_terms", [])),
    ]
    return " | ".join(p for p in parts if p.strip(" |"))

# ─────────────────────────────────────────────────────────────────────────────
# Full document processing - all 3 passes
# ─────────────────────────────────────────────────────────────────────────────

def process_document(doc_id: str, source: str, title: str, raw_text: str) -> dict | None:
    if not raw_text or not raw_text.strip():
        return None

    # Pass 1: extraction
    extracted = extract_cancer_record(raw_text)
    if not extracted:
        return None

    # Override title if extraction found a cleaner one
    if not extracted.get("title"):
        extracted["title"] = title

    # Pass 2: Q&A
    qa_pairs = generate_cancer_qa(raw_text, context_summary=extracted.get("summary", ""))

    # Pass 3: embedding
    embed_text = build_embed_string(extracted)
    embedding  = ollama_embed(embed_text)

    record = {
        "doc_id":       doc_id,
        "source":       source,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "raw_title":    title,
        **extracted,
        "qa_pairs":     qa_pairs,
        "embed_string": embed_text,      # keep for debugging / re-embedding
        "embedding":    embedding,       # 768-dim float32 list
    }
    return record

# ─────────────────────────────────────────────────────────────────────────────
# Source: PubMed  (E-utilities)
# ─────────────────────────────────────────────────────────────────────────────

PUBMED_BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_EMAIL = os.getenv("NCBI_EMAIL", "cancer.pipeline@research.org")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_DELAY   = 0.12 if NCBI_API_KEY else 0.4


def _ncbi_params(extra: dict) -> dict:
    """Base params for every NCBI request. Adds API key + random jitter delay."""
    import random
    p = {"tool": "cancer_pipeline", "email": PUBMED_EMAIL, **extra}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    time.sleep(NCBI_DELAY + random.uniform(0.05, 0.25))
    return p


def pubmed_search_ids(query: str, retmax: int = 5000) -> list[str]:
    """Return list of PubMed IDs for a query."""
    params = _ncbi_params({
        "db":      "pubmed",
        "term":    query,
        "retmax":  retmax,
        "retmode": "json",
    })
    for attempt in range(3):
        try:
            r = requests.get(f"{PUBMED_BASE}/esearch.fcgi", params=params, timeout=30)
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                log.warning(f"PubMed rate limited (429) - waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()["esearchresult"]["idlist"]
        except Exception as e:
            log.warning(f"PubMed search failed for '{query}' attempt {attempt+1}: {e}")
            time.sleep(5 * (attempt + 1))
    return []


def pubmed_fetch_abstracts(pmids: list[str], batch: int = 200) -> dict[str, dict]:
    """Fetch abstracts for a list of PMIDs. Returns {pmid: {title, abstract, year, mesh}}."""
    results = {}
    for i in range(0, len(pmids), batch):
        chunk = pmids[i:i+batch]
        params = _ncbi_params({
            "db":      "pubmed",
            "id":      ",".join(chunk),
            "retmode": "xml",
        })
        for attempt in range(3):
            try:
                r = requests.get(f"{PUBMED_BASE}/efetch.fcgi", params=params, timeout=60)
                if r.status_code == 429:
                    wait = 60 * (attempt + 1)
                    log.warning(f"PubMed fetch rate limited - waiting {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                break
            except Exception as e:
                log.warning(f"PubMed fetch error attempt {attempt+1}: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            continue

        try:
            root = ET.fromstring(r.content)
        except ET.ParseError as e:
            log.warning(f"PubMed XML parse error: {e}")
            continue

        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            if pmid_el is None:
                continue
            pmid = pmid_el.text.strip()

            title_el    = article.find(".//ArticleTitle")
            abstract_el = article.find(".//AbstractText")
            year_el     = article.find(".//PubDate/Year")

            # MeSH terms - gold-standard controlled vocabulary
            mesh_terms = [
                m.text.strip()
                for m in article.findall(".//MeshHeading/DescriptorName")
                if m.text
            ]

            title    = (title_el.text    or "").strip() if title_el    is not None else ""
            abstract = (abstract_el.text or "").strip() if abstract_el is not None else ""
            year     = int(year_el.text) if year_el is not None and (year_el.text or "").isdigit() else None

            if abstract:
                results[pmid] = {
                    "title":    title,
                    "abstract": abstract,
                    "year":     year,
                    "mesh":     mesh_terms,
                }

        # delay already applied inside _ncbi_params()
    return results


def fetch_pubmed(limit: int = 50000):
    """
    Yield (doc_id, title, text) from PubMed across all cancer queries.
    Text includes abstract + MeSH terms for richer extraction.
    """
    seen = set()
    # Fetch generously per query - we deduplicate and stop at total limit
    per_query = max(500, limit // 3)

    for query in PUBMED_QUERIES:
        if len(seen) >= limit:
            break
        log.info(f"PubMed query: {query} (total so far: {len(seen)})")
        pmids = pubmed_search_ids(query, retmax=per_query)
        new_pmids = [p for p in pmids if p not in seen]
        seen.update(new_pmids)

        if not new_pmids:
            continue

        # Only fetch up to what we still need
        remaining = limit - (len(seen) - len(new_pmids))
        new_pmids = new_pmids[:remaining]

        abstracts = pubmed_fetch_abstracts(new_pmids)
        count = 0
        for pmid, data in abstracts.items():
            doc_id = f"pubmed_{pmid}"
            mesh_str = "MeSH terms: " + "; ".join(data["mesh"]) if data["mesh"] else ""
            text = f"{data['title']}\n\n{data['abstract']}\n\n{mesh_str}".strip()
            yield doc_id, data["title"], text
            count += 1

        log.info(f"  → {count} new docs from this query")

# ─────────────────────────────────────────────────────────────────────────────
# Source: ClinicalTrials.gov  (v2 API, cancer-filtered)
# ─────────────────────────────────────────────────────────────────────────────

CT_API = "https://clinicaltrials.gov/api/v2/studies"


def fetch_trials(limit: int = 20000):
    """
    Yield (doc_id, title, text) for cancer clinical trials.
    Filters to interventional trials with cancer conditions.
    """
    fetched   = 0
    next_page = None
    page_size = min(100, limit)

    while fetched < limit:
        params = {
            "format":   "json",
            "pageSize": page_size,
            "query.cond": "cancer OR carcinoma OR leukemia OR lymphoma OR sarcoma OR melanoma OR glioma",
            "filter.overallStatus": "COMPLETED,TERMINATED,ACTIVE_NOT_RECRUITING",
            "fields": (
                "NCTId,BriefTitle,OfficialTitle,BriefSummary,DetailedDescription,"
                "Condition,InterventionName,InterventionType,Phase,OverallStatus,"
                "PrimaryOutcomeMeasure,StudyType,EnrollmentCount"
            ),
        }
        if next_page:
            params = {"format": "json", "pageToken": next_page, "pageSize": page_size}

        try:
            r = requests.get(CT_API, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning(f"ClinicalTrials fetch error: {e}")
            time.sleep(5)
            continue

        for study in data.get("studies", []):
            proto   = study.get("protocolSection", {})
            id_mod  = proto.get("identificationModule",  {})
            desc_mod= proto.get("descriptionModule",     {})
            cond_mod= proto.get("conditionsModule",      {})
            int_mod = proto.get("armsInterventionsModule", {})
            stat_mod= proto.get("statusModule",          {})
            out_mod = proto.get("outcomesModule",        {})

            nct_id  = id_mod.get("nctId", "")
            title   = id_mod.get("officialTitle") or id_mod.get("briefTitle", "")
            summary = desc_mod.get("briefSummary", "")
            detail  = desc_mod.get("detailedDescription", "")
            conds   = "; ".join(cond_mod.get("conditions", []))
            phase   = stat_mod.get("phase", "")
            status  = stat_mod.get("overallStatus", "")
            outcomes= "; ".join(
                o.get("measure", "") for o in out_mod.get("primaryOutcomes", [])
            )

            interventions = []
            for i in int_mod.get("interventions", []):
                iname = i.get("interventionName", "")
                itype = i.get("interventionType", "")
                if iname:
                    interventions.append(f"{iname} ({itype})" if itype else iname)

            if not nct_id or not (summary or detail):
                continue

            text = (
                f"{title}\n\n"
                f"Status: {status} | Phase: {phase}\n"
                f"Conditions: {conds}\n"
                f"Interventions: {'; '.join(interventions)}\n"
                f"Primary outcomes: {outcomes}\n\n"
                f"{summary}\n\n{detail}"
            ).strip()

            yield f"trial_{nct_id}", title, text
            fetched += 1
            if fetched >= limit:
                break

        next_page = data.get("nextPageToken")
        if not next_page:
            break
        time.sleep(0.5)

# ─────────────────────────────────────────────────────────────────────────────
# Source: PubChem  (cancer-relevant bioassay compounds)
# ─────────────────────────────────────────────────────────────────────────────

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def fetch_pubchem_compound(cid: int) -> dict | None:
    """Fetch compound data from PubChem by CID."""
    try:
        # Properties
        r = requests.get(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/property/"
            "IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES/JSON",
            timeout=20,
        )
        if r.status_code != 200:
            return None
        props = r.json()["PropertyTable"]["Properties"][0]

        # Description
        r2 = requests.get(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/description/JSON", timeout=20
        )
        desc = ""
        if r2.status_code == 200:
            info_list = r2.json().get("InformationList", {}).get("Information", [])
            for item in info_list:
                if item.get("Description"):
                    desc = item["Description"]
                    break

        # Get the common name from description Title if IUPACName is missing
        title_name = ""
        if r2.status_code == 200:
            for item in r2.json().get("InformationList", {}).get("Information", []):
                if item.get("Title"):
                    title_name = item["Title"]
                    break

        return {
            "cid":      cid,
            "name":     title_name or props.get("IUPACName", f"Compound {cid}"),
            "formula":  props.get("MolecularFormula", ""),
            "weight":   str(props.get("MolecularWeight", "")),
            "smiles":   props.get("CanonicalSMILES", "") or props.get("IsomericSMILES", ""),
            "description": desc,
        }
    except Exception as e:
        log.warning(f"PubChem CID {cid} fetch error: {e}")
        return None


def pubchem_search_cancer_cids(query: str, limit: int = 500) -> list[int]:
    """Search PubChem for compounds related to a cancer query."""
    try:
        r = requests.get(
            f"{PUBCHEM_BASE}/compound/name/{requests.utils.quote(query)}/cids/JSON",
            timeout=20,
        )
        if r.status_code == 200:
            return r.json()["IdentifierList"]["CID"][:limit]
    except Exception:
        pass
    return []


def fetch_pubchem(limit: int = 10000):
    """
    Yield (doc_id, title, text) for cancer-relevant compounds from PubChem.
    Uses specific compound names known to be cancer-relevant - PubChem API
    requires exact compound names, not generic search terms.
    """
    # Specific compounds - top cancer drugs + natural compounds + targeted therapy
    # These are the exact names PubChem recognizes
    CANCER_COMPOUNDS = [
        # Chemotherapy classics
        "doxorubicin", "paclitaxel", "cisplatin", "carboplatin", "oxaliplatin",
        "cyclophosphamide", "fluorouracil", "gemcitabine", "vincristine",
        "methotrexate", "irinotecan", "topotecan", "etoposide", "bleomycin",
        # Targeted therapy
        "imatinib", "erlotinib", "gefitinib", "sorafenib", "sunitinib",
        "vemurafenib", "dabrafenib", "trametinib", "osimertinib", "alectinib",
        "crizotinib", "ceritinib", "lorlatinib", "palbociclib", "ribociclib",
        "abemaciclib", "lapatinib", "neratinib", "tucatinib", "afatinib",
        "sotorasib", "adagrasib", "idelalisib", "copanlisib", "alpelisib",
        # PARP inhibitors
        "olaparib", "rucaparib", "niraparib", "talazoparib", "veliparib",
        # Immunotherapy small molecules
        "pembrolizumab", "nivolumab", "ipilimumab", "atezolizumab",
        # Natural compounds with anticancer evidence
        "curcumin", "resveratrol", "quercetin", "berberine", "capsaicin",
        "epigallocatechin gallate", "sulforaphane", "lycopene", "apigenin",
        "luteolin", "kaempferol", "fisetin", "honokiol", "triptolide",
        # Other important compounds
        "tamoxifen", "letrozole", "anastrozole", "exemestane", "fulvestrant",
        "enzalutamide", "abiraterone", "bicalutamide", "docetaxel", "cabazitaxel",
        "temozolomide", "bevacizumab", "trastuzumab", "rituximab", "cetuximab",
        "bortezomib", "carfilzomib", "ixazomib", "thalidomide", "lenalidomide",
        "azacitidine", "decitabine", "vorinostat", "romidepsin", "panobinostat",
        "melatonin", "metformin", "aspirin", "celecoxib", "rapamycin",
        "everolimus", "temsirolimus", "ixabepilone", "eribulin", "trabectedin",
    ]

    seen = set()
    fetched = 0

    for compound_name in CANCER_COMPOUNDS:
        if fetched >= limit:
            break

        # Search by exact compound name
        try:
            r = requests.get(
                f"{PUBCHEM_BASE}/compound/name/{requests.utils.quote(compound_name)}/cids/JSON",
                timeout=20,
            )
            if r.status_code != 200:
                log.warning(f"PubChem: no CID for '{compound_name}'")
                continue
            cids = r.json()["IdentifierList"]["CID"][:3]  # top 3 matches
        except Exception as e:
            log.warning(f"PubChem search error for '{compound_name}': {e}")
            continue

        for cid in cids:
            if cid in seen or fetched >= limit:
                continue
            seen.add(cid)

            compound = fetch_pubchem_compound(cid)
            if not compound:
                continue

            # Use compound name even if description is empty
            doc_id = f"pubchem_{cid}"
            title  = compound_name
            desc   = compound["description"] or f"Chemical compound used in cancer research. Formula: {compound['formula']}."
            text   = (
                f"{title}\n\n"
                f"Formula: {compound['formula']} | MW: {compound['weight']}\n"
                f"SMILES: {compound['smiles']}\n\n"
                f"Description: {desc}"
            ).strip()

            if not text.strip():
                continue

            yield doc_id, title, text
            fetched += 1
            time.sleep(0.2)

# ─────────────────────────────────────────────────────────────────────────────
# Output writer
# ─────────────────────────────────────────────────────────────────────────────

def get_output_paths(source: str) -> tuple[Path, Path]:
    """
    Two separate output files written simultaneously:
      kg_*.jsonl  - knowledge graph: title, summary, entities, embeddings
      qa_*.jsonl  - finetune Q&A pairs + minimal context
    """
    kg = OUTPUT_DIR / f"kg_{source}.jsonl"
    qa = OUTPUT_DIR / f"qa_{source}.jsonl"
    return kg, qa


def write_split(kg_writer, qa_writer, record: dict):
    """Write one record split across KG file and QA file."""
    # KG: everything except qa_pairs
    kg_rec = {k: v for k, v in record.items() if k != "qa_pairs"}
    kg_writer.write(json.dumps(kg_rec, ensure_ascii=False) + "\n")

    # QA: one line per pair with just enough context for fine-tuning
    for pair in record.get("qa_pairs", []):
        qa_rec = {
            "instruction":  pair.get("question", ""),
            "output":       pair.get("answer",   ""),
            "type":         pair.get("type",     ""),
            "input":        record.get("title",  ""),
            "source_id":    record["doc_id"],
            "source":       record["source"],
            "cancer_types": record.get("cancer_types", []),
            "pathways":     record.get("pathways_mentioned", []),
        }
        qa_writer.write(json.dumps(qa_rec, ensure_ascii=False) + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# Cross-reference pass - the secret weapon
# ─────────────────────────────────────────────────────────────────────────────

def run_crossref(conn: sqlite3.Connection):
    """
    Find cross-source connections: a PubMed paper and a clinical trial
    that share the same compound + cancer type but cite different pathways.

    These are the 'missed connections' - the whole point of this pipeline.
    """
    log.info("Starting cross-reference pass...")

    # Find all (compound, cancer_type) pairs that appear in more than one source
    cursor = conn.execute("""
        SELECT a.entity_name, a.doc_id, a.source, b.entity_name, b.doc_id, b.source
        FROM entity_index a
        JOIN entity_index b ON (
            a.doc_id   != b.doc_id AND
            a.source   != b.source AND
            a.entity_type = 'compound' AND
            b.entity_type = 'compound' AND
            a.entity_name = b.entity_name
        )
        LIMIT 100000
    """)

    compound_pairs = cursor.fetchall()
    log.info(f"Found {len(compound_pairs)} compound cross-source pairs")

    connections_found = 0
    for row in compound_pairs:
        compound, doc_a, src_a, _, doc_b, src_b = row

        # Check if they also share a cancer type (stronger signal)
        shared_cancer = conn.execute("""
            SELECT a.entity_name FROM entity_index a
            JOIN entity_index b ON a.entity_name = b.entity_name
            WHERE a.doc_id=? AND b.doc_id=? AND a.entity_type='cancer_type'
        """, (doc_a, doc_b)).fetchall()

        # Check if they share pathways
        shared_path = conn.execute("""
            SELECT a.entity_name FROM entity_index a
            JOIN entity_index b ON a.entity_name = b.entity_name
            WHERE a.doc_id=? AND b.doc_id=? AND a.entity_type='pathway'
        """, (doc_a, doc_b)).fetchall()

        shared_keys = {
            "compound":    compound,
            "cancer_types": [r[0] for r in shared_cancer],
            "pathways":    [r[0] for r in shared_path],
        }

        # Confidence: compound match=0.4, each shared cancer=+0.2, each shared pathway=+0.15
        confidence = 0.4
        confidence += min(0.3, len(shared_cancer) * 0.2)
        confidence += min(0.3, len(shared_path)   * 0.15)
        confidence = min(1.0, confidence)

        if confidence >= 0.5:
            conn.execute("""
                INSERT OR IGNORE INTO crossref
                (doc_id_a, doc_id_b, source_a, source_b, shared_keys, confidence, surfaced_at)
                VALUES (?,?,?,?,?,?,?)
            """, (
                doc_a, doc_b, src_a, src_b,
                json.dumps(shared_keys),
                round(confidence, 3),
                datetime.now(timezone.utc).isoformat(),
            ))
            connections_found += 1

    conn.commit()
    log.info(f"Cross-reference pass complete: {connections_found} connections surfaced")

    # Print top connections
    top = conn.execute("""
        SELECT doc_id_a, source_a, doc_id_b, source_b, shared_keys, confidence
        FROM crossref
        ORDER BY confidence DESC
        LIMIT 20
    """).fetchall()

    print("\n" + "═"*70)
    print("TOP 20 CROSS-SOURCE CONNECTIONS FOUND")
    print("═"*70)
    for row in top:
        keys = json.loads(row[4])
        print(f"\nConfidence: {row[5]:.2f}")
        print(f"  {row[1]:10} → {row[0]}")
        print(f"  {row[3]:10} → {row[2]}")
        print(f"  Shared compound: {keys['compound']}")
        if keys['cancer_types']:
            print(f"  Shared cancers:  {', '.join(keys['cancer_types'][:3])}")
        if keys['pathways']:
            print(f"  Shared pathways: {', '.join(keys['pathways'][:3])}")
    print("\n" + "═"*70)

# ─────────────────────────────────────────────────────────────────────────────
# Stats printer
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(conn: sqlite3.Connection):
    total = conn.execute("SELECT COUNT(*) FROM processed").fetchone()[0]
    by_source = conn.execute(
        "SELECT source, COUNT(*) FROM processed GROUP BY source"
    ).fetchall()
    top_compounds = conn.execute("""
        SELECT entity_name, COUNT(*) as c FROM entity_index
        WHERE entity_type='compound' GROUP BY entity_name ORDER BY c DESC LIMIT 10
    """).fetchall()
    top_cancers = conn.execute("""
        SELECT entity_name, COUNT(*) as c FROM entity_index
        WHERE entity_type='cancer_type' GROUP BY entity_name ORDER BY c DESC LIMIT 10
    """).fetchall()
    crossrefs = conn.execute("SELECT COUNT(*) FROM crossref").fetchone()[0]

    print("\n" + "═"*60)
    print("PIPELINE STATS")
    print("═"*60)
    print(f"Total documents processed: {total}")
    for src, cnt in by_source:
        print(f"  {src:20} {cnt:>6}")
    print(f"\nCross-source connections:  {crossrefs}")
    print(f"\nTop compounds across corpus:")
    for name, cnt in top_compounds:
        print(f"  {name:30} {cnt:>5} docs")
    print(f"\nTop cancer types across corpus:")
    for name, cnt in top_cancers:
        print(f"  {name:40} {cnt:>5} docs")
    print("═"*60 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# Ollama health check
# ─────────────────────────────────────────────────────────────────────────────

def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = {m["name"] for m in r.json().get("models", [])}
        missing = []
        for needed in [LLM_MODEL, EMBED_MODEL]:
            if not any(needed.split(":")[0] in m for m in models):
                missing.append(needed)
        if missing:
            log.error(f"Missing Ollama models: {missing}")
            log.error(f"Run: ollama pull {' && ollama pull '.join(missing)}")
            sys.exit(1)
        log.info(f"Ollama OK | models: {sorted(models)}")
    except requests.exceptions.ConnectionError:
        log.error(f"Cannot reach Ollama at {OLLAMA_BASE}")
        log.error("Make sure Ollama is running: ollama serve")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_MAP = {
    "pubmed":  fetch_pubmed,
    "trials":  fetch_trials,
    "pubchem": fetch_pubchem,
}


def run(source: str, limit: int, workers: int):
    check_ollama()
    conn          = init_db()
    fetcher       = SOURCE_MAP[source]
    kg_path, qa_path = get_output_paths(source)
    kg_writer     = open(kg_path, "a", encoding="utf-8")
    qa_writer     = open(qa_path, "a", encoding="utf-8")

    log.info(f"Pipeline start | source={source} limit={limit} workers={workers}")
    log.info(f"KG  output → {kg_path}")
    log.info(f"QA  output → {qa_path}")

    done = skipped = errors = 0

    # Cache raw docs locally so we never re-fetch from the API
    cache_path = OUTPUT_DIR / f"raw_cache_{source}_{limit}.jsonl"
    if cache_path.exists():
        log.info(f"Loading raw docs from cache: {cache_path}")
        docs = []
        with open(cache_path, encoding="utf-8") as cf:
            for line in cf:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    docs.append((item["doc_id"], item["title"], item["text"]))
        log.info(f"Loaded {len(docs)} docs from cache")
    else:
        log.info(f"Fetching docs from {source} (will cache to {cache_path})...")
        docs = list(fetcher(limit=limit))
        log.info(f"Fetched {len(docs)} raw docs - saving cache...")
        with open(cache_path, "w", encoding="utf-8") as cf:
            for doc_id, title, text in docs:
                cf.write(json.dumps({
                    "doc_id": doc_id,
                    "title":  title,
                    "text":   text,
                }, ensure_ascii=False) + "\n")
        log.info(f"Cache saved → {cache_path}")

    # Load all done IDs into memory once - avoids hammering SQLite with workers
    done_ids = set(
        row[0] for row in
        conn.execute("SELECT doc_id FROM processed WHERE source=?", (source,)).fetchall()
    )
    log.info(f"Skipping {len(done_ids)} already-processed docs")

    def process_one(item):
        doc_id, title, text = item
        if doc_id in done_ids:
            return None, doc_id, "skip"
        try:
            record = process_document(doc_id, source, title, text)
            return record, doc_id, "ok" if record else "empty"
        except Exception as e:
            log.warning(f"Error on {doc_id}: {e}")
            return None, doc_id, "error"

    with tqdm(total=len(docs), unit="doc", desc=source, dynamic_ncols=True) as bar:
        executor_cls = ThreadPoolExecutor if workers > 1 else None

        if workers == 1:
            for item in docs:
                record, doc_id, status = process_one(item)
                bar.update(1)
                if status == "ok" and record:
                    write_split(kg_writer, qa_writer, record)
                    kg_writer.flush()
                    qa_writer.flush()
                    mark_done(conn, doc_id, source, str(kg_path))
                    index_entities(conn, doc_id, source, record)
                    done += 1
                elif status == "skip":
                    skipped += 1
                else:
                    errors += 1
                bar.set_postfix(done=done, skip=skipped, err=errors, refresh=False)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(process_one, item): item for item in docs}
                for fut in as_completed(futs):
                    record, doc_id, status = fut.result()
                    bar.update(1)
                    if status == "ok" and record:
                        write_split(kg_writer, qa_writer, record)
                        kg_writer.flush()
                        qa_writer.flush()
                        mark_done(conn, doc_id, source, str(kg_path))
                        index_entities(conn, doc_id, source, record)
                        done += 1
                    elif status == "skip":
                        skipped += 1
                    else:
                        errors += 1
                    bar.set_postfix(done=done, skip=skipped, err=errors, refresh=False)

    kg_writer.close()
    qa_writer.close()
    log.info(f"Done | processed={done} skipped={skipped} errors={errors}")
    log.info(f"KG  → {kg_path}")
    log.info(f"QA  → {qa_path}")
    print_stats(conn)
    conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Open Cancer Knowledge Graph Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cancer_pipeline.py --source pubmed  --limit 200          # quick test
  python cancer_pipeline.py --source pubmed  --limit 50000        # full run
  python cancer_pipeline.py --source trials  --limit 20000
  python cancer_pipeline.py --source pubchem --limit 10000
  python cancer_pipeline.py --crossref                            # find connections
  python cancer_pipeline.py --stats                               # print stats
        """,
    )
    parser.add_argument("--source",   choices=list(SOURCE_MAP),
                        help="Data source to process")
    parser.add_argument("--limit",    type=int, default=5000,
                        help="Max documents (default: 5000)")
    parser.add_argument("--workers",  type=int, default=2,
                        help="Parallel workers (default: 2)")
    parser.add_argument("--crossref", action="store_true",
                        help="Run cross-reference pass on processed data")
    parser.add_argument("--stats",    action="store_true",
                        help="Print corpus statistics")
    args = parser.parse_args()

    if args.crossref:
        conn = init_db()
        run_crossref(conn)
        conn.close()
    elif args.stats:
        conn = init_db()
        print_stats(conn)
        conn.close()
    elif args.source:
        run(source=args.source, limit=args.limit, workers=args.workers)
    else:
        parser.print_help()