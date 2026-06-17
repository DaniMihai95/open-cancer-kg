---
license: cc-by-4.0
language:
- en
tags:
- cancer
- biomedical
- knowledge-graph
- embeddings
- pubmed
- clinical-trials
- drug-discovery
- literature-based-discovery
size_categories:
- 10K<n<100K
task_categories:
- other
- feature-extraction
pretty_name: Open Cancer Knowledge Graph (OCKG)
---

# Open Cancer Knowledge Graph (OCKG)

> *The first open, locally-runnable pipeline combining LLM-based structured extraction, vector embeddings, and cross-database linking of PubMed, ClinicalTrials.gov, and PubChem for cancer research gap detection - requiring no budget, no institutional access, and no proprietary tools.*

**Pipeline code on GitHub →** [github.com/DaniMihai95/open-cancer-kg](https://github.com/DaniMihai95/open-cancer-kg)

---

## Dataset name

**OCKG - Open Cancer Knowledge Graph v1.0**

---

## The problem

Cancer research is fragmented across three major public databases that have never been systematically cross-referenced at the document level:

- **PubMed** - 35M+ paper abstracts, unstructured text
- **ClinicalTrials.gov** - 500k+ registered trials, siloed
- **PubChem** - 100M+ chemical compounds, disconnected from literature

A compound tested in a 1994 breast cancer paper may share a biological pathway with a 2021 lung trial that failed for an unrelated reason. Because vocabulary differs, journals differ, and no system links them semantically, that connection is never made.

This is the *undiscovered public knowledge* problem (Swanson, 1986). This pipeline solves it automatically, at scale, across all cancer types simultaneously.

---

## Dataset statistics (v1.0)

| Source | Documents | Status |
|--------|-----------|--------|
| PubMed | 22,301 | ✅ complete |
| ClinicalTrials.gov | 19,988 | ✅ complete |
| PubChem | 92 | ✅ complete |
| **Total** | **42,381** | ✅ |

Additional outputs (not released publicly):
- 200,000+ Q&A pairs for LLM fine-tuning (5 per document)
- 10,346 research gap hypotheses flagged by the pipeline

Known limitations:
- 2 corrupted records excluded (pipeline interruption during writing)
- ~15% of records may have incomplete entity extraction (vague abstracts)
- `followed_up` field is an LLM judgment from abstract text alone, not citation-verified
- First 2,090 PubMed records processed with qwen2.5:14b, remainder with qwen2.5:7b

---

## How it differs from existing systems

| System | LLM extraction | Embeddings | Cross-DB | Gap detection | Open/free | Cancer-focused |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| Open Targets | ❌ | ❌ | Partial | ❌ | Partial | Partial |
| SemMedDB | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| SPOKE | ❌ | ❌ | ✅ | ❌ | Partial | ❌ |
| BioGPT | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| iKraph | ✅ | ❌ | Partial | ❌ | ❌ | ❌ |
| PKG2.0 | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| **OCKG (this work)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

No existing public system combines all six properties.

---

## Top entities in the corpus

**Top compounds:** doxorubicin (1,212 docs), paclitaxel (578), cisplatin (542), curcumin (428), chitosan (330), melatonin (327), hyaluronic acid (263), docetaxel (253), gemcitabine (246), PARP inhibitors (240)

**Top cancer types:** breast cancer (2,007 docs), colorectal cancer (1,377), prostate cancer (773), lung cancer (686), ovarian cancer (659), melanoma (633), hepatocellular carcinoma (624)

---

## What each record contains

Every document - regardless of source - is structured into the same schema:

```json
{
  "doc_id": "pubmed_38291045",
  "source": "pubmed",
  "title": "...",
  "summary": "3-5 sentence plain-English summary",
  "document_type": "research_paper",
  "cancer_types": ["glioblastoma", "NSCLC"],
  "pathways_mentioned": ["PI3K/AKT/mTOR", "apoptosis"],
  "compounds": ["temozolomide", "bevacizumab"],
  "genes_proteins": ["EGFR", "p53", "KRAS"],
  "mechanism_of_action": "...",
  "experimental_result": {
    "effect": "inhibited tumor growth by 60%",
    "model": "xenograft mouse",
    "outcome": "positive",
    "followed_up": false
  },
  "potential_connections": [
    "Compound X blocks KRAS-G12C - never tested in pancreatic cancer"
  ],
  "similar_terms": ["kinase inhibitor", "targeted therapy"],
  "study_phase": "preclinical",
  "data_quality": "high",
  "embed_string": "...",
  "embedding": [0.021, -0.034, "..."]
}
```

The `followed_up: false` flag marks findings the LLM judged as never built upon - research gap candidates. The `embedding` field is a 768-dimensional semantic fingerprint (nomic-embed-text) enabling cosine similarity search across the entire corpus regardless of vocabulary, journal, or decade.

Q&A pairs are not included in this public release.

---

## Cross-source connections found

After processing all three sources, the pipeline identified 20 cross-source connections - documents from different databases sharing the same compound, cancer type, and biological pathway without citing each other.

Example finding:
```
Confidence: 0.75
  pubmed → pubmed_37326467
  trials → trial_NCT05372640
  Shared compound:  abemaciclib
  Shared cancer:    breast cancer
  Shared pathway:   CDK4/6 pathway
```

Another finding:
```
Confidence: 0.55
  trial → NCT06328387
  pubmed → 9 separate papers
  Shared compound:  chloroquine
  Shared pathway:   autophagy
```

---

## Setup

```bash
pip install requests tqdm
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Full pipeline code at: [github.com/DaniMihai95/open-cancer-kg](https://github.com/DaniMihai95/open-cancer-kg)

Optional - free NCBI API key for higher rate limits (10 req/sec vs 3):
1. Register at https://www.ncbi.nlm.nih.gov/account/
2. Account Settings → API Key Management → Generate
3. Use: `NCBI_API_KEY=your_key python pipeline.py ...`

---

## Run order

```bash
# Test first
python pipeline.py --source pubmed --limit 100 --workers 2

# Full runs - fully resumable if interrupted
python pipeline.py --source pubmed  --limit 50000 --workers 3
python pipeline.py --source trials  --limit 20000 --workers 3
python pipeline.py --source pubchem --limit 10000 --workers 3

# Find cross-source connections
python pipeline.py --crossref

# Statistics
python pipeline.py --stats
```

---

## Actual performance measured

| Source | Docs | Time (qwen2.5:7b, RTX 4060 Ti 16GB) |
|--------|------|--------------------------------------|
| PubMed | 22,338 | ~55 hours |
| ClinicalTrials | 19,988 | ~68 hours |
| PubChem | 92 | ~2 hours |

Workers=3, power-limited to 125W for sustained operation.

---

## Query your graph

```bash
# Find all documents mentioning a compound
python query.py --entity compound "sotorasib"

# Find cross-source connections
python query.py --connections "sotorasib"

# Semantic search
python query.py --search "KRAS mutation untested compound pancreatic"

# Export connections to CSV
python query.py --export-connections connections.csv
```

---

## Data sources and licensing

All source data is public domain:

| Source | Owner | License |
|--------|-------|---------|
| PubMed abstracts | US National Library of Medicine | Public domain |
| ClinicalTrials.gov | US federal government | Public domain |
| PubChem | NIH | Public domain |

This dataset (extracted JSON records + embeddings) is released under **CC BY 4.0** - free to use with attribution.

Pipeline code is released under **MIT License**.

Q&A pairs are not released publicly.

---

## Academic context

This work is a contribution to **literature-based discovery (LBD)** and the *undiscovered public knowledge* problem (Swanson, 1986).

**Related work:** Arsenyan et al. 2024 (BioNLP), iKraph 2023, PubMed KG 2.0 (Xu et al. 2024), Borchert et al. 2024, Sarol et al. 2024, BioStrataKG 2024.

**Affiliation:** Independent student research, Tilburg University, The Netherlands.

---

## Citation

```bibtex
@dataset{ockg2026,
  title     = {Open Cancer Knowledge Graph (OCKG) v1.0},
  author    = {Pocatilu Daniel Mihai},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/pdm95/open-cancer-kg}
}
```

---

*Built on a student's GPU. Costs nothing to run. Free for any researcher anywhere.*