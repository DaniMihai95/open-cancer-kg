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

This is the *undiscovered public knowledge* problem (Swanson, 1986). This pipeline structures papers, trials and compounds into one schema and proposes links between them, so a person can check the ones worth reading together.

---

## Dataset statistics (v1.0)

| Source | Documents | Status |
|--------|-----------|--------|
| PubMed | 22,301 | ✅ complete |
| ClinicalTrials.gov | 19,988 | ✅ complete |
| PubChem | 92 | ✅ complete |
| **Total** | **42,381** | ✅ |

Not released: about 200,000 Q&A pairs for fine-tuning (5 per document).

Known limitations, measured on the released files (see the paper and `analysis/paper_numbers.py`):
- 2 corrupted lines excluded (pipeline interruption during writing); 12 document IDs appear twice.
- Only 21.8% of the trial records mention cancer. The trial fetcher sent the cancer filter with the first page only, and ClinicalTrials.gov returns all studies when a later page request carries no filter. The current code repeats the filter on every page.
- All four entity fields (cancer types, pathways, compounds, genes) are empty in 30.0% of records: 13.6% of PubMed records, 48.4% of trial records.
- Entity names are not normalised: breast cancer appears under 124 different strings.
- `followed_up` is false in 75.5% of records, trials included, so it is not a usable gap signal. The `outcome` field of trial records comes from registration text, which holds no results.
- First 2,090 PubMed records processed with qwen2.5:14b, the rest with qwen2.5:7b.

---

## Related systems

SemMedDB, SPOKE, the Open Targets Platform, PubMed Knowledge Graph 2.0 and iKraph are larger biomedical knowledge graphs, and PubMed Knowledge Graph 2.0 already links papers to clinical trials. OCKG is small and cancer-specific: one local language model fills the same schema for every source, every record has an embedding, and the whole pipeline runs on one consumer GPU. The paper compares them in its Related Work section.

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

The `followed_up: false` flag marks findings the LLM judged as never built upon - research gap candidates. The `embedding` field is a semantic fingerprint (768 numbers from nomic-embed-text in dataset v1.0, 2,560 from qwen3-embedding:4b with the current code) enabling cosine similarity search across the entire corpus regardless of vocabulary, journal, or decade.

Q&A pairs are not included in this public release.

---

## Cross-source connections found

The linking rule pairs documents from different sources that share a compound, and keeps a pair when it also shares a cancer type or a pathway. On the released data it finds 13,324 such pairs, 43 of which share a compound, a cancer type and a pathway. `--crossref` prints the top 20. The pipeline does not check citations, so the two documents in a pair may already cite each other. The code used for dataset v1.0 read at most 100,000 candidate rows, about 15% of the candidates. The current `--crossref` scores every candidate pair once.

Example finding:
```
Confidence: 0.75
  pubmed → pubmed_37326467
  trials → trial_NCT05372640
  Shared compound:  abemaciclib
  Shared cancer:    breast cancer
  Shared pathway:   CDK4/6 pathway
```


---

## Setup

```bash
pip install requests tqdm
ollama pull qwen3.6:35b          # extraction (dataset v1.0 used qwen2.5:7b)
ollama pull qwen3-embedding:4b   # embeddings (dataset v1.0 used nomic-embed-text)
```

The two models can run on different machines. `OLLAMA_BASE` is the Ollama server for extraction, `EMBED_BASE` the one for embeddings (it defaults to `OLLAMA_BASE`), and `LLM_MODEL` and `EMBED_MODEL` choose the models. Thinking is turned off for the extraction model. `query.py --search` must embed the query with the same model that embedded the records, so set `EMBED_MODEL=nomic-embed-text` when you search dataset v1.0.

Full pipeline code at: [github.com/DaniMihai95/open-cancer-kg](https://github.com/DaniMihai95/open-cancer-kg)

Optional - free NCBI API key for higher rate limits (10 req/sec vs 3):
1. Register at https://www.ncbi.nlm.nih.gov/account/
2. Account Settings → API Key Management → Generate
3. Use: `NCBI_API_KEY=your_key python pipeline.py ...`

---

## Run everything on one machine

`run_all.sh` sets up a Python environment, checks that PubMed, ClinicalTrials.gov and PubChem answer, then runs every step below with the same limits as dataset v1.0. Ollama with both models runs on the same machine. The embedding model runs on the CPU (`EMBED_NUM_GPU=0`) so the extraction model keeps the graphics cards, and Ollama must be allowed to keep both loaded (`OLLAMA_MAX_LOADED_MODELS=2`), or it swaps them on every record.

```bash
./run_all.sh test   # 100 PubMed records, to check the setup
./run_all.sh        # the full run, several days; run it again to resume
```

Outputs go to `work/`.

## Run order

```bash
# Test first
python pipeline.py --source pubmed --limit 100 --workers 2

# Full runs - fully resumable if interrupted. Add --qa to also write 5 Q&A pairs per document (doubles the time)
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

Working hours from the record timestamps (gaps under ten minutes):

| Source | Docs | Time (RTX 4060 Ti 16GB) |
|--------|------|--------------------------|
| PubMed | 22,301 | ~76 hours (first 2,090 with qwen2.5:14b) |
| ClinicalTrials | 19,988 | ~68 hours |
| PubChem | 92 | ~0.4 hours |

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

| Source | Owner | Terms |
|--------|-------|-------|
| PubMed abstracts | Their publishers (served by the US National Library of Medicine) | Abstracts can be under copyright, so the release holds no abstract text, only model-written summaries and extracted fields |
| ClinicalTrials.gov | US federal government | Public domain |
| PubChem | NIH | Public domain |

This dataset (extracted JSON records + embeddings) is released under **CC BY 4.0** - free to use with attribution.

Pipeline code is free software under the GNU Affero General Public License 3.0 (`LICENSE`), from this version on. If you run a changed version as a service for others, you share your changes under the same licence. Copies obtained before this change keep the MIT licence they came with.

Q&A pairs are not released publicly.

---

## Academic context

This work is a contribution to **literature-based discovery (LBD)** and the *undiscovered public knowledge* problem (Swanson, 1986).

**Paper:** Pocatilu, D.-M. (2026). *Open Cancer Knowledge Graph*. pdm.ai. https://www.pdmzerotrustbusiness.com/papers/

**Author:** Daniel-Mihai Pocatilu, pdm.ai, ORCID 0009-0004-9820-6708.

---

## Citation

```bibtex
@dataset{ockg2026,
  title     = {Open Cancer Knowledge Graph (OCKG) v1.0},
  author    = {Pocatilu Daniel-Mihai},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/pdm95/open-cancer-kg}
}
```

---

*Built on a student's GPU. Costs nothing to run. Free for any researcher anywhere.*