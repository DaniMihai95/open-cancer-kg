# OCKG - Open Cancer Knowledge Graph

> Built by a student on a consumer GPU. Costs nothing to run. Free for any researcher anywhere.

**Dataset on Hugging Face →** [pdm95/open-cancer-kg](https://huggingface.co/datasets/pdm95/open-cancer-kg)

---

## What this is

A pipeline that processes cancer research from three free public sources - PubMed, ClinicalTrials.gov, and PubChem - and structures every document into the same schema using a local LLM, then cross-references them by shared compounds, genes, pathways, and cancer types to surface connections that keyword search cannot find.

This is the *undiscovered public knowledge* problem (Swanson, 1986). A compound tested in a 1994 breast cancer paper may share a pathway with a 2021 lung trial that failed for an unrelated reason - and nobody connected them. This pipeline does.

---

## Dataset (v1.0)

| Source | Documents |
|--------|-----------|
| PubMed | 22,301 |
| ClinicalTrials.gov | 19,988 |
| PubChem | 92 |
| **Total** | **42,381** |

Download the structured dataset from Hugging Face - do not run the pipeline yourself unless you want to extend it. The full run takes ~125 hours on an RTX 4060 Ti.

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
| **OCKG** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Requirements

```
Python 3.10+
Ollama (https://ollama.com)
NVIDIA GPU with 8GB+ VRAM recommended
```

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Optional - free NCBI API key for 10x faster PubMed fetching:
- Register at https://www.ncbi.nlm.nih.gov/account/
- Account Settings → API Key Management → Generate
- Use: `NCBI_API_KEY=your_key python pipeline.py ...`

---

## Run the pipeline

```bash
# Test with 100 docs first
python pipeline.py --source pubmed --limit 100 --workers 2

# Full runs - leave overnight, fully resumable if interrupted
python pipeline.py --source pubmed  --limit 50000 --workers 3
python pipeline.py --source trials  --limit 20000 --workers 3
python pipeline.py --source pubchem --limit 10000 --workers 3

# Find cross-source connections
python pipeline.py --crossref

# Statistics
python pipeline.py --stats
```

The pipeline is fully resumable - restart with the same command and it skips already-processed documents.

---

## Query the dataset

```bash
# Semantic search
python query.py --search "KRAS mutation untested compound pancreatic"

# Find all documents mentioning a compound
python query.py --entity compound "sotorasib"

# Find cross-source connections
python query.py --connections "cisplatin"

# Export connections to CSV for Gephi / Cytoscape
python query.py --export-connections connections.csv
```

---

## Output schema

Each record contains:

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
  "embedding": [0.021, -0.034, "..."]
}
```

---

## Performance

| Source | Docs | Time |
|--------|------|------|
| PubMed | 22,338 | ~55 hours |
| ClinicalTrials | 19,988 | ~68 hours |
| PubChem | 92 | ~2 hours |

Measured on NVIDIA RTX 4060 Ti 16GB, power-limited to 125W, qwen2.5:7b, workers=3.

---

## Known limitations

- ~15% of records may have incomplete entity extraction from vague abstracts
- `followed_up` field is an LLM judgment from abstract text alone, not citation-verified
- First 2,090 PubMed records used qwen2.5:14b, remainder qwen2.5:7b
- 2 corrupted records excluded from final dataset

---

## Academic context

This work contributes to literature-based discovery (LBD) and the undiscovered public knowledge problem (Swanson, 1986).

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

## License

- Pipeline code: MIT
- Dataset: CC BY 4.0