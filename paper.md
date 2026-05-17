# Open Cancer Knowledge Graph (OCKG): Surfacing Undiscovered Public Knowledge Across Siloed Biomedical Databases

## Abstract
The exponential growth of biomedical literature and clinical trial data has led to the fragmentation of cancer research across disparate, siloed databases. This fragmentation exacerbates the "undiscovered public knowledge" problem, where implicit connections between distinct research findings remain unidentified because the underlying documents never cite each other. In this paper, we present the Open Cancer Knowledge Graph (OCKG), an open-source, fully automated pipeline that synthesizes unstructured data from PubMed, ClinicalTrials.gov, and PubChem. By employing a localized Large Language Model (Qwen2.5:7b) for consistent structured extraction and semantic embeddings (Nomic-Embed-Text), OCKG normalizes 42,409 documents into a unified schema cross-referenced by shared compounds, genes, pathways, and cancer types. Our methodology successfully surfaced 14,163 implicit cross-source connections, including 104 high-confidence links involving shared compounds, cancer types, and biological pathways simultaneously. OCKG is free, executable on consumer-grade hardware, and designed to bridge critical research gaps in oncology by computationally identifying unpursued therapeutic synergies.

## 1. Introduction
The volume of cancer research published annually far exceeds the reading capacity of any individual researcher. While valuable insights are continuously generated in the form of research papers, clinical trials, and compound profiles, this information is distributed across independent repositories such as PubMed, ClinicalTrials.gov, and PubChem. This siloing of data gives rise to what Swanson (1986) termed the "undiscovered public knowledge" problem. For instance, a compound tested in a historical study may influence a biological pathway implicated in a contemporary clinical trial, yet no formal connection is made because the studies do not directly cite one another.

We introduce the Open Cancer Knowledge Graph (OCKG), a novel, computational approach designed to systematically surface these hidden connections. Unlike existing proprietary or partial systems, OCKG provides a comprehensive, open-access, cancer-focused pipeline that utilizes Large Language Models (LLMs) to structure data uniformly across multiple domains and compute semantic similarities for robust cross-referencing.

## 2. Related Work
Literature-Based Discovery (LBD) has sought to uncover implicit relationships since Swanson's early work on Raynaud's disease and fish oil. Modern iterations have increasingly leveraged artificial intelligence to extract entities and construct knowledge graphs. Systems like Open Targets, SemMedDB, and SPOKE have made significant strides but often lack localized LLM extraction, comprehensive gap detection, or remain entirely proprietary or generalized. Recent advancements, including BioGPT, iKraph, and PubMed KG 2.0 (Xu et al., 2024), offer varying degrees of LLM integration and cross-database querying but do not combine all critical properties: localized extraction, semantic embeddings, cross-database integration, specific research gap detection, open-access availability, and a dedicated focus on oncology. OCKG fills this void by unifying these six properties into a singular, highly accessible pipeline.

## 3. Methodology
The OCKG pipeline is architected to process and unify data through a three-pass system utilizing a local LLM, operating entirely on consumer-grade hardware (e.g., an NVIDIA RTX 4060 Ti with 16GB VRAM).

### 3.1 Data Acquisition
Data is fetched from three primary sources:
- **PubMed:** Literature abstracts and MeSH terms retrieved via specialized, mechanism- and compound-focused queries.
- **ClinicalTrials.gov:** Clinical trial summaries and protocols filtered for cancer conditions.
- **PubChem:** Detailed chemical and structural data for prominent cancer-relevant compounds.

In the current version (v1.0), the corpus comprises 22,338 PubMed documents, 19,979 clinical trials, and 92 chemical compounds, totaling 42,409 documents.

### 3.2 Unified Structured Extraction
Regardless of the source type, each document is processed by a quantized local LLM (Qwen2.5:7b via Ollama) to extract a unified JSON schema. This schema captures cancer types, biological pathways, compounds, genes/proteins, mechanism of action, experimental results, and a specific Boolean flag (`followed_up`) indicating whether the LLM judges the finding as an unfollowed research gap.

### 3.3 Semantic Embedding and Q&A Generation
Following extraction, a rich semantic string concatenating the title, summary, pathways, compounds, and genes is constructed. This string is embedded into a 768-dimensional vector using `nomic-embed-text`. Simultaneously, the LLM generates five specific Question-and-Answer pairs for each document (covering factual, mechanistic, clinical, gap-spotting, and connection aspects) to facilitate future fine-tuning of domain-specific research assistants.

### 3.4 Cross-Referencing and Graph Indexing
Extracted entities are indexed in a SQLite database. A subsequent cross-reference pass computationally identifies "missed connections." It seeks out document pairs (e.g., a paper and a trial) from different sources that share a compound, cancer type, or pathway, but lack direct citation. A confidence score is calculated based on the overlap of these shared entities.

## 4. Results
The pipeline successfully processed 42,409 documents in approximately 125 hours on a consumer GPU. The resultant knowledge graph identified 14,163 cross-source connections. Crucially, the system surfaced 104 high-confidence connections where documents simultaneously share a compound, cancer type, and biological pathway.

An illustrative example of OCKG's utility involves a completed clinical trial (NCT00501410) testing cetuximab and dasatinib for EGFR resistance in colorectal cancer, and a separate publication (PMID 27636997) discovering that combining cetuximab with MEK1/2 inhibition creates a synthetic lethality in NRAS-mutant colorectal cancer. Although dealing with the same cancer, compound, and clinical problem, neither document cited the other. OCKG computationally surfaced this connection, immediately suggesting a novel therapeutic hypothesis: combining all three approaches.

## 5. Discussion
The relevance of OCKG is profound. By democratizing access to complex, cross-domain biomedical data synthesis, it empowers researchers globally without requiring exorbitant computational budgets or proprietary software licenses.

The revolutionary aspect of OCKG lies in its application of open-weight LLMs for unified structuring combined with deep semantic embeddings, all running locally. It actively detects "research gaps"—findings that were never built upon—and explicitly highlights them. This transforms literature review from a passive keyword search into an active, hypothesis-generating process. By mapping the silences and disconnections in the literature, OCKG acts as an automated collaborative partner, bridging decades of disparate research. Furthermore, by making cross-referencing agnostic to vocabulary differences, the embeddings bypass the limitations of traditional keyword indexing.

## 6. Conclusion
The Open Cancer Knowledge Graph addresses the critical bottleneck of undiscovered public knowledge in oncology. By leveraging localized LLMs to enforce a unified schema across PubMed, ClinicalTrials.gov, and PubChem, OCKG has proven capable of surfacing thousands of implicit connections and unpursued research gaps. As an open, free, and computationally accessible tool, OCKG represents a significant step forward in literature-based discovery, offering a revolutionary resource for cancer researchers seeking to uncover novel therapeutic synergies.

## References
1. Swanson, D. R. (1986). Fish oil, Raynaud's syndrome, and undiscovered public knowledge. *Perspectives in Biology and Medicine*, 30(1), 7-18.
2. Xu, et al. (2024). PubMed KG 2.0.
3. Mihai, P. D. (2026). Open Cancer Knowledge Graph (OCKG) v1.0. *Hugging Face*.
