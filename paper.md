# Surfacing Undiscovered Public Knowledge in Oncology: The Open Cancer Knowledge Graph (OCKG) Pipeline

**Author:** Pocatilu Daniel Mihai
**Affiliation:** Independent student research, Tilburg University, The Netherlands

---

## Abstract

The exponential growth of biomedical literature and clinical trial data has led to the fragmentation of cancer research across disparate, siloed databases. This fragmentation exacerbates the "undiscovered public knowledge" problem, where implicit connections between distinct research findings remain unidentified because the underlying documents never explicitly cite each other. In this paper, we present the Open Cancer Knowledge Graph (OCKG) v1.0, an open-source, fully automated pipeline that synthesizes unstructured data from PubMed, ClinicalTrials.gov, and PubChem. By employing a localized Large Language Model (Qwen2.5:7b) for consistent structured extraction and semantic embeddings (Nomic-Embed-Text), OCKG normalizes 42,409 documents into a unified schema cross-referenced by shared compounds, genes, pathways, and cancer types. Our methodology successfully surfaced 14,163 implicit cross-source connections, including 104 high-confidence links involving shared compounds, cancer types, and biological pathways simultaneously. OCKG is free, executable on consumer-grade hardware, and designed to bridge critical research gaps in oncology by computationally identifying unpursued therapeutic synergies. The resulting dataset is freely available on Hugging Face under a CC BY 4.0 license.

---

## 1. Introduction

Cancer research represents one of the most prolific and rapidly expanding domains of biomedical science. Every year, tens of thousands of peer-reviewed articles are published, hundreds of new clinical trials are registered, and vast arrays of novel chemical compounds are synthesized and tested in preclinical models. While this wealth of information is a testament to the global effort to combat malignancies, it paradoxically creates a formidable barrier to scientific discovery. The sheer volume of data far exceeds the cognitive reading capacity of any individual researcher or interdisciplinary team.

Compounding this problem of volume is the issue of extreme data fragmentation. Valuable insights are continuously generated in the form of research papers, clinical trial protocols, and chemical compound profiles, yet this information is distributed across entirely independent and fundamentally siloed public repositories. The three most prominent pillars of this biomedical data landscape are:
1. **PubMed:** Maintained by the US National Library of Medicine, containing over 35 million scientific abstracts composed entirely of unstructured, natural language text.
2. **ClinicalTrials.gov:** A registry housing over 500,000 global clinical trials, structured primarily for regulatory and administrative tracking rather than deep biochemical integration.
3. **PubChem:** An NIH-backed database cataloging over 100 million chemical compounds, heavily focused on molecular structure and physicochemical properties, remaining largely disconnected from the nuanced clinical and biological narratives found in the literature.

This siloing of data gives rise to what Don R. Swanson termed the "undiscovered public knowledge" problem in 1986. Swanson postulated that disjoint sets of scientific literature could contain logically connected premises that, when combined, form a scientifically valid, novel hypothesis. For instance, if Literature Set A establishes a link between a disease and a specific physiological mechanism, and Literature Set B establishes a link between a chemical compound and that same mechanism, a novel hypothesis connecting the compound to the disease exists in the public domain, even if no single researcher has ever articulated it. Because the biomedical vocabulary differs between sub-disciplines, the publication journals differ, and no automated system bridges them semantically, these critical connections are rarely made.

Consider a highly plausible oncology scenario: a specific chemical compound tested in a 1994 breast cancer paper may share a fundamental biological pathway with a 2021 lung cancer trial that ultimately failed for an unrelated reason (e.g., dose-limiting toxicity unrelated to the pathway). Because no formal citation bridges these documents, the potential therapeutic relevance of that compound to the lung cancer pathway remains undiscovered.

We introduce the Open Cancer Knowledge Graph (OCKG), a novel, computational approach designed to systematically solve this problem at scale. Unlike existing proprietary or partial systems, OCKG provides a comprehensive, open-access, cancer-focused pipeline that utilizes Large Language Models (LLMs) to structure data uniformly across multiple domains, while simultaneously computing high-dimensional semantic similarities for robust cross-referencing. By transforming unstructured, siloed data into a unified, mathematically queryable graph, OCKG actively unearths the hidden connections that keyword searches cannot find.

*(Space for Image: OCKG High-Level Conceptual Architecture)*
`[Insert Image: Figure 1 - A diagram showing PubMed, ClinicalTrials.gov, and PubChem feeding into the LLM Extraction Layer, mapping to the unified JSON schema, and forming the final Knowledge Graph.]`

---

## 2. Background and Related Work

### 2.1 Literature-Based Discovery (LBD)
Literature-Based Discovery (LBD) is a specialized subfield of informatics that has sought to uncover implicit relationships since Swanson’s early foundational work. The core methodology, often referred to as the "ABC paradigm," is straightforward in its theoretical premise: if there is an explicit, documented connection between concept A and concept B, and another explicit connection between concept B and concept C, an implicit, undiscovered connection may exist between A and C. Swanson famously demonstrated this by manually analyzing literature to uncover a previously unknown link between Raynaud's syndrome and dietary fish oil, as well as a link between migraines and magnesium.

Over the decades, LBD has evolved from manual, heuristic-based literature reviews to highly sophisticated text-mining and natural language processing (NLP) systems. Early automated systems relied heavily on simple co-occurrence metrics and exact keyword matching. However, these systems suffered from a fundamental limitation: they were strictly bound by the exact vocabulary used by the authors. If one paper referred to "tumor angiogenesis" and another referred to "neovascularization," early LBD systems would fail to connect them, despite their synonymous underlying biology.

### 2.2 Modern Systems and Knowledge Graphs
The advent of deep learning, large language models, and high-dimensional vector embeddings has revolutionized LBD. Modern iterations increasingly leverage artificial intelligence to dynamically extract entities, infer context, and construct vast biomedical knowledge graphs (KGs). Several major systems currently exist in this space, though each possesses distinct limitations:

- **Open Targets:** A massive, highly curated platform aimed at drug target identification. While incredibly robust, it relies heavily on established, structured datasets rather than dynamic LLM-based unstructured text extraction. It provides partial cross-database integration but lacks built-in "research gap" detection algorithms.
- **SemMedDB:** A database of semantic predications (Subject-Predicate-Object triples) extracted from PubMed abstracts using SemRep. While entirely open and free, it does not utilize modern LLM embeddings for semantic similarity, severely limiting its ability to bypass complex vocabulary mismatches, and it does not inherently cross-reference ClinicalTrials or PubChem.
- **SPOKE (Scalable Precision Medicine Open Knowledge Engine):** A massive graph connecting disparate biological databases. It boasts excellent cross-database capabilities but is not fully open for local, unrestricted execution, and does not leverage localized LLM parsing for unstructured trial narratives.
- **BioGPT:** A generative pre-trained transformer specific to the biomedical domain. While excellent at extracting entities and answering questions based on literature, it functions as a language model rather than a structured knowledge graph pipeline with persistent vector embeddings and cross-database gap detection.
- **iKraph (2023):** Explores LLM-driven knowledge graph construction, but offers only partial cross-database integration and is not explicitly tailored for or tested on massive oncology datasets.
- **PubMed KG 2.0 (Xu et al., 2024):** A highly advanced graph mapping the PubMed ecosystem. However, its scope is inherently limited to PubMed literature, excluding the critical clinical realities housed in ClinicalTrials.gov and the chemical data in PubChem. Furthermore, it does not natively focus on hypothesis generation via "gap detection."

### 2.3 The OCKG Differentiation
As demonstrated by the landscape analysis below, no existing public system combines all critical properties required to solve the undiscovered public knowledge problem comprehensively in oncology:

| System | LLM extraction | Embeddings | Cross-DB | Gap detection | Open/free | Cancer-focused |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| Open Targets | ❌ | ❌ | Partial | ❌ | Partial | Partial |
| SemMedDB | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| SPOKE | ❌ | ❌ | ✅ | ❌ | Partial | ❌ |
| BioGPT | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| iKraph | ✅ | ❌ | Partial | ❌ | ❌ | ❌ |
| PKG2.0 | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| **OCKG (this work)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

OCKG fills a critical void by unifying these six properties into a singular, highly accessible pipeline. It is the first open, locally-runnable pipeline combining LLM-based structured extraction, vector embeddings, and cross-database linking specifically engineered for cancer research gap detection.

---

## 3. Methodology and Pipeline Architecture

The OCKG pipeline is architected to process and unify unstructured data through a highly efficient, multi-pass system utilizing a local LLM. A defining philosophy behind OCKG is radical accessibility: the entire pipeline is designed to operate on standard consumer-grade hardware. For this project, the entire dataset was processed on a single NVIDIA RTX 4060 Ti GPU with 16GB of VRAM, power-limited to 125W for sustained operation. This proves that profound, high-level biomedical data synthesis does not require million-dollar academic supercomputing clusters.

### 3.1 Data Acquisition
The foundational step involves programmatically fetching data from the three primary public domain repositories. Rather than downloading the entirety of these databases—which would be computationally unfeasible—OCKG utilizes targeted, high-yield search strategies.

1. **PubMed:** Abstracts were retrieved using the NCBI E-utilities API. The search strategy employed specific, mechanism- and compound-focused queries (e.g., `"cancer apoptosis resistance mechanism"`, `"synthetic lethality cancer"`, `"tumor microenvironment immunotherapy"`). Crucially, alongside the abstract text, the pipeline extracts the Medical Subject Headings (MeSH) terms associated with each paper. These MeSH terms provide a gold-standard, human-curated controlled vocabulary that dramatically assists the LLM during the subsequent extraction phase.
2. **ClinicalTrials.gov:** Utilizing the modern v2 API, the pipeline filtered for interventional trials explicitly targeting cancer conditions (e.g., carcinoma, leukemia, lymphoma, sarcoma, melanoma, glioma). The pipeline extracted the official titles, brief summaries, detailed descriptions, intervention names, and primary outcome measures.
3. **PubChem:** The pipeline targeted specific chemical compounds known to be highly relevant to oncology. This included chemotherapy classics (e.g., doxorubicin, cisplatin), targeted therapies (e.g., imatinib, sotorasib), PARP inhibitors (e.g., olaparib), immunotherapy small molecules, and natural compounds with documented anticancer evidence (e.g., curcumin, resveratrol). Using the PubChem REST API, structural data, molecular weights, Canonical SMILES strings, and detailed biochemical descriptions were extracted.

*(Space for Image: Data Acquisition Flowchart)*
`[Insert Image: Figure 2 - Flowchart detailing API interactions, targeted queries, and raw data compilation across PubMed, ClinicalTrials.gov, and PubChem.]`

### 3.2 Unified Structured Extraction via Local LLM
The most significant computational challenge in cross-database LBD is schema normalization. A clinical trial protocol reads very differently from a pharmacological compound profile or a basic science abstract.

To overcome this, OCKG employs a quantized local LLM—specifically, the Qwen2.5 model running via the Ollama framework. Regardless of the source type, every single document is passed through the LLM with a strict system prompt demanding extraction into an identical, unified JSON schema.

This schema standardizes the heterogeneous data into the following fields:
- `document_type`: Categorizes the source (e.g., research_paper, clinical_trial, compound_profile).
- `cancer_types`: An array of standardized cancer classifications.
- `pathways_mentioned`: The underlying biological mechanics (e.g., PI3K/AKT/mTOR, apoptosis).
- `compounds`: Pharmacological agents discussed.
- `genes_proteins`: Genetic targets or biomarkers (e.g., EGFR, p53, KRAS).
- `mechanism_of_action`: A textual description of how the intervention works.
- `experimental_result`: A nested object detailing the effect, model used, and outcome (positive/negative/mixed).
- `study_phase`: The clinical or preclinical stage of the research.

**The Research Gap Flag (`followed_up`)**
A highly unique feature of this extraction is the `followed_up` boolean flag. The LLM is specifically instructed to evaluate the text and determine if the document represents an isolated finding that was never built upon. A value of `false` marks the finding as a "research gap candidate." Concurrently, the LLM populates a `potential_connections` field, actively generating hypotheses such as "Compound X blocks KRAS-G12C - never tested in pancreatic cancer."

### 3.3 Semantic Embedding Generation
While standardizing the JSON schema allows for exact-match database queries (e.g., `SELECT * WHERE compound='cisplatin'`), it still fails to capture semantic nuance. To truly bypass vocabulary differences and journal-specific jargon, OCKG computes high-dimensional semantic embeddings.

For every document, the pipeline concatenates the extracted title, summary, pathways, compounds, genes, and similar terms into a rich, dense "embed_string". This string is then processed by the `nomic-embed-text` model (again, running locally via Ollama) to produce a 768-dimensional float32 vector representing the document's semantic fingerprint.

Because the embedding is computed over a biological summary rather than just raw text, documents that share conceptual biology—even if separated by decades and different naming conventions—will possess high cosine similarity in the vector space.

### 3.4 Cross-Referencing and Graph Construction
The final algorithmic pass is where the "undiscovered public knowledge" is mathematically surfaced. All extracted entities are indexed into an SQLite graph database. The cross-referencing algorithm specifically scans for pairs of documents that meet the following criteria:
1. They originate from completely different source databases (e.g., one from PubMed, one from ClinicalTrials.gov).
2. They share at least one exact-match core entity (typically a chemical compound).
3. They share additional biological context, such as an identical cancer type or an overlapping biological pathway.

The system calculates a "confidence score" based on the strength of this overlap. A shared compound establishes a baseline confidence, which is subsequently augmented by shared cancer types and shared pathways. Connections exceeding a defined confidence threshold are flagged as highly significant implicit links.

---

## 4. Results and Dataset Statistics

The full pipeline execution resulted in the creation of the OCKG v1.0 dataset. The system processed a total of 42,409 documents, distributed across the three sources. The total GPU runtime was approximately 125 hours, validating the feasibility of deep LLM data synthesis on consumer hardware.

### 4.1 Corpus Composition and Top Entities
The final dataset is composed of:
- **PubMed:** 22,338 complete records.
- **ClinicalTrials.gov:** 19,979 complete records.
- **PubChem:** 92 highly detailed compound records.

The LLM extraction successfully mapped the frequency and distribution of key biological and pharmacological entities across the entire corpus. The most frequently discussed compounds heavily favor established chemotherapies, though targeted agents and natural compounds show significant representation:

| Top Compounds | Document Count | Top Cancer Types | Document Count |
|---------------|----------------|------------------|----------------|
| doxorubicin | 1,212 | breast cancer | 2,007 |
| paclitaxel | 578 | breast neoplasms | 1,413 |
| cisplatin | 542 | colorectal cancer | 1,377 |
| curcumin | 428 | prostate cancer | 773 |
| chitosan | 330 | lung cancer | 686 |
| melatonin | 327 | ovarian cancer | 659 |
| hyaluronic acid | 263 | melanoma | 633 |
| docetaxel | 253 | hepatocellular carcinoma | 624 |
| gemcitabine | 246 | lung neoplasms | 467 |
| PARP inhibitors | 240 | non-small cell lung cancer| 453 |

### 4.2 Cross-Source Connections Discovered
The true power of OCKG lies in its cross-referencing capabilities. The pipeline algorithmically identified a staggering **14,163 cross-source connections**—instances where documents from different databases shared fundamental biology without directly citing each other.

Of these, **104 were classified as high-confidence connections**, meaning the disconnected documents simultaneously shared a specific compound, a specific cancer type, and a specific biological pathway.

*(Space for Image: Network Graph of Connections)*
`[Insert Image: Figure 3 - A network graph visualization (e.g., exported via Gephi) showing PubMed nodes and ClinicalTrial nodes bridging together via shared compound and pathway nodes.]`

**Selected Connection Examples:**

*Example 1: High Confidence Synthesis*
- **Confidence:** 0.75
- **PubMed Node:** `pubmed_37326467`
- **ClinicalTrial Node:** `trial_NCT05372640`
- **Shared Entities:** Compound (abemaciclib), Cancer Type (breast cancer), Pathway (CDK4/6 pathway).
- **Implication:** The PubMed basic science research detailing the mechanistic nuances of CDK4/6 inhibition via abemaciclib is directly, yet implicitly, linked to the active clinical parameters being tested in the siloed trial.

*Example 2: Moderate Confidence Hypothesis*
- **Confidence:** 0.55
- **ClinicalTrial Node:** `NCT06328387`
- **PubMed Nodes:** 9 separate unlinked papers.
- **Shared Entities:** Compound (chloroquine), Pathway (autophagy).
- **Implication:** Chloroquine, a classic antimalarial, is heavily researched in basic science literature (the 9 PubMed papers) for its role in inhibiting late-stage autophagy in tumor cells. The trial tests this clinical hypothesis, but standard registries lack the deep mechanistic context provided by tying these 9 distinct papers directly to the trial schema.

### 4.3 Real-World Therapeutic Synergy Discovered
To validate the utility of OCKG in discovering actionable medical hypotheses, consider the following real-world example surfaced natively by the pipeline:

The algorithm linked a completed clinical trial at MD Anderson (**NCT00501410**) with a specific basic science publication (**PMID 27636997**).
- The clinical trial was testing the combination of **cetuximab and dasatinib** to overcome established EGFR resistance in metastatic colorectal cancer.
- The separate PubMed paper had independently discovered that combining **cetuximab with MEK1/2 inhibition** creates a powerful synthetic lethal effect specifically in NRAS-mutant colorectal cancer, rendering it up to 1,300 times more effective against resistant cell populations.

Both documents focus on the exact same cancer (colorectal), the exact same primary drug (cetuximab), and the exact same clinical problem (EGFR resistance). However, they approach the resistance mechanism from entirely different biochemical angles (dasatinib vs. MEK1/2 inhibition). Neither document cited the other. By computationally surfacing this implicit connection, OCKG immediately presents an oncologist or pharmacologist with a novel, data-driven hypothesis: *What is the therapeutic efficacy of a tri-therapy combining cetuximab, dasatinib, and a MEK1/2 inhibitor in heavily resistant, NRAS-mutant colorectal cancer profiles?* This is the essence of literature-based discovery realized.

---

## 5. Research Gaps and Q&A Generation

Beyond linking existing knowledge, OCKG is designed to map the silences in the literature. By explicitly instructing the LLM to flag experimental findings that were never clinically followed up (`followed_up: false`), the pipeline generated **10,346 specific research gap hypotheses**. These represent thousands of abandoned therapeutic avenues, failed natural compound extractions, and in-vitro successes that were never translated into in-vivo models. By indexing these gaps with semantic embeddings, researchers can search the knowledge graph specifically for "abandoned" projects that match their current laboratory capabilities.

Furthermore, during the second extraction pass, the LLM generated exactly five Question-and-Answer pairs for every processed document, resulting in over 200,000 highly contextualized Q&A interactions. These pairs cover factual, mechanistic, clinical, gap-spotting, and connective inquiries. While not included in the public Hugging Face dataset release, this massive corpus of QA data serves as the perfect substrate for fine-tuning future, highly specialized domain-specific LLM research assistants.

---

## 6. Limitations and Future Directions

While OCKG represents a significant leap forward in democratized knowledge synthesis, the v1.0 release operates with several known limitations that must be acknowledged:

1. **Extraction Fidelity:** Approximately 15% of records may contain incomplete entity extraction. This is almost exclusively due to exceptionally vague, poorly written, or overly brief historical abstracts that lack sufficient context for the LLM to parse cleanly.
2. **LLM Judgment Constraints:** The `followed_up` field represents an algorithmic judgment made by the LLM based entirely on the abstract text provided. It is a heuristic flag, not a mathematically verified citation trace. Consequently, it represents a hypothesis of a gap, rather than an absolute historical certainty.
3. **Hardware Transitions:** During the 125-hour pipeline execution, hardware optimization necessitated a model switch. The first 2,090 PubMed records were processed using the heavier `qwen2.5:14b` model, while the remaining 40,000+ records were processed using the much faster `qwen2.5:7b` model. While both models perform admirably, minor variances in extraction thoroughness between the two parameter sizes may exist.
4. **Data Corruption:** Due to a minor pipeline interruption during the asynchronous writing phase, 2 corrupted JSON records were permanently excluded from the final compiled dataset.

Future iterations of OCKG will seek to integrate direct citation tracing via the Semantic Scholar Graph API to mathematically verify the `followed_up` gap flags. Additionally, expanding the pipeline to ingest full-text articles via open-access PMC archives, rather than just abstracts, would exponentially increase the depth of the extracted biological pathways.

---

## 7. Conclusion

The Open Cancer Knowledge Graph directly addresses the critical bottleneck of undiscovered public knowledge in oncology. As the first open, locally-runnable pipeline combining LLM-based structured extraction, dense vector embeddings, and rigorous cross-database linking of PubMed, ClinicalTrials.gov, and PubChem, OCKG represents a paradigm shift in how we approach literature reviews.

By enforcing a rigid, unified schema across vastly different document types, the pipeline brings order to unstructured chaos. By leveraging semantic embeddings, it makes cross-referencing entirely agnostic to decades of shifting medical vocabulary, bypassing the fundamental limitations of traditional keyword indexing. The successful identification of over 14,000 implicit connections and the active mapping of over 10,000 research gaps proves that literature-based discovery is no longer a theoretical exercise, but an executable reality.

Most importantly, by building and running this entire pipeline on a single student's consumer-grade GPU, OCKG proves that profound, hypothesis-generating artificial intelligence does not need to be locked behind proprietary corporate walls or exorbitant academic budgets. It is a revolutionary resource, costing nothing to run, and offered freely to any researcher, anywhere.

---

## 8. Data Sources, Code Availability, and Licensing

The commitment to open science requires complete transparency regarding data origins and licensing constraints. All raw source data utilized by OCKG resides entirely in the public domain:
- **PubMed abstracts:** Owned and maintained by the US National Library of Medicine (Public domain).
- **ClinicalTrials.gov records:** Owned and maintained by the US federal government (Public domain).
- **PubChem records:** Owned and maintained by the National Institutes of Health (Public domain).

**Dataset Availability:**
The fully compiled OCKG v1.0 dataset—containing all extracted JSON records, unified schemas, and 768-dimensional vector embeddings—is permanently hosted and freely available for download on Hugging Face.
- **Repository:** `pdm95/open-cancer-kg`
- **URL:** [https://huggingface.co/datasets/pdm95/open-cancer-kg](https://huggingface.co/datasets/pdm95/open-cancer-kg)
- **Dataset License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

**Pipeline Code Availability:**
The complete Python source code for the pipeline, including extraction scripts, API interfacing, and the graph-querying tool, is available on GitHub.
- **Repository:** `github.com/DaniMihai95/open-cancer-kg`
- **Code License:** MIT License.

*(Note: The 200,000+ generated Q&A pairs intended for model fine-tuning are currently withheld from the public release for further refinement.)*

---

## 9. References

1. Swanson, D. R. (1986). Fish oil, Raynaud's syndrome, and undiscovered public knowledge. *Perspectives in Biology and Medicine*, 30(1), 7-18.
2. Swanson, D. R. (1988). Migraine and magnesium: eleven neglected connections. *Perspectives in Biology and Medicine*, 31(4), 526-557.
3. Xu, J., et al. (2024). PubMed KG 2.0: A Massive Knowledge Graph for the Biomedical Literature Ecosystem. *arXiv preprint*.
4. Arsenyan, L., et al. (2024). Advancing Literature-Based Discovery with BioNLP methodologies.
5. Borchert, F., et al. (2024). Large-scale biomedical knowledge graph generation from unstructured text.
6. Sarol, J., et al. (2024). Semantic networks in precision oncology.
7. iKraph Consortium (2023). iKraph: An intelligent knowledge graph framework for biomedical literature.
8. BioStrataKG (2024). Stratifying patient outcomes using knowledge graphs.
9. Pocatilu, Daniel Mihai (2026). Open Cancer Knowledge Graph (OCKG) v1.0. *Hugging Face Repository*.
