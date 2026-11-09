"""
Recompute every number in the OCKG paper from the published dataset files.

Download kg_pubmed.jsonl, kg_trials.jsonl and kg_pubchem.jsonl from
https://huggingface.co/datasets/pdm95/open-cancer-kg into one folder, then:

    python analysis/paper_numbers.py /path/to/that/folder

Needs numpy. Runs in a few minutes and about 6 GB of memory. The random
samples use a fixed seed, so the output is the same on every run.
"""

import collections
import datetime
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

FILES = ["kg_pubmed.jsonl", "kg_trials.jsonl", "kg_pubchem.jsonl"]
ENTITY_FIELDS = ["cancer_types", "pathways_mentioned", "compounds", "genes_proteins"]
CANCER_WORD = re.compile(
    r"cancer|carcinom|tumou?r|leuk|lymphom|sarcom|melanom|gliom|neoplas|oncolog|"
    r"myelom|malignan|metasta|blastom|mesotheliom|adenom",
    re.I,
)
# The first 2,090 PubMed records were extracted with qwen2.5:14b, the rest with qwen2.5:7b.
FIRST_MODEL_RECORDS = 2090


def entity_set(record, field):
    """Entity strings as pipeline.py indexes them: lower case, stripped, empty ones dropped."""
    out = set()
    for x in record.get(field) or []:
        if isinstance(x, dict):
            x = x.get("name", "") or x.get("term", "") or str(x)
        x = str(x).strip() if x else ""
        if x:
            out.add(x.lower())
    return out


def pct(a, b):
    return f"{a:,} of {b:,} ({100 * a / b:.1f}%)"


def main(folder):
    recs, unreadable, ids = [], 0, collections.Counter()
    for name in FILES:
        for line in open(Path(folder) / name, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                unreadable += 1
                continue
            ids[r["doc_id"]] += 1
            recs.append(r)
    by_source = collections.defaultdict(list)
    for r in recs:
        by_source[r["source"]].append(r)

    print("== Records")
    print("readable", len(recs), "unreadable lines", unreadable,
          "distinct ids", len(ids), "ids seen twice", sum(1 for n in ids.values() if n > 1))
    for s, rs in by_source.items():
        ts = sorted(datetime.datetime.fromisoformat(r["processed_at"]) for r in rs)
        gaps = [(ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1)]
        active = sum(g for g in gaps if g < 600) / 3600
        print(f"{s}: {len(rs):,} records, processed {ts[0]:%Y-%m-%d} to {ts[-1]:%Y-%m-%d}, "
              f"active hours {active:.1f}")

    print("\n== Extraction completeness (records with the field empty)")
    for s, rs in by_source.items():
        n = len(rs)
        row = {f: pct(sum(1 for r in rs if not entity_set(r, f)), n) for f in ENTITY_FIELDS}
        row["all four"] = pct(sum(1 for r in rs if not any(entity_set(r, f) for f in ENTITY_FIELDS)), n)
        print(s, row)
    total_empty = sum(1 for r in recs if not any(entity_set(r, f) for f in ENTITY_FIELDS))
    print("all sources, all four empty:", pct(total_empty, len(recs)))

    print("\n== Trials and PubMed records that mention cancer in title, summary or cancer types")
    for s in ["trials", "pubmed"]:
        rs = sorted(by_source[s], key=lambda r: r["processed_at"])
        hits = [bool(CANCER_WORD.search(" ".join([
            r.get("raw_title") or "", r.get("title") or "", r.get("summary") or "",
            " ".join(map(str, r.get("cancer_types") or []))]))) for r in rs]
        print(s, pct(sum(hits), len(hits)), "first 100 processed:", sum(hits[:100]),
              "first 1,000:", sum(hits[:1000]))

    print("\n== PubMed by extraction model (records in processing order)")
    pm = sorted(by_source["pubmed"], key=lambda r: r["processed_at"])
    for label, rs in [("qwen2.5:14b", pm[:FIRST_MODEL_RECORDS]), ("qwen2.5:7b", pm[FIRST_MODEL_RECORDS:])]:
        n = len(rs)
        means = {f: round(sum(len(entity_set(r, f)) for r in rs) / n, 2) for f in ENTITY_FIELDS}
        empty = sum(1 for r in rs if not any(entity_set(r, f) for f in ENTITY_FIELDS))
        print(label, n, "all four empty", pct(empty, n), "mean entities", means)

    print("\n== Gap flag, outcome and hypotheses")
    for s, rs in by_source.items():
        fu = collections.Counter()
        oc = collections.Counter()
        for r in rs:
            er = r.get("experimental_result") if isinstance(r.get("experimental_result"), dict) else {}
            fu[repr(er.get("followed_up"))] += 1
            oc[str(er.get("outcome")).lower()] += 1
        print(s, "followed_up", dict(fu), "outcome", oc.most_common(6))
    with_h = sum(1 for r in recs if r.get("potential_connections"))
    print("records with at least one potential_connections entry:", pct(with_h, len(recs)),
          "entries", sum(len(r.get("potential_connections") or []) for r in recs))
    quality = collections.Counter(r.get("data_quality") for r in recs)
    print("data_quality", quality.most_common())

    print("\n== Vocabulary")
    for f in ENTITY_FIELDS:
        c = collections.Counter(x for r in recs for x in entity_set(r, f))
        print(f, "distinct", len(c), "seen once", sum(1 for v in c.values() if v == 1),
              "top", c.most_common(8))
    breast = collections.Counter(x for r in recs for x in entity_set(r, "cancer_types") if "breast" in x)
    print("distinct cancer_types strings containing 'breast':", len(breast))

    print("\n== Cross-source pairs (the scoring rule of pipeline.py run_crossref)")
    seen = {}
    for r in recs:
        seen.setdefault(r["doc_id"], r)
    docs = {d: dict(s=r["source"], c=entity_set(r, "compounds"), k=entity_set(r, "cancer_types"),
                    p=entity_set(r, "pathways_mentioned"), e=r["embedding"]) for d, r in seen.items()}
    by_compound = collections.defaultdict(list)
    for d, x in docs.items():
        for c in x["c"]:
            by_compound[c].append(d)
    pairs = {}
    for ds in by_compound.values():
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                a, b = ds[i], ds[j]
                if docs[a]["s"] == docs[b]["s"]:
                    continue
                key = tuple(sorted((a, b)))
                if key not in pairs:
                    pairs[key] = (len(docs[a]["k"] & docs[b]["k"]), len(docs[a]["p"] & docs[b]["p"]))
    print("distinct cross-source document pairs sharing a compound:", f"{len(pairs):,}")

    def group(sc, sp):
        if sc and sp:
            return "compound, cancer type and pathway"
        if sc:
            return "compound and cancer type"
        if sp:
            return "compound and pathway"
        return "compound only"

    def cos(a, b):
        x = np.asarray(docs[a]["e"], dtype=np.float32)
        y = np.asarray(docs[b]["e"], dtype=np.float32)
        return float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))

    rng = random.Random(0)
    groups = collections.defaultdict(list)
    for key, (sc, sp) in pairs.items():
        groups[group(sc, sp)].append(key)
    for g in ["compound only", "compound and cancer type", "compound and pathway",
              "compound, cancer type and pathway"]:
        keys = groups[g]
        sample = rng.sample(keys, min(3000, len(keys)))
        cs = [cos(a, b) for a, b in sample]
        print(f"{g}: {len(keys):,} pairs, cosine over {len(sample):,}: mean {np.mean(cs):.3f} "
              f"10th {np.percentile(cs, 10):.3f} 90th {np.percentile(cs, 90):.3f}")
    rng = random.Random(0)
    pmids = [d for d, x in docs.items() if x["s"] == "pubmed"]
    nct = [d for d, x in docs.items() if x["s"] == "trials"]
    rnd = [cos(rng.choice(pmids), rng.choice(nct)) for _ in range(5000)]
    print(f"random PubMed and trial pairs: 5,000, cosine mean {np.mean(rnd):.3f} "
          f"10th {np.percentile(rnd, 10):.3f} 90th {np.percentile(rnd, 90):.3f}")
    top = collections.Counter()
    for (a, b), (sc, sp) in pairs.items():
        if sc and sp:
            top.update(docs[a]["c"] & docs[b]["c"])
    print("compounds in the pairs that share all three:", top.most_common(10))
    kept = sum(1 for sc, sp in pairs.values() if sc or sp)
    print("pairs at or above the 0.5 threshold:", f"{kept:,}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
