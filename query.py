"""
Cancer Knowledge Graph - Query Tool
====================================
Run AFTER the pipeline has built your dataset.

Usage examples
--------------
  # Find all docs mentioning KRAS + apoptosis
  python query.py --entity compound --name "KRAS inhibitor"

  # Find cross-source connections for a compound
  python query.py --connections --compound "sotorasib"

  # Semantic search: find papers similar to a query
  python query.py --search "KRAS mutation pancreatic cancer untested compound"

  # Export connections to CSV for visualization
  python query.py --export-connections connections.csv

  # Show what's untested (potential_connections field)
  python query.py --gaps
"""

import argparse
import json
import math
import sqlite3
from pathlib import Path

DB_PATH    = Path("cancer_pipeline.db")
OUTPUT_DIR = Path("cancer_output")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x*x for x in a))
    nb   = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def load_all_records() -> list[dict]:
    """Load all NDJSON records from output dir."""
    records = []
    for path in OUTPUT_DIR.glob("*.ndjson"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def search_by_entity(entity_type: str, name: str):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT doc_id, source FROM entity_index
        WHERE entity_type=? AND entity_name LIKE ?
        ORDER BY source
    """, (entity_type, f"%{name.lower()}%")).fetchall()
    conn.close()

    print(f"\nDocuments mentioning {entity_type}: '{name}'")
    print(f"Found: {len(rows)}\n")
    for doc_id, source in rows:
        print(f"  [{source:10}] {doc_id}")


def show_connections(compound: str):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT doc_id_a, source_a, doc_id_b, source_b, shared_keys, confidence
        FROM crossref
        WHERE shared_keys LIKE ?
        ORDER BY confidence DESC
        LIMIT 50
    """, (f"%{compound.lower()}%",)).fetchall()
    conn.close()

    print(f"\nCross-source connections involving: '{compound}'")
    print(f"Found: {len(rows)}\n")
    for row in rows:
        keys = json.loads(row[4])
        print(f"  Confidence: {row[5]:.2f}")
        print(f"    {row[1]:12} → {row[0]}")
        print(f"    {row[3]:12} → {row[2]}")
        if keys.get("cancer_types"):
            print(f"    Shared cancers:  {', '.join(keys['cancer_types'][:3])}")
        if keys.get("pathways"):
            print(f"    Shared pathways: {', '.join(keys['pathways'][:3])}")
        print()


def semantic_search(query: str, top_k: int = 10):
    """
    Find records most similar to a query string using embedding cosine similarity.
    Requires Ollama to be running for query embedding.
    """
    try:
        import requests
        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": query},
            timeout=30,
        )
        query_vec = r.json()["embedding"]
    except Exception as e:
        print(f"Could not embed query (is Ollama running?): {e}")
        return

    print(f"\nSemantic search: '{query}'")
    print("Loading records...")

    records = load_all_records()
    scored  = []
    for rec in records:
        emb = rec.get("embedding", [])
        if emb:
            sim = cosine_similarity(query_vec, emb)
            scored.append((sim, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    print(f"\nTop {top_k} results:\n")
    for sim, rec in scored[:top_k]:
        print(f"  [{sim:.3f}] [{rec['source']:10}] {rec.get('title','')[:80]}")
        summary = rec.get("summary", "")
        if summary:
            print(f"           {summary[:120]}...")
        cancers = rec.get("cancer_types", [])
        if cancers:
            print(f"           Cancers: {', '.join(cancers[:3])}")
        print()


def show_gaps():
    """Show records where followed_up=False - potential research gaps."""
    records = load_all_records()
    gaps    = []
    for rec in records:
        er = rec.get("experimental_result", {})
        if er.get("followed_up") is False:
            gaps.append(rec)

    print(f"\nPotential research gaps (findings not followed up): {len(gaps)}\n")
    for rec in gaps[:50]:
        print(f"  [{rec['source']:10}] {rec.get('title','')[:70]}")
        er = rec.get("experimental_result", {})
        print(f"           Effect: {er.get('effect','')[:80]}")
        conns = rec.get("potential_connections", [])
        for c in conns[:2]:
            print(f"           ► {c[:100]}")
        print()


def export_connections(out_path: str):
    import csv
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT doc_id_a, source_a, doc_id_b, source_b, shared_keys, confidence
        FROM crossref ORDER BY confidence DESC
    """).fetchall()
    conn.close()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id_a","source_a","doc_id_b","source_b",
                    "compound","cancer_types","pathways","confidence"])
        for row in rows:
            keys = json.loads(row[4])
            w.writerow([
                row[0], row[1], row[2], row[3],
                keys.get("compound",""),
                "|".join(keys.get("cancer_types",[])),
                "|".join(keys.get("pathways",[])),
                row[5],
            ])
    print(f"Exported {len(rows)} connections → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cancer Knowledge Graph Query Tool")
    parser.add_argument("--entity",   nargs=2, metavar=("TYPE","NAME"),
                        help="Search by entity type and name (e.g. --entity compound KRAS)")
    parser.add_argument("--connections", metavar="COMPOUND",
                        help="Show cross-source connections for a compound")
    parser.add_argument("--search",   metavar="QUERY",
                        help="Semantic similarity search")
    parser.add_argument("--gaps",     action="store_true",
                        help="Show unfollowed research findings")
    parser.add_argument("--export-connections", metavar="FILE",
                        help="Export connections to CSV")
    args = parser.parse_args()

    if args.entity:
        search_by_entity(args.entity[0], args.entity[1])
    elif args.connections:
        show_connections(args.connections)
    elif args.search:
        semantic_search(args.search)
    elif args.gaps:
        show_gaps()
    elif args.export_connections:
        export_connections(args.export_connections)
    else:
        parser.print_help()
