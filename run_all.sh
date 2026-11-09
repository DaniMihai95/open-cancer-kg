#!/bin/bash
# Build the whole graph on this machine: Ollama with both models runs here.
# The two models are never loaded together: every source is extracted first
# (--no-embed), then one --embed pass adds the embeddings.
#   ./run_all.sh            full run, the same limits as dataset v1.0 (several days)
#   ./run_all.sh test       100 PubMed records, to check the setup first
# Outputs go to ./work (cancer_output/, cancer_pipeline.db, cancer_pipeline.log).
# It resumes where it stopped: run it again after an interruption.
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"
WORK="${OCKG_WORK:-$REPO/work}"
mkdir -p "$WORK" && cd "$WORK" || exit 1

export OLLAMA_BASE="${OLLAMA_BASE:-http://localhost:11434}"
export EMBED_BASE="${EMBED_BASE:-$OLLAMA_BASE}"
export LLM_MODEL="${LLM_MODEL:-qwen3.6:35b}"
export EMBED_MODEL="${EMBED_MODEL:-qwen3-embedding:4b}"

PY="$WORK/venv/bin/python"
if [ ! -x "$PY" ]; then
  python3 -m venv "$WORK/venv" || { echo "Install python3-venv first"; exit 1; }
  "$WORK/venv/bin/pip" -q install -r "$REPO/requirements.txt" || exit 1
fi

# The three public sources must answer before days of work start.
for u in "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi" \
         "https://clinicaltrials.gov/api/v2/version" \
         "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/Title/TXT"; do
  curl -fsS -m 20 -o /dev/null "$u" || { echo "Cannot reach $u"; exit 1; }
done

P="$REPO/pipeline.py"
if [ "${1:-}" = "test" ]; then
  $PY "$P" --source pubmed --limit 100 --workers 2 --no-embed && $PY "$P" --embed && $PY "$P" --stats
  exit
fi
$PY "$P" --source pubmed  --limit 50000 --workers 2 --no-embed &&
$PY "$P" --source trials  --limit 20000 --workers 2 --no-embed &&
$PY "$P" --source pubchem --limit 10000 --workers 2 --no-embed &&
$PY "$P" --embed &&
$PY "$P" --crossref &&
$PY "$P" --stats
echo "OCKG run finished: $(date)"
