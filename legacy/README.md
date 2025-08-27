## Statement verification against a large corpus (RAG + LLM)

This project verifies whether each statement (about ~2,000) is consistent with a large text corpus (~millions of tokens) by:
- Chunking the corpus and creating embeddings (vector index)
- Retrieving top-k relevant chunks for each statement
- Asking the LLM (gpt-5-mini) to judge "accurate" or "not accurate" with a brief explanation
- Streaming results to an Excel file incrementally
- Saving a checkpoint to resume if interrupted
- Showing a progress bar

### Files
- `api.txt`: holds `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`
- `Hearing Transcripts ALL.txt`: the large source corpus (TXT)
- `JUDGEMENT XLS V1.xlsx` or `RozhkovaWS.txt`: statements to verify (Excel or TXT)
- `verify_statements.py`: main script
- `requirements.txt`: dependencies

### Install
PowerShell (Windows):
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run (defaults point to files in this folder)
```powershell
python verify_statements.py `
  --api "api.txt" `
  --corpus "Hearing Transcripts ALL.txt" `
  --statements "JUDGEMENT XLS V1.xlsx" `
  --out "verification_results.xlsx" `
  --ckpt "verification_checkpoint.jsonl" `
  --judge-model "gpt-5-mini" `
  --embed-model "text-embedding-3-large" `
  --chunk-chars 8000 `
  --overlap-chars 1000 `
  --top-k 8 `
  --max-snippet-chars 1800
```
- The script resumes automatically from the checkpoint file if present.
- Results are appended row-by-row to `verification_results.xlsx` as they are produced.

### Notes
- Costs: The first run embeds the entire corpus. Subsequent runs reuse the cached index in `.index_cache/` unless the corpus or parameters change.
- If your statements are in a different Excel column name, the script auto-detects a non-empty column (or you can rename a column to `statement`).
- You can use a TXT file with one statement per line.
