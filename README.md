# SHL Assessment Recommender — AI Intern Assignment

A conversational FastAPI agent that recommends SHL Individual Test Solutions through dialogue.

---

## Project Structure

```
shl_recommender/
├── main.py           # FastAPI app + retrieval + prompt engineering
├── catalog.json      # SHL Individual Test Solutions (scraped + curated)
├── requirements.txt  # Python dependencies
├── render.yaml       # One-click Render deployment
├── test_local.py     # Local evaluation harness
└── README.md
```

---

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Start the server
uvicorn main:app --reload --port 8000

# 4. Test health
curl http://localhost:8000/health
# {"status":"ok"}

# 5. Run evaluation traces
python test_local.py --url http://localhost:8000
```

---

## API

### `GET /health`
Returns `{"status": "ok"}` with HTTP 200.

### `POST /chat`

**Request**
```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response**
```json
{
  "reply": "Here are 5 assessments that fit a mid-level Java developer...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r",       "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

---

## Deploy to Render (free tier)

1. Push this folder to a GitHub repo
2. Go to [render.com](https://render.com) → New → Web Service → connect your repo
3. Render auto-detects `render.yaml`
4. Add environment variable `ANTHROPIC_API_KEY` in the Render dashboard
5. Deploy — your `/health` and `/chat` endpoints will be live

---

## Design Decisions

### Retrieval
- **TF-IDF cosine similarity** over a pre-built numpy matrix — zero dependencies beyond numpy, sub-millisecond latency, no cold-start
- Query is assembled from the full conversation history (all user turns concatenated) so context accumulates naturally
- Top-20 catalog items injected into the system prompt each call

### Agent Behaviour
- System prompt enforces: clarify-before-recommend, 1–10 recs, URL safety, scope refusal, comparison grounding
- JSON-only output contract + regex fallback for fence stripping
- Post-generation validation: every URL checked against catalog set; name-matched fallback rewrites hallucinated URLs

### Catalog
- 60 Individual Test Solutions covering: cognitive (Verify range), personality (OPQ, MQ), knowledge (Java, Python, SQL, etc.), simulations (Automata, Contact Center), and behavioural (SJT, Sales predictor)
- Each entry includes: name, url, test_type, description, job_levels, remote_testing, adaptive, keywords, job_families

### Limits compliance
- Max 8 turns: agent is designed to recommend within 2–3 turns for most queries
- 30s timeout: single Claude Sonnet call, no chained calls

---

## Test Types Reference
| Code | Meaning |
|------|---------|
| A | Ability & Aptitude |
| B | Biodata & Situational Judgement |
| C | Competencies |
| D | Development & 360 |
| E | Assessment Exercises |
| K | Knowledge & Skills |
| P | Personality & Behavior |
| S | Simulations |
