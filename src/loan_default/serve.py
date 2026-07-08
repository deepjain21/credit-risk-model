"""Serving app: a FastAPI /predict endpoint plus a simple form UI at /.

Run locally:
    uv run uvicorn loan_default.serve:app --reload
    # then open http://127.0.0.1:8000  for the form, or:
    curl -s localhost:8000/predict -H 'content-type: application/json' \
         -d @examples/sample_applicant.json | jq

The endpoint accepts the fields available at the lending decision, rebuilds the
model features (same cleaning as training) and returns a default probability, a
decision, a risk band, and the top per-applicant factors behind the score. This
is a design stub, not a hardened service — see docs/DESIGN_NOTE.md.
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .predict import load_artifact, predict_one

app = FastAPI(title="Personal-Loan Default Risk", version="0.1.0")


class Applicant(BaseModel):
    """Decision-time inputs. Extra bureau/customer fields are optional because
    a thin bureau pull or missing master record still needs to be scorable."""
    # application
    application_date: str = Field(..., examples=["2023-06-15"])
    requested_amount: float
    loan_term_months: int
    declared_income: float
    employment_type: Optional[str] = None
    employer_name: Optional[str] = None
    emirate: Optional[str] = None
    product_id: str
    branch_id: str
    channel: Optional[str] = None
    app_form_version: Optional[str] = None
    # customer master
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    nationality_group: Optional[str] = None
    marital_status: Optional[str] = None
    dependents_count: Optional[int] = None
    # bureau pull (point-in-time)
    bureau_score: Optional[float] = None
    num_inquiries_last_3m: Optional[int] = None
    total_outstanding_debt: Optional[float] = None
    num_dpd_90_last_12m: Optional[int] = None
    has_long_tenor_loan: Optional[int] = None


class Factor(BaseModel):
    feature: str
    value: float | str | None = None
    direction: str          # "increases" or "decreases" (default risk)
    impact: float


class Prediction(BaseModel):
    default_probability: float
    decision: str
    risk_band: str
    top_factors: list[Factor]
    threshold: float
    model_name: str
    model_trained_at: str


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness + confirms the model artifact loads."""
    try:
        art = load_artifact()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"status": "ok", "model": art["model_name"], "trained_at": art["trained_at"]}


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant) -> Prediction:
    try:
        return Prediction(**predict_one(applicant.model_dump()))
    except FileNotFoundError as e:      # model not trained yet
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def form() -> str:
    """A minimal single-page form that posts to /predict and shows the reasons."""
    return FORM_HTML


# --------------------------------------------------------------------------- #
# Self-contained form page (inline CSS/JS; same-origin fetch to /predict).
# Pre-filled with the sample applicant so it scores on the first click.
# --------------------------------------------------------------------------- #
FORM_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loan Default Risk</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1f2937; --muted:#6b7280; --line:#e5e7eb;
          --up:#c0392b; --down:#16a085; --brand:#4f46e5; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--ink); }
  .wrap { max-width:960px; margin:0 auto; padding:24px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 20px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; align-items:start; }
  @media (max-width:760px){ .grid { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px; }
  form { display:grid; grid-template-columns:1fr 1fr; gap:10px 14px; }
  label { display:flex; flex-direction:column; font-size:12px; color:var(--muted); gap:3px; }
  input { font:inherit; padding:7px 9px; border:1px solid var(--line); border-radius:7px; color:var(--ink); }
  .full { grid-column:1 / -1; }
  button { grid-column:1 / -1; margin-top:6px; background:var(--brand); color:#fff; border:0;
           padding:11px; border-radius:8px; font-weight:600; font-size:15px; cursor:pointer; }
  button:hover { filter:brightness(1.07); }
  .score { font-size:44px; font-weight:700; margin:2px 0; }
  .badge { display:inline-block; padding:4px 12px; border-radius:999px; font-weight:600; font-size:13px; }
  .approve { background:#e7f7f1; color:var(--down); }
  .decline { background:#fdecea; color:var(--up); }
  .meta { color:var(--muted); font-size:12px; margin-top:6px; }
  h3 { font-size:14px; margin:18px 0 8px; }
  .factor { display:grid; grid-template-columns:18px 1fr 120px; align-items:center; gap:8px; margin:7px 0; font-size:13px; }
  .dir.up { color:var(--up); } .dir.down { color:var(--down); }
  .bar { background:#eef0f3; border-radius:5px; height:10px; overflow:hidden; }
  .bar > span { display:block; height:100%; }
  .bar > span.up { background:var(--up); } .bar > span.down { background:var(--down); }
  .placeholder { color:var(--muted); }
  .err { color:var(--up); }
</style>
</head>
<body>
<div class="wrap">
  <h1>Personal-Loan Default Risk</h1>
  <p class="sub">Enter an applicant's decision-time details to get a default-risk score and the factors behind it.</p>
  <div class="grid">
    <div class="card">
      <form id="f" onsubmit="score(event)">
        <label>Application date<input name="application_date" value="2023-06-15"></label>
        <label>Date of birth<input name="date_of_birth" value="1988-04-10"></label>
        <label>Requested amount (AED)<input name="requested_amount" data-num value="150000"></label>
        <label>Loan term (months)<input name="loan_term_months" data-num value="36"></label>
        <label>Declared monthly income<input name="declared_income" data-num value="14000"></label>
        <label>Employment type<input name="employment_type" value="SALARIED"></label>
        <label>Employer name<input name="employer_name" value="Employer_0575"></label>
        <label>Emirate<input name="emirate" value="DXB"></label>
        <label>Product ID<input name="product_id" value="PRD-01"></label>
        <label>Branch ID<input name="branch_id" value="BR-DXB1"></label>
        <label>Channel<input name="channel" value="Branch"></label>
        <label>App form version<input name="app_form_version" value="v3"></label>
        <label>Gender<input name="gender" value="M"></label>
        <label>Nationality group<input name="nationality_group" value="Expat - Asia"></label>
        <label>Marital status<input name="marital_status" value="Married"></label>
        <label>Dependents<input name="dependents_count" data-num value="2"></label>
        <label>Bureau score<input name="bureau_score" data-num value="640"></label>
        <label>Inquiries (last 3m)<input name="num_inquiries_last_3m" data-num value="3"></label>
        <label>Outstanding debt (AED)<input name="total_outstanding_debt" data-num value="210000"></label>
        <label>90+ DPD (last 12m)<input name="num_dpd_90_last_12m" data-num value="1"></label>
        <label>Has long-tenor loan (0/1)<input name="has_long_tenor_loan" data-num value="0"></label>
        <button type="submit">Score applicant</button>
      </form>
    </div>
    <div class="card" id="result">
      <p class="placeholder">The score and its top factors will appear here.</p>
    </div>
  </div>
</div>
<script>
async function score(e){
  e.preventDefault();
  const payload = {};
  for (const el of e.target.elements){
    if(!el.name || el.value.trim()==="") continue;
    payload[el.name] = el.hasAttribute("data-num") ? Number(el.value) : el.value.trim();
  }
  const out = document.getElementById("result");
  out.innerHTML = "<p class='placeholder'>Scoring…</p>";
  try {
    const res = await fetch("/predict", {method:"POST",
      headers:{"content-type":"application/json"}, body: JSON.stringify(payload)});
    if(!res.ok){ out.innerHTML = "<p class='err'>Error "+res.status+": "+(await res.text())+"</p>"; return; }
    render(await res.json());
  } catch(err){ out.innerHTML = "<p class='err'>"+err+"</p>"; }
}
function render(d){
  const pct = (d.default_probability*100).toFixed(1);
  const decline = d.decision === "decline";
  const factors = d.top_factors.map(f => {
    const up = f.direction === "increases";
    const w = Math.min(100, f.impact*180);
    const val = (f.value===null||f.value===undefined) ? "" : ": <b>"+f.value+"</b>";
    return `<div class="factor">
      <span class="dir ${up?'up':'down'}">${up?'▲':'▼'}</span>
      <span>${f.feature}${val}</span>
      <span class="bar"><span class="${up?'up':'down'}" style="width:${w}%"></span></span>
    </div>`;
  }).join("");
  document.getElementById("result").innerHTML = `
    <div class="score">${pct}%</div>
    <span class="badge ${decline?'decline':'approve'}">${d.decision.toUpperCase()}</span>
    &nbsp;<span class="meta">risk band: ${d.risk_band} · threshold ${(d.threshold*100).toFixed(0)}%</span>
    <h3>Top factors behind this score</h3>
    <div>${factors}</div>
    <p class="meta">▲ increases risk · ▼ reduces risk (bar = relative strength).
       Model: ${d.model_name}.</p>`;
}
</script>
</body>
</html>
"""
