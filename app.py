"""Industry Intel Agent: Hugging Face Space version.

Pick an industry; an AI proposes the leading listed companies, the agent scrapes live
market data (Yahoo Finance, Wikidata), rejects tickers it cannot verify, charts the
results and an AI writes a strategic read-out grounded only in the scraped table.

Secrets (Space Settings > Variables and secrets):
  GEMINI_API_KEY  main AI
  GROQ_API_KEY    optional but recommended, free backup AI (console.groq.com)
  HF_TOKEN        optional, second backup through Hugging Face Inference Providers
"""
import os, re, json, time, glob, base64, zipfile, threading, datetime as dt

import gradio as gr
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go

# ---------- AI connection: Gemini first, Hugging Face backup ----------
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GEMINI_MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_MODELS", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",") if m.strip()]
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODELS = [m.strip() for m in os.environ.get(
    "GROQ_MODELS", "llama-3.3-70b-versatile,openai/gpt-oss-120b").split(",") if m.strip()]
HF_MODELS = [m.strip() for m in os.environ.get(
    "HF_MODELS", "meta-llama/Llama-3.3-70B-Instruct,Qwen/Qwen2.5-72B-Instruct").split(",") if m.strip()]

gemini = None
if GEMINI_KEY:
    from google import genai
    from google.genai import types
    gemini = genai.Client(api_key=GEMINI_KEY, http_options=types.HttpOptions(timeout=90_000))

hf = None
if HF_TOKEN:
    from huggingface_hub import InferenceClient
    hf = InferenceClient(token=HF_TOKEN, timeout=180)

BUSY = ("503", "504", "429", "UNAVAILABLE", "DEADLINE", "timed out", "Timeout", "timeout", "overloaded")
WORKING = {"model": None}           # last model that answered is tried first next time


def _call(model, prompt):
    if model.startswith("groq:"):
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {GROQ_KEY}"},
                          json={"model": model[5:], "temperature": 0.3,
                                "messages": [{"role": "user", "content": prompt}]}, timeout=90)
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} {r.text[:150]}")
        return r.json()["choices"][0]["message"]["content"]
    if model.startswith("hf:"):
        r = hf.chat_completion(model=model[3:], messages=[{"role": "user", "content": prompt}],
                               max_tokens=1500, temperature=0.3)
        return r.choices[0].message.content
    return gemini.models.generate_content(model=model, contents=prompt).text


def ask_llm(prompt, log, tries=2):
    order = ([m for m in GEMINI_MODELS] if gemini else []) + ([f"groq:{m}" for m in GROQ_MODELS] if GROQ_KEY else []) \
        + ([f"hf:{m}" for m in HF_MODELS] if hf else [])
    if not order:
        raise RuntimeError("No AI key found. Add GEMINI_API_KEY and/or GROQ_API_KEY as environment variables.")
    order.sort(key=lambda m: m != WORKING["model"])
    errors = []
    for m in order:
        for attempt in range(tries):
            try:
                log(f"  calling {m} (attempt {attempt + 1})...")
                text = _call(m, prompt)
                WORKING["model"] = m
                return text, m
            except Exception as e:
                msg = str(e)
                errors.append(f"{m} try {attempt + 1}: {msg[:120]}")
                if any(k in msg for k in ("504", "DEADLINE", "404", "NOT_FOUND", "timed out", "Timeout")):
                    log(f"  {m} unavailable, switching model...")
                    break
                if any(k in msg for k in BUSY):
                    wait = 5 * (attempt + 1)
                    log(f"  {m} busy, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    break
    raise RuntimeError("Every AI model failed:\n" + "\n".join(errors))


def parse_json(text):
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if not m:
        raise ValueError("No JSON found in the AI reply")
    return json.loads(m.group(1))


# ---------- AI step 1: pick companies ----------
def pick_companies(industry, n, log):
    prompt = f"""You are an equity research assistant.
List the {n} largest publicly listed companies in this industry: "{industry}".
Prefer companies listed on US exchanges (NYSE or NASDAQ). For non-US listings use the Yahoo Finance ticker format (for example AIR.PA).
Return ONLY a JSON array, no other text, where each item is:
{{"name": "company name", "ticker": "TICKER", "founded_guess": 1950}}
founded_guess is your best estimate of the founding year (integer)."""
    text, model = ask_llm(prompt, log)
    clean = []
    for it in parse_json(text):
        t = str(it.get("ticker", "")).strip().upper()
        if t:
            clean.append({"name": it.get("name", t), "ticker": t, "founded_guess": it.get("founded_guess")})
    return clean[:n], model, prompt, text


# ---------- scrape and verify ----------
WIKIDATA = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "IndustryIntelAgent/1.0 (ASU student project; huggingface.co/spaces)"}
FX = {"USD": 1.0}


def to_usd(cur):
    cur = (cur or "USD").upper()
    if cur not in FX:
        try:
            FX[cur] = float(yf.Ticker(f"{cur}USD=X").fast_info["last_price"])
        except Exception:
            FX[cur] = None
    return FX[cur]


def founded_from_wikidata(ticker):
    q = f"""SELECT ?inception WHERE {{
      VALUES ?ex {{ wd:Q13677 wd:Q82059 }}
      ?item p:P414 ?st . ?st ps:P414 ?ex ; pq:P249 "{ticker}" .
      ?item wdt:P571 ?inception .
    }} LIMIT 1"""
    try:
        r = requests.get(WIKIDATA, params={"query": q, "format": "json"}, headers=HEADERS, timeout=20)
        rows = r.json()["results"]["bindings"]
        if rows:
            return int(rows[0]["inception"]["value"][:4])
    except Exception:
        pass
    return None


def scrape_company(c):
    info = yf.Ticker(c["ticker"]).info or {}
    fx_mc = to_usd(info.get("currency"))
    fx_rev = to_usd(info.get("financialCurrency") or info.get("currency"))
    officers = info.get("companyOfficers") or []
    execs = "; ".join(f"{o.get('name', '')} ({o.get('title', '')})" for o in officers[:3])
    founded, src = founded_from_wikidata(c["ticker"]), "Wikidata"
    if founded is None and c.get("founded_guess"):
        founded, src = int(c["founded_guess"]), "AI estimate (unverified)"
    year = dt.date.today().year
    return {
        "Company": info.get("longName") or c["name"],
        "Ticker": c["ticker"],
        "Market cap ($B)": round(info["marketCap"] * fx_mc / 1e9, 2) if info.get("marketCap") and fx_mc else None,
        "Stock price (local)": info.get("currentPrice") or info.get("regularMarketPrice"),
        "Currency": info.get("currency"),
        "Revenue ($B)": round(info["totalRevenue"] * fx_rev / 1e9, 2) if info.get("totalRevenue") and fx_rev else None,
        "Employees": info.get("fullTimeEmployees"),
        "Founded": founded,
        "Years in business": (year - founded) if founded else None,
        "Founded source": src if founded else "not found",
        "Executive team (top 3)": execs or "not listed",
        "Country": info.get("country"),
    }


def ai_figures(companies, log):
    """Fallback when Yahoo Finance blocks this server: AI-supplied figures, clearly labelled."""
    names = ", ".join(f"{c['name']} ({c['ticker']})" for c in companies)
    prompt = f"""For each company below give your best recent estimate of market capitalization and annual revenue
in US dollar billions, full-time employees, CEO name, headquarters country and founding year.
Companies: {names}
Return ONLY a JSON array, each item:
{{"ticker": "TICKER", "name": "Company", "market_cap_b": 0.0, "revenue_b": 0.0, "employees": 0, "ceo": "Name", "country": "Country", "founded": 1950}}"""
    text, _ = ask_llm(prompt, log)
    out = {}
    for it in parse_json(text):
        out[str(it.get("ticker", "")).upper()] = it
    return out


def build_table(companies, log):
    rows, dropped, blocked = [], [], []
    for c in companies:
        row = None
        for attempt in range(2):
            try:
                row = scrape_company(c)
                break
            except Exception as e:
                row = None
                if attempt == 0:
                    log(f"  {c['ticker']}: {type(e).__name__}, retrying in 3s...")
                    time.sleep(3)
                else:
                    blocked.append(c)
        if row is None:
            continue
        if row["Market cap ($B)"] is None and row["Stock price (local)"] is None:
            dropped.append(f"{c['ticker']} ({c['name']}): no live market data, AI suggestion rejected")
        else:
            row["Data source"] = "Yahoo Finance (live)"
            rows.append(row)
            log(f"  scraped {c['ticker']}")
    if blocked:
        log(f"  Yahoo Finance did not respond for {len(blocked)} ticker(s); using AI estimates for those, labelled unverified")
        try:
            est = ai_figures(blocked, log)
            year = dt.date.today().year
            for c in blocked:
                it = est.get(c["ticker"])
                if not it:
                    dropped.append(f"{c['ticker']} ({c['name']}): no data from Yahoo or AI")
                    continue
                fy = it.get("founded")
                fy = int(fy) if isinstance(fy, (int, float)) and fy else None
                rows.append({
                    "Company": it.get("name") or c["name"], "Ticker": c["ticker"],
                    "Market cap ($B)": it.get("market_cap_b"), "Stock price (local)": None, "Currency": "USD",
                    "Revenue ($B)": it.get("revenue_b"), "Employees": it.get("employees"),
                    "Founded": fy, "Years in business": (year - fy) if fy else None,
                    "Founded source": "AI estimate (unverified)" if fy else "not found",
                    "Executive team (top 3)": (it.get("ceo") or "not listed") + " (CEO)",
                    "Country": it.get("country"), "Data source": "AI estimate (unverified)"})
        except Exception as e:
            for c in blocked:
                dropped.append(f"{c['ticker']} ({c['name']}): Yahoo blocked and AI fallback failed ({type(e).__name__})")
    df = pd.DataFrame(rows)
    for col in ("Market cap ($B)", "Revenue ($B)", "Employees"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if not df.empty:
        df = df.sort_values("Market cap ($B)", ascending=False, na_position="last").reset_index(drop=True)
    return df, dropped


# ---------- AI step 2: read-out ----------
def strategic_readout(df, industry, log):
    cols = ["Company", "Ticker", "Market cap ($B)", "Revenue ($B)", "Employees", "Years in business"]
    prompt = f"""You are a strategy analyst. Using ONLY the data table below for the "{industry}" industry,
write 4 short bullet points: who leads on scale, which firms look young or fast-scaling, where the
market cap to revenue gap suggests investor expectations of growth, and one caution about the data.
Do not add facts that are not in the table.

{df[cols].to_csv(index=False)}"""
    text, model = ask_llm(prompt, log)
    return text, model, prompt


# ---------- charts and page styling ----------
NAVY, INK = "#0F2A47", "#1F2937"
PALETTE = ["#1F4E79", "#E8833A", "#2E8B8B", "#7A4E9E", "#C0504D", "#4F8A3C", "#D4A017", "#5B6B7F",
           "#8C564B", "#17A2B8", "#B85C8A", "#6C757D"]


def short(name):
    name = str(name)
    for w in [", Inc.", " Corporation", " Incorporated", " Company", " Group", " Holdings", " Inc.", " Inc",
              " plc", " SE", " SA", " N.V.", " Corp."]:
        name = name.replace(w, "")
    return name.replace("The ", "").strip()


def style(fig, title, h=460):
    fig.update_layout(template="plotly_white", height=h,
                      title=dict(text=title, x=0.01, font=dict(size=18, color=NAVY)),
                      font=dict(family="Arial", size=13, color=INK), margin=dict(l=60, r=70, t=70, b=60),
                      showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(automargin=True, showgrid=False)
    fig.update_yaxes(automargin=True, gridcolor="#E5E7EB")
    return fig


def make_charts(df, industry):
    d = df.copy()
    d["Name"] = d["Company"].map(short)
    col = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(d["Ticker"])}
    figs = []
    a = d.dropna(subset=["Market cap ($B)"]).sort_values("Market cap ($B)")
    f1 = go.Figure(go.Bar(x=a["Market cap ($B)"], y=a["Name"], orientation="h",
                          marker_color=[col[t] for t in a["Ticker"]],
                          text=[f"${v:,.1f}B" for v in a["Market cap ($B)"]], textposition="outside",
                          cliponaxis=False, hovertemplate="%{y}: $%{x:,.1f}B<extra></extra>"))
    f1.update_xaxes(title="Market cap, USD billions")
    figs.append(style(f1, f"Market capitalization: {industry}", h=max(380, 52 * len(a) + 120)))
    b = d.dropna(subset=["Market cap ($B)", "Years in business"])
    if not b.empty:
        emp = b["Employees"].fillna(b["Employees"].median() if b["Employees"].notna().any() else 1000).astype(float)
        f2 = go.Figure(go.Scatter(
            x=b["Years in business"], y=b["Market cap ($B)"], mode="markers+text",
            text=b["Ticker"], textposition="top center",
            marker=dict(size=emp, sizemode="area", sizeref=2.0 * emp.max() / 70 ** 2, sizemin=8,
                        color=[col[t] for t in b["Ticker"]], opacity=0.8, line=dict(width=1, color="white")),
            customdata=list(zip(b["Name"], emp, b["Founded source"])),
            hovertemplate="<b>%{customdata[0]}</b><br>%{x} years (%{customdata[2]})<br>"
                          "$%{y:,.1f}B market cap<br>%{customdata[1]:,.0f} employees<extra></extra>"))
        f2.update_xaxes(title="Years in business (current legal entity)")
        f2.update_yaxes(title="Market cap, USD billions")
        figs.append(style(f2, "Age versus scale (bubble size = employees)"))
    c = d.dropna(subset=["Revenue ($B)", "Market cap ($B)"]).copy()
    if not c.empty:
        c["mult"] = (c["Market cap ($B)"] / c["Revenue ($B)"]).round(2)
        c = c.sort_values("mult", ascending=False)
        f3 = go.Figure(go.Bar(x=c["Name"], y=c["mult"], marker_color=[col[t] for t in c["Ticker"]],
                              text=[f"{v:.1f}x" for v in c["mult"]], textposition="outside", cliponaxis=False,
                              hovertemplate="%{x}: %{y:.2f}x revenue<extra></extra>"))
        f3.update_yaxes(title="Market cap / revenue (x)")
        figs.append(style(f3, "Valuation multiple: market cap to revenue"))
    return figs


def model_label(m):
    if m.startswith("hf:") or m.startswith("groq:"):
        return m.split(":", 1)[1].split("/")[-1]
    return m


def kpi_html(df, secs, model):
    top = df.iloc[0]
    cards = [("Companies (live data)", f"{int((df['Data source'] == 'Yahoo Finance (live)').sum())} of {len(df)}"),
             ("Combined market cap", f"${df['Market cap ($B)'].sum():,.0f}B"),
             ("Largest by market cap", short(top["Company"])),
             ("Run time", f"{secs:.0f} s <div style='font-size:12px;font-weight:400;color:#6B7280'>"
                          f"AI model: {model_label(model)}</div>")]
    return "<div class='kpis'>" + "".join(
        f"<div class='kpi'><div class='kl'>{k}</div><div class='kv'>{v}</div></div>" for k, v in cards) + "</div>"


def display_table(df):
    t = df.copy()
    t["Company"] = t["Company"].map(short)
    t["Employees"] = t["Employees"].map(lambda v: f"{v:,.0f}" if v == v and v is not None else "")
    t["CEO / top executive"] = t["Executive team (top 3)"].astype(str).str.split(";").str[0]
    return t[["Company", "Ticker", "Market cap ($B)", "Revenue ($B)", "Stock price (local)", "Currency",
              "Employees", "Founded", "Founded source", "Country", "CEO / top executive", "Data source"]]


def hero_style():
    """Uses background.jpg/png from the Space files if present, else a gradient."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in sorted(glob.glob(os.path.join(here, "background.*"))):
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in ("jpg", "jpeg", "png", "webp"):
            b64 = base64.b64encode(open(path, "rb").read()).decode()
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            return ("background-image:linear-gradient(90deg,rgba(15,42,71,.92),rgba(15,42,71,.45)),"
                    f"url(data:image/{mime};base64,{b64});background-size:cover;background-position:center;")
    return "background:linear-gradient(120deg,#0F2A47 0%,#1F4E79 60%,#2E8B8B 100%);"


HERO = ("<div class='hero' style='" + hero_style() + "'>"
        "<div class='eyebrow'>TMC 531 &middot; Generative AI agent</div>"
        "<h1>Industry Intel Agent</h1>"
        "<p>Pick an industry. The agent asks an AI for the leading listed companies, scrapes live market data, "
        "rejects anything it cannot verify, charts the results and writes a strategic read-out.</p></div>")

CSS = """
.gradio-container{max-width:1200px!important;margin:auto!important;font-family:Arial,Helvetica,sans-serif}
.hero{border-radius:14px;padding:34px 36px;color:#fff;margin-bottom:6px}
.hero h1{color:#fff!important;font-size:38px;margin:4px 0 8px;font-weight:700}
.hero p{color:#E5ECF4!important;font-size:16px;max-width:760px;margin:0;line-height:1.5}
.hero .eyebrow{color:#F6C49A;font-size:14px;font-weight:600}
.panel{border:1px solid #E5E7EB;border-radius:12px;padding:16px!important}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:2px 0 6px}
.kpi{background:#fff;border:1px solid #E5E7EB;border-left:5px solid #E8833A;border-radius:10px;padding:12px 14px}
.kl{font-size:13px;color:#6B7280}.kv{font-size:19px;font-weight:700;color:#0F2A47;margin-top:2px}
@media(max-width:760px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.hero h1{font-size:28px}}
footer{display:none!important}
"""


# ---------- one run (each visitor gets their own log) ----------
def pipeline(industry, n, res, log_lines):
    def log(m):
        log_lines.append(m)
        print(m, flush=True)
    try:
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        log(f"[{stamp}] Industry: {industry} | companies requested: {n}")
        log("Step 1: asking the AI for the top companies...")
        companies, m1, p1, r1 = pick_companies(industry, n, log)
        log(f"  {m1} proposed: " + ", ".join(c["ticker"] for c in companies))
        log("Step 2 and 3: scraping Yahoo Finance and Wikidata, verifying tickers...")
        df, dropped = build_table(companies, log)
        for d in dropped:
            log("  REJECTED " + d)
        if df.empty:
            res["error"] = ("No company returned live data. Try a broader industry name. "
                            "If this repeats for every industry, Yahoo Finance may be limiting requests from this server.")
            return
        figs = make_charts(df, industry)
        log("Step 4: asking the AI for a strategic read-out...")
        text, m2, p2 = strategic_readout(df, industry, log)
        log("Done.")
        res.update(df=df, figs=figs, text=text, m1=m1, m2=m2, stamp=stamp,
                   prompts=[p1, p2], replies=[r1, text])
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"


def save_files(res, industry, log_lines):
    folder = f"/tmp/outputs/{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    os.makedirs(folder, exist_ok=True)
    files = []
    p = f"{folder}/industry_intel_results.csv"; res["df"].to_csv(p, index=False); files.append(p)
    for f, nme in zip(res["figs"], ["chart1_market_cap", "chart2_age_vs_scale", "chart3_valuation_multiple"]):
        p = f"{folder}/{nme}.html"; f.write_html(p, include_plotlyjs="cdn"); files.append(p)
    p = f"{folder}/ai_readout.md"
    with open(p, "w") as fh:
        fh.write(f"# AI strategic read-out: {industry}\n\nModel: {res['m2']} | Data retrieved {res['stamp']}\n\n{res['text']}\n")
    files.append(p)
    p = f"{folder}/run_log_and_prompts.txt"
    with open(p, "w") as fh:
        fh.write("RUN LOG\n" + "\n".join(log_lines) + "\n\n")
        for i, (pr, rp) in enumerate(zip(res["prompts"], res["replies"]), 1):
            fh.write(f"PROMPT {i}\n{pr}\n\nRESPONSE {i}\n{rp}\n\n")
    files.append(p)
    z = f"{folder}/industry_intel_all_outputs.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in files:
            zf.write(f, os.path.basename(f))
    return z, files


INDUSTRIES = [
    "Aerospace and defense", "Airlines", "Automobiles", "Electric vehicles", "Auto parts",
    "Banks", "Insurance", "Asset management", "Payments and fintech",
    "Semiconductors", "Semiconductor equipment", "Software", "Cloud computing", "Cybersecurity",
    "Artificial intelligence", "Consumer electronics", "Telecommunications", "Internet and e-commerce",
    "Media and entertainment", "Video games", "Pharmaceuticals", "Biotechnology", "Medical devices",
    "Health insurance", "Oil and gas", "Renewable energy", "Electric utilities", "Nuclear energy",
    "Mining", "Chemicals", "Steel", "Construction and engineering", "Industrial machinery",
    "Railroads", "Logistics and shipping", "Space launch and satellites", "Drones and unmanned systems",
    "Retail", "Food and beverage", "Restaurants", "Hotels and travel", "Real estate (REITs)",
]
OTHER = "Other (type your own)"
BLANK = (None, None, None, "", None, None, None)


def run(choice, other, n):
    industry = (other or "").strip() if choice == OTHER else choice
    if not industry:
        yield ("Please type an industry in the box.", "", "") + BLANK
        return
    res, log_lines, t0 = {}, [], time.time()
    worker = threading.Thread(target=pipeline, args=(industry, int(n), res, log_lines), daemon=True)
    worker.start()
    while worker.is_alive():
        last = log_lines[-1].strip() if log_lines else "Starting..."
        yield (f"**Running... {time.time() - t0:.0f} s**  \n{last}", "\n".join(log_lines), "") + BLANK
        time.sleep(1)
    secs = time.time() - t0
    if "error" in res:
        yield (f"**Stopped after {secs:.0f} s.** {res['error']}", "\n".join(log_lines), "") + BLANK
        return
    figs = res["figs"] + [None] * (3 - len(res["figs"]))
    readout = (f"#### AI strategic read-out ({model_label(res['m2'])}, data retrieved {res['stamp']})\n\n"
               + res["text"].replace("$", "\\$")
               + "\n\n*Sources: Yahoo Finance via yfinance, Wikidata; rows marked 'AI estimate (unverified)' in the Data table were not verified. "
               "AI output can contain errors; check figures before use.*")
    z, files = save_files(res, industry, log_lines)
    n_live = int((res["df"]["Data source"] == "Yahoo Finance (live)").sum())
    n_est = len(res["df"]) - n_live
    note = f" {n_est} row(s) are AI estimates because Yahoo Finance did not respond." if n_est else ""
    yield (f"**Finished in {secs:.0f} s.** {n_live} companies verified with live data for *{industry}*.{note} "
           "See the tabs for charts, read-out, data and downloads.",
           "\n".join(log_lines), kpi_html(res["df"], secs, res["m2"]),
           figs[0], figs[1], figs[2], readout, display_table(res["df"]), z, files)


with gr.Blocks(title="Industry Intel Agent") as app:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=280, elem_classes="panel"):
            industry_in = gr.Dropdown(INDUSTRIES + [OTHER], value="Aerospace and defense", label="Industry")
            other_in = gr.Textbox(label="Type your industry", placeholder="for example Quantum computing",
                                  visible=False)
            n_in = gr.Slider(3, 12, value=8, step=1, label="Number of companies")
            run_btn = gr.Button("Run agent", variant="primary", size="lg")
            status = gr.Markdown("Ready. Choose an industry and press **Run agent**. A run takes 1 to 5 minutes.")
        with gr.Column(scale=3):
            kpis = gr.HTML()
            with gr.Tabs():
                with gr.Tab("Charts"):
                    c1 = gr.Plot(show_label=False)
                    c2 = gr.Plot(show_label=False)
                    c3 = gr.Plot(show_label=False)
                with gr.Tab("AI read-out"):
                    readout_md = gr.Markdown("The AI read-out appears here after a run.")
                with gr.Tab("Data table"):
                    table = gr.Dataframe(wrap=True, interactive=False, show_label=False)
                with gr.Tab("Downloads"):
                    zip_out = gr.File(label="Everything in one ZIP")
                    downloads = gr.File(label="Individual files: CSV, charts (HTML), read-out, run log with prompts",
                                        file_count="multiple")
                with gr.Tab("Agent log"):
                    log_box = gr.Textbox(lines=16, max_lines=24, show_label=False)
    industry_in.change(lambda c: gr.update(visible=(c == OTHER)), industry_in, other_in)
    run_btn.click(run, [industry_in, other_in, n_in],
                  [status, log_box, kpis, c1, c2, c3, readout_md, table, zip_out, downloads],
                  concurrency_limit=2)

theme = gr.themes.Soft(primary_hue="orange", neutral_hue="slate", font=["Arial", "Helvetica", "sans-serif"])

if __name__ == "__main__":
    app.queue(max_size=10).launch(css=CSS, theme=theme, server_name="0.0.0.0",
                                  server_port=int(os.environ.get("PORT", 7860)))
