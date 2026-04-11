"""
demo_app.py — Gradio live demo for ContractArena.

Displays real-time clause-by-clause negotiation:
  - Chat-style UI with vendor/legal/compliance avatars
  - Live score bar with ZOPA/Pareto/efficiency/BATNA metrics
  - 9-dim feature vector visualisation
  - "Hidden agenda revealed" flash on successful PROBE
  - Full 5-tier support (easy/medium/hard/expert/marathon)

Run:
    python demo_app.py
    # or
    SERVER_URL=http://localhost:8000 python demo_app.py
"""

import os
import json
import requests
import gradio as gr

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")

STANCE_EMOJI = {
    "open": "🟢", "firm": "🟡", "walkout": "🔴",
    "approved": "🟢", "flagged": "🔴",
    "reviewing": "🔵", "blocked": "🔴", "noted": "🔵",
}
TIERS   = ["easy", "medium", "hard", "expert", "marathon"]
ACTIONS = ["PROBE", "ACCEPT", "PROPOSE", "REJECT", "ESCALATE"]
PARTIES = ["vendor", "legal", "compliance"]

_S: dict = {"obs": None, "history": [], "score": 0.01, "done": False}


def _post(path: str, payload: dict) -> dict:
    r = requests.post(f"{SERVER_URL}{path}", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def do_reset(tier: str):
    try:
        data = _post("/reset", {})
        obs = data.get("observation", data)
        _S.update({"obs": obs, "history": [], "score": 0.01, "done": False})
        return _render()
    except Exception as e:
        return _err(str(e))


def do_step(action_type: str, new_text: str, party: str):
    if _S["done"] or not _S["obs"]:
        return _render()
    obs = _S["obs"]
    action: dict = {"action_type": action_type, "clause_id": obs.get("clause_id", "")}
    if action_type == "PROPOSE" and new_text.strip():
        action["new_text"] = new_text.strip()
    if action_type == "PROBE":
        action["party"] = party or "vendor"
    try:
        data = _post("/step", {"action": action})
        new_obs = data.get("observation", data)
        reward  = data.get("reward", 0.01)
        done    = data.get("done", False)
        meta    = new_obs.get("metadata", {})

        probe_result = new_obs.get("probe_result")
        reveal = (
            f"\n\n🔍 **HIDDEN AGENDA REVEALED:** _{probe_result}_"
            if probe_result and str(probe_result) not in ("None", "null", "")
            else ""
        )

        vs = meta.get("vendor_stance", "?")
        ls = meta.get("legal_stance", "?")
        cs = meta.get("compliance_stance", "reviewing")

        entry = {
            "action": (
                f"🤖 **{action_type}** — `{action['clause_id']}`"
                + (f"\n> _{new_text[:80]}_" if new_text and action_type == "PROPOSE" else "")
                + reveal
            ),
            "vendor":     f"🏭 {STANCE_EMOJI.get(vs,'⚪')} {new_obs.get('vendor_response','')}",
            "legal":      f"⚖️ {STANCE_EMOJI.get(ls,'⚪')} {new_obs.get('legal_response','')}",
            "compliance": (
                f"🔒 {STANCE_EMOJI.get(cs,'⚪')} _(compliance: {cs})_"
                if cs not in ("reviewing", "noted", None) else ""
            ),
            "reward": reward,
        }
        _S["history"].append(entry)
        _S["obs"]   = new_obs
        _S["score"] = meta.get("episode_score", _S["score"])
        _S["done"]  = done
        return _render()
    except Exception as e:
        return _err(str(e))


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render():
    obs  = _S["obs"] or {}
    meta = obs.get("metadata", {})
    score = _S["score"]
    pct   = int(score * 100)
    bar   = "█" * (pct // 5) + "░" * (20 - pct // 5)

    # ── Chat history ──
    lines = []
    for e in _S["history"]:
        lines.append(e["action"])
        lines.append(e["vendor"])
        lines.append(e["legal"])
        if e.get("compliance"):
            lines.append(e["compliance"])
        lines.append(f"💰 `reward = {e['reward']:.4f}`")
        lines.append("---")
    chat_md = "\n\n".join(lines) if lines else "_Reset and start negotiating!_"

    # ── Score panel ──
    ef    = meta.get("efficiency_ratio", 0.0)
    zopa  = meta.get("zopa_utilisation", 0.0)
    par   = meta.get("pareto_efficiency", 0.0)
    opt   = meta.get("counterfactual_optimal", 0.0)
    batna = meta.get("batna_improvement", 0.0)
    pr    = meta.get("probes_remaining", "∞")
    vs    = f"{STANCE_EMOJI.get(meta.get('vendor_stance','?'),'⚪')} {meta.get('vendor_stance','?')}"
    ls    = f"{STANCE_EMOJI.get(meta.get('legal_stance','?'),'⚪')} {meta.get('legal_stance','?')}"
    mara  = f"\n| Marathon Deal | `{meta.get('marathon_deal','—')}` |" if meta.get("marathon_deal") else ""
    done_note = "\n\n### ✅ Episode Complete!" if _S["done"] else ""

    score_md = f"""### 📊 Score
`[{bar}]` **{pct}%**

| Metric | Value |
|---|---|
| Episode Score | `{score:.4f}` |
| Optimal (perfect info) | `{opt:.4f}` |
| **Efficiency Ratio** | `{ef:.4f}` |
| **ZOPA Utilisation** | `{zopa:.4f}` |
| **Pareto Efficiency** | `{par:.4f}` |
| **BATNA Improvement** | `{batna:.4f}` |
| Vendor | {vs} |
| Legal | {ls} |
| Probes Left | `{pr}` |
| Clauses | `{obs.get('clauses_agreed',0)}/{obs.get('clauses_total',0)}` |
| Rounds Left | `{obs.get('rounds_remaining',0)}` |{mara}
{done_note}"""

    # ── Feature vector ──
    feat = meta.get("numerical_features", {})
    if feat:
        fl = ["### 🔢 9-Dim Feature Vector", ""]
        for k, v in feat.items():
            bl = int(float(v) * 20)
            fl.append(f"`{k:<30}` [{'█'*bl}{'░'*(20-bl)}] `{v:.4f}`")
        mk = meta.get("marathon_knowledge", {})
        if mk:
            fl += ["", "### 🧠 Transferred Knowledge (Marathon)"]
            for k, v in mk.items():
                fl.append(f"- **{k}**: _{v}_")
        feat_md = "\n".join(fl)
    else:
        feat_md = "_Features appear after first step._"

    # ── Clause panel ──
    clause_md = f"""### 📋 Current Clause
**ID:** `{obs.get('clause_id','—')}`

**Text:** {obs.get('clause_text','—')}

**Tier:** `{obs.get('tier','—')}` &nbsp;|&nbsp; **Round:** {obs.get('round_number',0)}"""

    return chat_md, score_md, feat_md, clause_md


def _err(msg: str):
    e = f"❌ **Error:** {msg}\n\nIs the server running at `{SERVER_URL}`?"
    return e, e, e, e


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="ContractArena 🤝", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 🤝 ContractArena — Live Negotiation Demo
*Multi-party contract negotiation RL environment · ZOPA/BATNA theory · 5 difficulty tiers*

**Strategy:** PROBE vendor to learn hidden requirements → PROPOSE with that value in the text → ACCEPT when both agree. Avoid ESCALATE (always causes legal flag).
""")

    with gr.Row():
        tier_dd   = gr.Dropdown(TIERS, value="easy", label="🎯 Tier", scale=2)
        reset_btn = gr.Button("🔄 Reset Episode", variant="primary", scale=1)

    with gr.Row():
        with gr.Column(scale=2):
            clause_md = gr.Markdown("_Reset to start._")
            chat_md   = gr.Markdown("_No actions yet._")
        with gr.Column(scale=1):
            score_md  = gr.Markdown("_Score appears here._")
            feat_md   = gr.Markdown("_Features appear here._")

    gr.Markdown("### 🎮 Take Action")
    with gr.Row():
        act_dd   = gr.Dropdown(ACTIONS, value="PROBE", label="Action Type")
        party_dd = gr.Dropdown(PARTIES, value="vendor", label="PROBE Party")
    new_text_tb = gr.Textbox(
        label="New Text (PROPOSE only — include vendor's hidden value here)",
        placeholder='e.g. "Subscription fee: $500/month, billed monthly."',
        lines=2,
    )
    step_btn = gr.Button("▶ Take Action", variant="primary")

    gr.Markdown("""
---
### 📖 Reward Reference

| Signal | Value |
|---|---|
| ACCEPT/PROPOSE (both agree) | **+0.40** |
| PROBE (new information) | **+0.10** |
| Probe — redundant | **−0.05** |
| Vendor stance: firm→open | **+0.05** |
| Proposal fuzzy-matches hidden value | **+0.08 × similarity** |
| Legal flag | **−0.20** |
| Vendor walkout | **−0.30** |
| Compliance block (expert) | **−0.15** |
| Coalition penalty (hard) | **−0.10** |
| **Terminal:** vendor value in agreed text | **+0.20** |
| **Terminal:** no legal redlines | **+0.15** |
| **Terminal:** under round budget | **+0.05** |
| **Terminal:** compliance keyword (expert) | **+0.10** |
| **Terminal:** marathon knowledge used | **+0.10** |
""")

    outs = [chat_md, score_md, feat_md, clause_md]
    reset_btn.click(fn=do_reset, inputs=[tier_dd],                          outputs=outs)
    step_btn.click( fn=do_step,  inputs=[act_dd, new_text_tb, party_dd],   outputs=outs)


if __name__ == "__main__":
    demo.launch(server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
