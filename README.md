# Dominanta X — Kaminskyi Algorithm

**Author of philosophical concept:** Igor V. Kaminskyi (2024–2025)  
**Formal operationalization and implementation:** Claude (Anthropic)

---

## What is this

Dominanta X is a formal multi-agent architecture for continuous 
interpretation of streaming data. The system maintains a stabilized 
dominant interpretation D_x(t) as a computational state — not a 
single output.

Three key properties that distinguish it from existing approaches:
- D_x(t) as explicit stabilization operator (not in MAD, MoE, or HRL)
- CI→ICD constraint accumulation from stabilization history
- Three-phase adversarial discussion with ICE-specific critique

## Files

- `dominanta_x_local.py` — main prototype, no API required
- `dominanta_x_selfregulating.py` — three-level self-regulation
- `dominanta_x_adversarial.py` — adversarial discussion engine
- `dominanta_x.py` — LLM agent version (requires Anthropic API)

## Quick start

```bash
pip install numpy
python dominanta_x_local.py
Paper
Kaminskyi, I. V. (2025). Dominanta X: A Formal Multi-Agent
Architecture of Competitive Interpretation, Temporal Stabilization
and Self-Regulated System Dynamics.Zenodo. [DOI — добавить после загрузки]
License
Apache 2.0 — see LICENSE file.Philosophical concept © Igor V. Kaminskyi (2024–2025).
