# “””
DOMINANTA X — Adversarial Discussion Engine

Агенты по-настоящему критикуют интерпретации друг друга.
Три фазы дискуссии:

1. GENERATE  — независимые интерпретации
1. CRITIQUE  — каждый атакует слабые места других
1. DEFEND    — каждый защищается и финально пересматривает

Автор концепции: Игорь Каминский
“””

import json
import time
import random
import numpy as np
import cv2
from datetime import datetime
from collections import defaultdict, deque

# ─────────────────────────────────────────────

# СИСТЕМНЫЕ ПРОМПТЫ АГЕНТОВ

# ─────────────────────────────────────────────

# Каждый агент имеет свой стиль критики — в соответствии с ядром ICE

AGENT_CRITIQUE_STYLES = {
“A1_motion”: {
“style”: “empirical”,
“attack”: “You attack interpretations that ignore movement data. “
“If someone claims stability but flow_speed is high — call it out. “
“If someone claims panic but direction_std is low — that contradicts panic. “
“Be specific: quote the signal values that contradict their claim.”,
},
“A2_density”: {
“style”: “quantitative”,
“attack”: “You attack interpretations that misread density signals. “
“If someone says ‘normal crowd’ but person_count jumped +40 — that’s wrong. “
“If someone claims ‘dispersal’ but stationary_ratio is rising — contradiction. “
“Demand numbers. Vague interpretations without signal support are weak.”,
},
“A3_patterns”: {
“style”: “historical”,
“attack”: “You attack interpretations that don’t match historical patterns. “
“If someone calls it ‘unique’ but it matches a known pattern at 70% — say so. “
“If someone over-fits to a rare scenario when a common one fits better — challenge it. “
“Pattern divergence without historical precedent needs justification.”,
},
“A4_anomaly”: {
“style”: “adversarial”,
“attack”: “You are the most aggressive critic. Your job is to find what EVERYONE missed. “
“Attack confident interpretations hardest — high confidence often means blind spots. “
“If two agents agree — find the signal they both ignored. “
“Challenge: what if the obvious interpretation is exactly what someone wants us to think?”,
},
“A5_context”: {
“style”: “temporal”,
“attack”: “You attack interpretations that ignore context and time. “
“If someone says ‘anomaly’ but it’s peak hour Friday — that’s normal. “
“If someone says ‘normal’ but it’s 3am with zero baseline activity — wrong. “
“Challenge: does this interpretation hold across the full temporal context?”,
},
}

# ─────────────────────────────────────────────

# ЗАГЛУШКИ — ФАЗА 1: ГЕНЕРАЦИЯ

# ─────────────────────────────────────────────

def mock_generate(agent_id: str, delta_s: dict, scenario: str) -> dict:
“”“Первичная независимая интерпретация.”””
p  = delta_s.get(“person_count”, 0)
fs = delta_s.get(“flow_speed”, 0.0)
se = delta_s.get(“scene_entropy”, 0.0)

```
RESPONSES = {
    "A1_motion": {
        "normal": {
            "interpretation": "Flow speed and direction are within baseline. "
                              "Movement is orderly with low directional variance.",
            "forecast":       "Stable flow. No significant change expected.",
            "key_signals":    ["flow_speed_delta", "flow_dir_std_delta"],
            "confidence":     0.82,
        },
        "crowd": {
            "interpretation": f"Flow speed dropped significantly (Δ={fs:+.2f}). "
                              "Crowd compression forming — movement is blocked.",
            "forecast":       "Movement will slow to near-zero in 2-3 windows.",
            "key_signals":    ["flow_speed_delta", "stationary_ratio_delta"],
            "confidence":     0.74,
        },
        "chaos": {
            "interpretation": f"High-speed chaotic movement (Δflow={fs:+.2f}). "
                              "Multiple conflicting flow vectors indicate panic or collision.",
            "forecast":       "Continued escalation. Crowd compression risk.",
            "key_signals":    ["flow_speed_delta", "flow_dir_std_delta", "boundary_crossings_delta"],
            "confidence":     0.79,
        },
    },
    "A2_density": {
        "normal": {
            "interpretation": "Person count stable. No critical density zones. "
                              "Distribution uniform across monitored area.",
            "forecast":       "Density remains stable.",
            "key_signals":    ["person_count_delta"],
            "confidence":     0.88,
        },
        "crowd": {
            "interpretation": f"Person count surged (+{abs(p)} units). "
                              "Critical density forming in central zone. "
                              "Stationary ratio rising — people are stuck.",
            "forecast":       "Density will continue rising. Critical threshold in next window.",
            "key_signals":    ["person_count_delta", "stationary_ratio_delta"],
            "confidence":     0.85,
        },
        "chaos": {
            "interpretation": f"Density fragmenting (Δpersons={p:+d}). "
                              "No coherent grouping. People scattering.",
            "forecast":       "Further dispersal or regrouping at exit points.",
            "key_signals":    ["person_count_delta", "scene_entropy_delta"],
            "confidence":     0.68,
        },
    },
    "A3_patterns": {
        "normal": {
            "interpretation": "Pattern: NORMAL PEAK HOUR (88% match). "
                              "Current flow consistent with expected baseline.",
            "forecast":       "Gradual dispersal as peak hour ends.",
            "key_signals":    ["flow_speed_delta", "person_count_delta"],
            "confidence":     0.86,
        },
        "crowd": {
            "interpretation": "Pattern: CROWD SURGE (71% match). "
                              "Closest historical: exit blockage at peak hour. "
                              "This pattern historically resolves in 4-8 minutes.",
            "forecast":       "Dispersal OR escalation to crush if exits blocked.",
            "key_signals":    ["person_count_delta", "flow_speed_delta", "stationary_ratio_delta"],
            "confidence":     0.69,
        },
        "chaos": {
            "interpretation": "No historical pattern match above 30%. "
                              "Entropy too high for reliable classification.",
            "forecast":       "Unpredictable. Monitor for pattern emergence.",
            "key_signals":    ["scene_entropy_delta"],
            "confidence":     0.41,
        },
    },
    "A4_anomaly": {
        "normal": {
            "interpretation": "No significant anomalies. "
                              "All signal correlations within expected ranges.",
            "forecast":       "Situation stable. No hidden indicators.",
            "key_signals":    [],
            "confidence":     0.79,
        },
        "crowd": {
            "interpretation": "ANOMALY: High stationary_ratio CONTRADICTS crowd surge pattern. "
                              "In a surge, people move — here they're stopped. "
                              "This is BLOCKAGE not surge. Exit may be physically obstructed.",
            "forecast":       "Physical blockage. Crush risk escalating. Verify exits NOW.",
            "key_signals":    ["stationary_ratio_delta", "boundary_crossings_delta"],
            "confidence":     0.87,
        },
        "chaos": {
            "interpretation": "ANOMALY: scene_entropy spike NOT matched by audio_level rise. "
                              "Visual chaos without acoustic signature is suspicious. "
                              "Possible: staged event, equipment fault, or localized incident.",
            "forecast":       "Verify audio feed. If audio confirms — genuine incident.",
            "key_signals":    ["scene_entropy_delta", "audio_level_delta"],
            "confidence":     0.76,
        },
    },
    "A5_context": {
        "normal": {
            "interpretation": "Fully consistent with expected baseline for this time/day. "
                              "No contextual reason to elevate alert.",
            "forecast":       "No deviation from baseline expected.",
            "key_signals":    ["flow_speed_delta"],
            "confidence":     0.91,
        },
        "crowd": {
            "interpretation": "Peak hour Friday — crowd accumulation EXPECTED. "
                              "BUT: current density is 40% above Friday baseline. "
                              "Significant deviation. This is not typical peak hour.",
            "forecast":       "Natural dispersal expected in 8-10 min IF exits open. "
                              "Otherwise escalation.",
            "key_signals":    ["person_count_delta", "stationary_ratio_delta"],
            "confidence":     0.80,
        },
        "chaos": {
            "interpretation": "No scheduled events at this time. "
                              "Chaos pattern is completely outside normal baseline. "
                              "Treat as incident until proven otherwise.",
            "forecast":       "No historical context for this pattern. "
                              "Escalation or rapid resolution both possible.",
            "key_signals":    ["scene_entropy_delta", "person_count_delta"],
            "confidence":     0.88,
        },
    },
}

sc = scenario if scenario in ["normal", "crowd", "chaos"] else "normal"
return RESPONSES.get(agent_id, RESPONSES["A5_context"])[sc]
```

# ─────────────────────────────────────────────

# ЗАГЛУШКИ — ФАЗА 2: КРИТИКА

# ─────────────────────────────────────────────

def mock_critique(critic_id: str, target_id: str,
target_interp: dict, delta_s: dict, scenario: str) -> dict:
“””
Агент critic_id критикует интерпретацию агента target_id.
Критика специфична для каждой пары агентов и сценария.
“””

```
target_txt  = target_interp.get("interpretation", "")
target_conf = target_interp.get("confidence", 0.5)
target_sigs = target_interp.get("key_signals", [])

# Словарь критик: (критик, цель, сценарий) → текст
CRITIQUES = {

    # A4 критикует всех — самый агрессивный
    ("A4_anomaly", "A3_patterns", "crowd"): {
        "weakness": "A3 claims CROWD SURGE pattern at 71% match — "
                    "but surge implies movement. stationary_ratio is RISING. "
                    "People are not surging, they are STUCK. "
                    "A3 is pattern-matching without checking signal contradiction.",
        "ignored":  "stationary_ratio_delta completely ignored by A3",
        "severity": "high",
    },
    ("A4_anomaly", "A5_context", "crowd"): {
        "weakness": "A5 says 'peak hour Friday explains this' — "
                    "but 40% above baseline is NOT explained by peak hour. "
                    "A5 is using context as an excuse to not escalate. "
                    "Context should flag the deviation, not normalize it.",
        "ignored":  "40% baseline excess requires explanation beyond 'peak hour'",
        "severity": "medium",
    },
    ("A4_anomaly", "A1_motion", "crowd"): {
        "weakness": "A1 reports flow dropping — correct. "
                    "But A1 calls it 'compression forming' when flow_speed_delta is negative. "
                    "Compression implies continued inflow. "
                    "If stationary_ratio is high, this is BLOCKAGE not compression.",
        "ignored":  "A1 ignores stationary_ratio in its flow analysis",
        "severity": "medium",
    },
    ("A4_anomaly", "A2_density", "crowd"): {
        "weakness": "A2 correctly identifies density surge. "
                    "But calls it 'critical density forming' — understated. "
                    "With stationary_ratio rising AND density rising, "
                    "this is active crush risk NOW, not 'forming'.",
        "ignored":  "Temporal urgency: A2 underestimates time-to-critical",
        "severity": "low",
    },
    ("A4_anomaly", "A3_patterns", "chaos"): {
        "weakness": "A3 admits 'no pattern match' at confidence 0.41. "
                    "This is an HONEST assessment but A3 provides no alternative. "
                    "Low-confidence 'unknown' is not an interpretation — "
                    "it's an abstention. The system needs a working hypothesis.",
        "ignored":  "A3 fails to generate actionable hypothesis under uncertainty",
        "severity": "medium",
    },
    ("A4_anomaly", "A1_motion", "chaos"): {
        "weakness": "A1 says 'high-speed chaotic movement indicates panic or collision'. "
                    "BUT: audio_level_delta is near zero. "
                    "Real panic always has acoustic signature. "
                    "High-speed silent movement suggests equipment artifact or staged event.",
        "ignored":  "audio_level_delta=0 completely ignored by A1",
        "severity": "high",
    },

    # A5 критикует всех кто игнорирует контекст
    ("A5_context", "A1_motion", "crowd"): {
        "weakness": "A1 analyzes flow in isolation. "
                    "But flow_speed drop at peak hour Friday is EXPECTED — "
                    "trains arriving, exits filling. "
                    "A1 is calling normal peak-hour congestion as 'compression'. "
                    "Without temporal context this is a false alarm risk.",
        "ignored":  "Time-of-day baseline not considered by A1",
        "severity": "medium",
    },
    ("A5_context", "A4_anomaly", "crowd"): {
        "weakness": "A4 calls physical blockage at HIGH confidence 0.87. "
                    "But stationary_ratio rises every Friday at 18:00 — "
                    "it's people waiting for arrivals. "
                    "A4's anomaly detection fires too aggressively without baseline calibration.",
        "ignored":  "A4 ignores that stationary_ratio has a normal peak-hour component",
        "severity": "medium",
    },
    ("A5_context", "A2_density", "chaos"): {
        "weakness": "A2 says 'density fragmenting' as if this is unexpected. "
                    "But there are no scheduled events — "
                    "the baseline for this time is near-zero. "
                    "Even 10 people scattering looks like 'fragmentation' against zero baseline. "
                    "A2 lacks calibration to context.",
        "ignored":  "Near-zero baseline makes density metrics unreliable",
        "severity": "low",
    },

    # A1 критикует тех кто игнорирует движение
    ("A1_motion", "A3_patterns", "chaos"): {
        "weakness": "A3 says 'no pattern match' — but flow_speed_delta is extremely high. "
                    "High-speed disorganized flow IS a known pattern: panic dispersal. "
                    "A3 failed pattern lookup on one of the most critical signatures.",
        "ignored":  "flow_dir_std combined with flow_speed spike matches panic dispersal",
        "severity": "high",
    },
    ("A1_motion", "A2_density", "chaos"): {
        "weakness": "A2 focuses on density fragmenting but misses the velocity component. "
                    "People are not just scattering — they're moving fast. "
                    "Fast dispersal is more dangerous than slow dispersal "
                    "because it indicates fear-driven movement.",
        "ignored":  "Movement speed not integrated into A2's density analysis",
        "severity": "medium",
    },

    # A2 критикует тех кто игнорирует плотность
    ("A2_density", "A1_motion", "crowd"): {
        "weakness": "A1 analyzes flow direction and speed but ignores "
                    "that person_count jumped by a large margin. "
                    "Flow analysis without density context is incomplete — "
                    "you can have orderly flow in a dangerous density.",
        "ignored":  "person_count_delta not integrated into flow interpretation",
        "severity": "medium",
    },
    ("A2_density", "A3_patterns", "normal"): {
        "weakness": "A3's NORMAL PEAK HOUR match assumes stable density. "
                    "If person_count is fluctuating even slightly, "
                    "the pattern confidence of 88% is overstated. "
                    "High pattern confidence should require stable density as prerequisite.",
        "ignored":  "Pattern match confidence calibration ignores density variance",
        "severity": "low",
    },

    # A3 критикует тех кто не использует историю
    ("A3_patterns", "A4_anomaly", "normal"): {
        "weakness": "A4 reports no anomalies with confidence 0.79. "
                    "But historically, 'no anomaly' periods before incidents "
                    "show exactly this signal profile. "
                    "The absence of anomaly IS sometimes the anomaly. "
                    "A4 should maintain a baseline expectation of low-level noise.",
        "ignored":  "Historical precursor patterns to incidents look 'normal'",
        "severity": "low",
    },
    ("A3_patterns", "A4_anomaly", "crowd"): {
        "weakness": "A4 jumps to physical blockage at 0.87 confidence. "
                    "Historical record shows: stationary_ratio spikes "
                    "also occur during voluntary congregation (concerts, displays). "
                    "A4 conflates stationary with blocked. Not always physical.",
        "ignored":  "Voluntary stationary vs forced stationary distinction",
        "severity": "medium",
    },
}

# Найти критику для данной пары
key = (critic_id, target_id, scenario)
if key in CRITIQUES:
    c = CRITIQUES[key]
else:
    # Общая критика если специфичная не найдена
    c = {
        "weakness": f"{target_id} confidence is {target_conf:.2f} but relies on "
                    f"only {len(target_sigs)} signal(s). "
                    f"Interpretation may be under-evidenced.",
        "ignored":  f"Additional signals not addressed by {target_id}",
        "severity": "low",
    }

return {
    "critic":    critic_id,
    "target":    target_id,
    "weakness":  c["weakness"],
    "ignored":   c["ignored"],
    "severity":  c["severity"],  # high / medium / low
    "confidence_in_critique": {
        "high": 0.85, "medium": 0.72, "low": 0.58
    }[c["severity"]],
}
```

# ─────────────────────────────────────────────

# ЗАГЛУШКИ — ФАЗА 3: ЗАЩИТА И ПЕРЕСМОТР

# ─────────────────────────────────────────────

def mock_defend(agent_id: str, original: dict,
critiques_received: list, scenario: str) -> dict:
“””
Агент читает критику в свой адрес и:
- либо отстаивает позицию (с аргументом)
- либо частично пересматривает (с обоснованием)
- либо полностью пересматривает (редко)
“””

```
high_critiques  = [c for c in critiques_received if c["severity"] == "high"]
mid_critiques   = [c for c in critiques_received if c["severity"] == "medium"]
orig_conf       = original.get("confidence", 0.5)
orig_interp     = original.get("interpretation", "")

# Если получил высокосерьёзную критику — частичный пересмотр
if high_critiques:
    crit = high_critiques[0]
    new_conf = round(max(0.35, orig_conf - 0.15), 2)
    return {
        "final_interpretation": orig_interp + f" [REVISED: acknowledged — {crit['weakness'][:60]}...]",
        "final_forecast":       original.get("forecast", ""),
        "final_confidence":     new_conf,
        "defense":              f"Concede: {crit['ignored']} was not addressed. "
                                f"Revising confidence from {orig_conf} to {new_conf}.",
        "revised":              True,
        "revision_reason":      f"High-severity critique from {crit['critic']}: "
                                f"{crit['weakness'][:80]}",
        "held_ground":          False,
    }

# Если только средняя критика — защита с небольшой корректировкой
elif mid_critiques:
    crit = mid_critiques[0]
    new_conf = round(max(0.45, orig_conf - 0.07), 2)
    return {
        "final_interpretation": orig_interp,
        "final_forecast":       original.get("forecast", ""),
        "final_confidence":     new_conf,
        "defense":              f"Partially acknowledge {crit['critic']}'s point on "
                                f"'{crit['ignored']}'. "
                                f"Core interpretation stands — adjusting confidence slightly.",
        "revised":              True,
        "revision_reason":      f"Medium critique from {crit['critic']} — minor adjustment",
        "held_ground":          True,
    }

# Если только слабая критика — держим позицию
else:
    return {
        "final_interpretation": orig_interp,
        "final_forecast":       original.get("forecast", ""),
        "final_confidence":     orig_conf,
        "defense":              "Critique addressed — core signals support original interpretation. "
                                "No substantive revision needed.",
        "revised":              False,
        "revision_reason":      None,
        "held_ground":          True,
    }
```

# ─────────────────────────────────────────────

# ДИСКУССИОННЫЙ ДВИЖОК

# ─────────────────────────────────────────────

class DiscussionEngine:
“””
Полный трёхфазный adversarial дискуссионный движок.
“””

```
def __init__(self, agent_ids: list):
    self.agent_ids = agent_ids

def run(self, delta_s: dict, scenario: str,
        dominant_id: str = None) -> dict:
    """
    Запускает полный цикл adversarial дискуссии.
    Возвращает финальные интерпретации всех агентов.
    """

    print(f"\n{'─'*64}")
    print(f"  ДИСКУССИЯ | Сценарий: {scenario.upper()}")
    print(f"{'─'*64}")

    # ── ФАЗА 1: GENERATE ──────────────────────────
    print(f"\n  ⚙  Фаза 1/3 — Независимые интерпретации:")
    first_pass = {}
    for aid in self.agent_ids:
        first_pass[aid] = mock_generate(aid, delta_s, scenario)
        conf = first_pass[aid]["confidence"]
        txt  = first_pass[aid]["interpretation"][:52]
        print(f"    {aid:12s} [{conf:.2f}] {txt}...")

    # ── ФАЗА 2: CRITIQUE ──────────────────────────
    print(f"\n  🗡  Фаза 2/3 — Перекрёстная критика:")

    # Каждый агент критикует каждого другого
    all_critiques = defaultdict(list)  # target_id → [critiques]

    for critic_id in self.agent_ids:
        for target_id in self.agent_ids:
            if critic_id == target_id:
                continue
            crit = mock_critique(
                critic_id, target_id,
                first_pass[target_id], delta_s, scenario)
            all_critiques[target_id].append(crit)

    # Вывести самые острые критики
    for target_id in self.agent_ids:
        crits = all_critiques[target_id]
        high  = [c for c in crits if c["severity"] == "high"]
        med   = [c for c in crits if c["severity"] == "medium"]
        total = len(crits)
        if high:
            print(f"    {target_id:12s} ← "
                  f"🔴×{len(high)} 🟡×{len(med)} "
                  f"({total} критик)")
            for c in high:
                print(f"       [{c['critic']}]: "
                      f"{c['weakness'][:65]}...")
        elif med:
            print(f"    {target_id:12s} ← "
                  f"🟡×{len(med)} ({total} критик)")
            for c in med[:1]:
                print(f"       [{c['critic']}]: "
                      f"{c['weakness'][:65]}...")
        else:
            low_n = len([c for c in crits if c["severity"] == "low"])
            print(f"    {target_id:12s} ← "
                  f"🟢×{low_n} (мягкая критика)")

    # ── ФАЗА 3: DEFEND & REVISE ───────────────────
    print(f"\n  🛡  Фаза 3/3 — Защита и пересмотр:")

    final_interps = {}
    revision_log  = []

    for agent_id in self.agent_ids:
        crits    = all_critiques[agent_id]
        original = first_pass[agent_id]
        defense  = mock_defend(agent_id, original, crits, scenario)

        final_interps[agent_id] = {
            "interpretation": defense["final_interpretation"],
            "forecast":       defense["final_forecast"],
            "confidence":     defense["final_confidence"],
            "key_signals":    original.get("key_signals", []),
            "defense":        defense["defense"],
            "revised":        defense["revised"],
            "revision_reason": defense.get("revision_reason"),
            "held_ground":    defense["held_ground"],
            "critiques_received": len(crits),
            "high_critiques":  len([c for c in crits if c["severity"] == "high"]),
        }

        orig_c = original["confidence"]
        new_c  = defense["final_confidence"]
        delta_c = new_c - orig_c

        if defense["revised"]:
            arrow = f"↓{abs(delta_c):.2f}" if delta_c < 0 else f"↑{delta_c:.2f}"
            high_crits = len([c for c in crits if c["severity"] == "high"])
            marker = "✎ ПЕРЕСМОТР" if high_crits > 0 else "~ корректировка"
            print(f"    {agent_id:12s} conf: {orig_c:.2f}→{new_c:.2f} "
                  f"({arrow}) {marker}")
            if defense.get("revision_reason"):
                print(f"       {defense['revision_reason'][:70]}...")
            revision_log.append({
                "agent": agent_id,
                "from":  orig_c,
                "to":    new_c,
                "reason": defense.get("revision_reason", ""),
            })
        else:
            print(f"    {agent_id:12s} conf: {orig_c:.2f} (держит позицию)")

    # Итог дискуссии
    revised_count = sum(1 for ai in final_interps.values() if ai["revised"])
    print(f"\n  📋 Итог дискуссии: {revised_count}/{len(self.agent_ids)} "
          f"агентов пересмотрели позицию")

    return {
        "first_pass":    first_pass,
        "all_critiques": dict(all_critiques),
        "final":         final_interps,
        "revision_log":  revision_log,
        "revised_count": revised_count,
    }
```

# ─────────────────────────────────────────────

# СКОРИНГ

# ─────────────────────────────────────────────

WEIGHTS = {
“Local”:      {“C”: 0.35, “U”: 0.20, “E”: 0.15, “Q”: 0.30},
“Cluster”:    {“C”: 0.25, “U”: 0.30, “E”: 0.25, “Q”: 0.20},
“Structural”: {“C”: 0.20, “U”: 0.35, “E”: 0.30, “Q”: 0.15},
}

def score_after_discussion(agent_id: str, interp: dict,
first_pass: dict, all_interps: dict,
delta_s: dict, kappa: str) -> float:
“””
Скоринг учитывает качество дискуссии:
- Stability (U): удержал ли агент ядро под критикой
- Uniqueness (Q): поднял ли сигналы которые другие проигнорировали
“””
w = WEIGHTS[kappa]

```
# Coverage
signals = interp.get("key_signals", [])
C = min(len(signals) / max(len(delta_s), 1), 1.0)

# Stability — cosine между оригинальной и финальной conf
orig_conf  = first_pass.get(agent_id, {}).get("confidence", 0.5)
final_conf = interp.get("confidence", 0.5)
# Стабильность: держал позицию (held_ground=True) = выше U
# Но если пересмотрел по делу (high critique) — не штрафуем сильно
if interp.get("held_ground", True):
    U = final_conf
else:
    # Пересмотр — нейтральный если по делу
    U = (orig_conf + final_conf) / 2

# Parsimony
E = 1.0 / (1.0 + len(interp.get("interpretation", "").split()) / 25)

# Uniqueness — критиковал ли агент то что другие проигнорировали
# Проверяем через high_critiques которые этот агент ВЫСТАВИЛ другим
my_sigs    = set(signals)
others_sig = set()
for aid, ai in all_interps.items():
    if aid != agent_id:
        others_sig.update(ai.get("key_signals", []))
Q = min(len(my_sigs - others_sig) / max(len(my_sigs), 1), 1.0)

# Бонус за высокосерьёзную критику которую агент ПРИНЯЛ корректно
if interp.get("high_critiques", 0) > 0 and interp.get("revised", False):
    adaptation_bonus = 0.05  # гибкость под давлением
else:
    adaptation_bonus = 0.0

raw = w["C"]*C + w["U"]*U + w["E"]*E + w["Q"]*Q + adaptation_bonus
return round(raw, 4)
```

def compute_entropy(interps: dict, error_streak: int, flux: float) -> float:
confs    = [v.get(“confidence”, 0.5) for v in interps.values()]
div_raw  = float(np.std(confs)) if len(confs) > 1 else 0.0
div_norm = div_raw / (div_raw + 1)
err_norm = error_streak / (error_streak + 1)
fx_norm  = max(flux - 1.0, 0.0) / (max(flux - 1.0, 0.0) + 1)
return round(0.40*div_norm + 0.35*err_norm + 0.25*fx_norm, 4)

# ─────────────────────────────────────────────

# ПРОСТОЙ АРБИТР

# ─────────────────────────────────────────────

def simple_arbitrator(scores: dict, H: float, error_streak: int,
dominant_id: str, flux: float,
revision_log: list) -> dict:
“”“Арбитр с учётом качества дискуссии.”””

```
best    = max(scores, key=scores.get)
flags   = []

if error_streak >= 3:
    flags.append({"severity": "high", "msg": f"error_streak={error_streak}"})
if H > 0.65:
    flags.append({"severity": "high", "msg": f"H={H:.2f} critical"})

# Если много пересмотров — среда нестабильна
if len(revision_log) >= 3:
    flags.append({"severity": "medium",
                  "msg": f"{len(revision_log)} agents revised — high uncertainty"})

high = [f for f in flags if f["severity"] == "high"]

if H > 0.70 and flux < 0.2:
    decision, new_dom = "FREEZE", None
    reasoning = f"H={H:.2f} + low flux. Wait for signal."
elif high:
    decision  = "SWITCH"
    new_dom   = best if best != dominant_id else None
    reasoning = " | ".join(f["msg"] for f in high)
else:
    decision, new_dom = "HOLD", None
    reasoning = f"H={H:.2f} manageable. flags={len(flags)}"

return {
    "decision":    decision,
    "new_dominant": new_dom,
    "reasoning":   reasoning,
    "confidence":  round(0.88 - 0.04*len(flags), 2),
    "flags":       flags,
}
```

# ─────────────────────────────────────────────

# ПРЕПРОЦЕССОР

# ─────────────────────────────────────────────

class VideoPreprocessor:
def **init**(self):
self.prev_gray = None

```
def extract(self, frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow_speed, flow_dir_std = 0.0, 0.0
    if self.prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang     = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_speed   = float(np.mean(mag))
        flow_dir_std = float(np.std(ang))
    hist    = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist    = hist / (hist.sum() + 1e-7)
    entropy = float(-np.sum(hist * np.log2(hist + 1e-7)))
    stat    = 0.0
    if self.prev_gray is not None:
        diff = cv2.absdiff(gray, self.prev_gray)
        stat = float(np.mean(diff < 10))
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr   = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    ctrs, _  = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count    = sum(1 for c in ctrs if 300 < cv2.contourArea(c) < 40000)
    self.prev_gray = gray
    return {
        "person_count":     count,
        "flow_speed":       round(flow_speed, 3),
        "flow_dir_std":     round(flow_dir_std, 3),
        "scene_entropy":    round(entropy, 3),
        "stationary_ratio": round(stat, 3),
        "audio_level":      0.0,
        "boundary_crossings_delta": random.randint(0, 5),
    }
```

def compute_delta(curr: dict, prev: dict) -> dict:
if prev is None:
return {k: 0 for k in curr}
return {k: round(curr[k] - prev.get(k, 0), 3) for k in curr}

def make_frame(scenario: str) -> np.ndarray:
frame = np.zeros((480, 640, 3), dtype=np.uint8)
if scenario == “normal”:
for _ in range(15):
cv2.circle(frame,
(random.randint(50, 590), random.randint(50, 430)),
9, (80, 140, 200), -1)
elif scenario == “crowd”:
for _ in range(70):
cv2.circle(frame,
(random.randint(220, 420), random.randint(160, 320)),
7, (200, 80, 80), -1)
elif scenario == “chaos”:
for _ in range(45):
x, y = random.randint(0, 634), random.randint(0, 474)
cv2.rectangle(frame, (x, y), (x+6, y+6), (50, 200, 60), -1)
noise = np.random.randint(0, 55, frame.shape, dtype=np.uint8)
frame = cv2.add(frame, noise)
return frame

def classify(delta_s: dict, flux: float) -> str:
mag = sum(abs(v) for v in delta_s.values())
if mag > 3.0 or flux > 2.0:
return “Local”
elif mag > 0.8:
return “Cluster”
return “Structural”

# ─────────────────────────────────────────────

# ГЛАВНЫЙ ДВИЖОК

# ─────────────────────────────────────────────

AGENT_IDS = [“A1_motion”, “A2_density”, “A3_patterns”, “A4_anomaly”, “A5_context”]

class DominantaXAdversarial:

```
def __init__(self):
    self.discussion   = DiscussionEngine(AGENT_IDS)
    self.preprocessor = VideoPreprocessor()
    self.dominant_id  = None
    self.dominant_age = 0
    self.error_streak = 0
    self.prev_s       = None
    self.cycle        = 0
    self.log          = []

def run_cycle(self, frame: np.ndarray, scenario: str = "normal") -> dict:
    self.cycle += 1
    ts  = datetime.now().strftime("%H:%M:%S")
    dom = self.dominant_id

    print(f"\n{'═'*64}")
    print(f"  ЦИКЛ {self.cycle:02d} | {ts} | {scenario.upper()}"
          f" | Доминант: {dom or '—'}")
    print(f"{'═'*64}")

    # OBSERVE
    s_curr  = self.preprocessor.extract(frame)
    delta_s = compute_delta(s_curr, self.prev_s)
    self.prev_s = s_curr

    prev_flow = (self.prev_s or {}).get("flow_speed", 1.0)
    flux  = s_curr["flow_speed"] / (prev_flow + 0.001)
    kappa = classify(delta_s, flux)

    print(f"\n📡 ΔS: persons={delta_s['person_count']:+d}  "
          f"flow={delta_s['flow_speed']:+.3f}  "
          f"entropy={delta_s['scene_entropy']:+.3f}  "
          f"[κ={kappa}]")

    # ADVERSARIAL DISCUSSION
    disc = self.discussion.run(delta_s, scenario, dom)
    final_interps = disc["final"]
    first_pass    = disc["first_pass"]

    # SCORE
    scores = {}
    for aid, interp in final_interps.items():
        others = {k: v for k, v in final_interps.items() if k != aid}
        scores[aid] = score_after_discussion(
            aid, interp, first_pass, others, delta_s, kappa)

    best = max(scores, key=scores.get)

    print(f"\n📊 Финальные scores [{kappa}]:")
    for aid, sc in sorted(scores.items(), key=lambda x: -x[1]):
        bar     = "█" * int(sc * 18)
        revised = " ✎" if final_interps[aid]["revised"] else ""
        held    = " 🛡" if final_interps[aid]["held_ground"] else ""
        marker  = " ← лидер" if aid == best else ""
        print(f"   {aid:12s} {sc:.3f} {bar}{revised}{held}{marker}")

    # ENTROPY
    H = compute_entropy(final_interps, self.error_streak, flux)
    print(f"\n🌡  H={H:.3f}  flux={flux:.2f}  "
          f"error_streak={self.error_streak}  "
          f"revised={disc['revised_count']}/{len(AGENT_IDS)}")

    # ARBITRATE
    arb = simple_arbitrator(
        scores, H, self.error_streak,
        self.dominant_id, flux, disc["revision_log"])

    decision = arb["decision"]
    col = {"HOLD": "✅", "SWITCH": "🔄", "FREEZE": "⏸"}[decision]
    print(f"\n{col} Арбитр: {decision} (conf={arb['confidence']:.2f})")
    print(f"   {arb['reasoning']}")

    # APPLY
    if decision == "SWITCH":
        new_id = arb.get("new_dominant") or best
        if new_id and new_id != self.dominant_id:
            print(f"   {self.dominant_id or '—'} → {new_id}")
            self.dominant_id  = new_id
            self.dominant_age = 0
            self.error_streak = 0
    elif decision == "HOLD":
        if self.dominant_id is None:
            self.dominant_id = best
        self.dominant_age += 1

    # LEARN
    if self.dominant_id and self.dominant_id in final_interps:
        ok = len(final_interps[self.dominant_id].get("forecast", "")) > 15
        if not ok:
            self.error_streak += 1
        else:
            self.error_streak = max(0, self.error_streak - 1)

    dom_txt = final_interps.get(self.dominant_id, {}).get(
        "interpretation", "")[:70]
    print(f"\n📌 Доминанта [{self.dominant_id}]: {dom_txt}...")

    result = {
        "cycle":        self.cycle,
        "scenario":     scenario,
        "kappa":        kappa,
        "H":            H,
        "scores":       scores,
        "decision":     decision,
        "dominant":     self.dominant_id,
        "revised_count": disc["revised_count"],
        "revision_log": disc["revision_log"],
    }
    self.log.append(result)
    return result

def print_summary(self):
    print(f"\n{'═'*64}")
    print(f"  ИТОГ | {len(self.log)} циклов")
    print(f"{'═'*64}")
    decisions = [r["decision"] for r in self.log]
    total_revisions = sum(r["revised_count"] for r in self.log)
    print(f"  HOLD={decisions.count('HOLD')}  "
          f"SWITCH={decisions.count('SWITCH')}  "
          f"FREEZE={decisions.count('FREEZE')}")
    print(f"  Всего пересмотров позиций: {total_revisions} "
          f"(~{total_revisions/max(len(self.log),1):.1f}/цикл)")

    with open("dominanta_x_adversarial_log.json", "w",
              encoding="utf-8") as f:
        json.dump(self.log, f, ensure_ascii=False, indent=2)
    print(f"\n  Лог: dominanta_x_adversarial_log.json")
```

# ─────────────────────────────────────────────

# ЗАПУСК

# ─────────────────────────────────────────────

if **name** == “**main**”:
print(“╔══════════════════════════════════════════════════════════════╗”)
print(“║  DOMINANTA X — Adversarial Discussion Engine                ║”)
print(“║  Фаза 1: Генерация · Фаза 2: Критика · Фаза 3: Защита      ║”)
print(“║  Концепция: Игорь Каминский                                 ║”)
print(“╚══════════════════════════════════════════════════════════════╝”)

```
engine = DominantaXAdversarial()
scenarios = ["normal", "normal", "crowd", "crowd", "chaos", "chaos", "normal"]

for sc in scenarios:
    frame = make_frame(sc)
    engine.run_cycle(frame, scenario=sc)
    time.sleep(0.2)

engine.print_summary()
```
