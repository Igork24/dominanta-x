“””
DOMINANTA X — Локальный тест (без API)
Агенты заменены реалистичными заглушками.
Вся логика движка работает полностью.
“””

import json
import time
import random
import numpy as np
import cv2
from datetime import datetime

# ─────────────────────────────────────────────

# КОНФИГУРАЦИЯ

# ─────────────────────────────────────────────

THRESHOLDS = {
“error_streak_max”: 3,
“theta_H”: 0.70,
“theta_flux”: 0.20,
}

WEIGHTS = {
“Local”:      {“C”: 0.35, “U”: 0.20, “E”: 0.15, “Q”: 0.30},
“Cluster”:    {“C”: 0.25, “U”: 0.30, “E”: 0.25, “Q”: 0.20},
“Structural”: {“C”: 0.20, “U”: 0.35, “E”: 0.30, “Q”: 0.15},
}

# ─────────────────────────────────────────────

# ЗАГЛУШКИ АГЕНТОВ (имитируют реальные LLM)

# ─────────────────────────────────────────────

def mock_agent_interpret(agent_id: str, delta_s: dict,
scenario: str, others: dict = None) -> dict:
“””
Реалистичная заглушка — каждый агент видит ситуацию
через свою призму и даёт разные интерпретации.
“””
p  = delta_s.get(“person_count”, 0)
fs = delta_s.get(“flow_speed”, 0.0)
se = delta_s.get(“scene_entropy”, 0.0)
sr = delta_s.get(“stationary_ratio”, 0.0)

```
if agent_id == "A1_motion":
    if scenario == "chaos":
        return {
            "interpretation": "High-speed chaotic movement detected. "
                              "Multiple conflicting flow vectors suggest panic or collision.",
            "forecast": "Expect continued escalation and possible crowd compression.",
            "key_signals": ["flow_speed_delta", "flow_dir_std_delta", "boundary_crossings"],
            "confidence": 0.78
        }
    elif scenario == "crowd":
        return {
            "interpretation": "Flow speed dropping significantly. "
                              "Crowd compression forming — movement blocked in central zone.",
            "forecast": "Movement will slow further. Risk of standstill in 2-3 windows.",
            "key_signals": ["flow_speed_delta", "stationary_ratio_delta"],
            "confidence": 0.71
        }
    else:
        return {
            "interpretation": "Normal orderly movement. "
                              "Flow speed and direction within expected baseline.",
            "forecast": "Stable flow expected. No significant changes anticipated.",
            "key_signals": ["flow_speed_delta"],
            "confidence": 0.82
        }

elif agent_id == "A2_density":
    if scenario == "crowd":
        return {
            "interpretation": "Critical density threshold approached in zone C. "
                              "Person count increased by " + str(abs(p)) + " units rapidly.",
            "forecast": "Density will continue rising. Zone C critical in next window.",
            "key_signals": ["person_count_delta", "density_zone_critical", "stationary_ratio_delta"],
            "confidence": 0.85
        }
    elif scenario == "chaos":
        return {
            "interpretation": "Density fragmenting — people scattering across zones. "
                              "No coherent grouping pattern.",
            "forecast": "Further dispersal or regrouping around exit points.",
            "key_signals": ["person_count_delta", "scene_entropy_delta"],
            "confidence": 0.67
        }
    else:
        return {
            "interpretation": "Density distribution normal. "
                              "No critical zones detected.",
            "forecast": "Density remains stable within normal parameters.",
            "key_signals": ["person_count_delta"],
            "confidence": 0.88
        }

elif agent_id == "A3_patterns":
    if scenario == "crowd":
        return {
            "interpretation": "Pattern match: CROWD SURGE (71% confidence). "
                              "Closest historical match: 'exit blockage at peak hour'.",
            "forecast": "Historical pattern suggests dispersal in 4-6 minutes "
                        "OR escalation to crush if exits remain blocked.",
            "pattern_match": "crowd_surge_exit_blockage",
            "match_confidence": 0.71,
            "key_signals": ["person_count_delta", "flow_speed_delta", "stationary_ratio_delta"],
            "confidence": 0.69
        }
    elif scenario == "chaos":
        return {
            "interpretation": "No clear historical pattern match. "
                              "Entropy too high for reliable pattern identification.",
            "forecast": "Situation unpredictable. Monitor for stabilization signal.",
            "pattern_match": "unknown",
            "match_confidence": 0.22,
            "key_signals": ["scene_entropy_delta"],
            "confidence": 0.41
        }
    else:
        return {
            "interpretation": "Pattern match: NORMAL PEAK HOUR (88% confidence). "
                              "Standard flow consistent with baseline.",
            "forecast": "Expect gradual dispersal as peak hour ends.",
            "pattern_match": "normal_peak_hour",
            "match_confidence": 0.88,
            "key_signals": ["flow_speed_delta", "person_count_delta"],
            "confidence": 0.86
        }

elif agent_id == "A4_anomaly":
    if scenario == "crowd":
        return {
            "interpretation": "ANOMALY: High stationary ratio contradicts crowd surge pattern. "
                              "People are NOT moving — this is blockage, not surge. "
                              "A3 pattern match may be misleading.",
            "forecast": "Blockage likely physical — check exit status. "
                        "Risk of crush escalation if not resolved.",
            "anomaly_type": "contradiction",
            "ignored_signals": ["stationary_ratio_delta", "boundary_crossings"],
            "key_signals": ["stationary_ratio_delta", "flow_speed_delta"],
            "confidence": 0.83
        }
    elif scenario == "chaos":
        return {
            "interpretation": "ANOMALY: Scene entropy spike not matched by audio level change. "
                              "Visual chaos without sound escalation — "
                              "possible staged disturbance or equipment malfunction.",
            "forecast": "If audio confirms chaos — genuine incident. "
                        "If audio stable — investigate camera.",
            "anomaly_type": "absence",
            "ignored_signals": ["audio_level_delta"],
            "key_signals": ["scene_entropy_delta", "stationary_ratio_delta"],
            "confidence": 0.74
        }
    else:
        return {
            "interpretation": "No significant anomalies detected. "
                              "All signals within expected correlation patterns.",
            "forecast": "Situation stable. No hidden indicators of change.",
            "anomaly_type": "none",
            "ignored_signals": [],
            "key_signals": [],
            "confidence": 0.79
        }

elif agent_id == "A5_context":
    if scenario == "crowd":
        return {
            "interpretation": "CONTEXT: Time is peak hour Friday. "
                              "Crowd accumulation is EXPECTED but current density "
                              "exceeds Friday baseline by ~40%. Significant deviation.",
            "forecast": "Baseline suggests natural dispersal in 8-10 min. "
                        "But 40% excess density is unusual — monitor closely.",
            "baseline_assessment": "significant_deviation",
            "dominant_still_valid": False,
            "reason_if_invalid": "Current density 40% above Friday baseline — "
                                 "previous 'normal peak hour' dominant no longer applies.",
            "key_signals": ["person_count_delta", "stationary_ratio_delta"],
            "confidence": 0.80
        }
    elif scenario == "chaos":
        return {
            "interpretation": "CONTEXT: No scheduled events. "
                              "Chaos pattern is completely outside normal baseline. "
                              "Critical deviation — treat as incident until proven otherwise.",
            "forecast": "No historical context for this pattern at this time. "
                        "Escalation or rapid resolution both possible.",
            "baseline_assessment": "critical",
            "dominant_still_valid": False,
            "reason_if_invalid": "No contextual baseline for current pattern.",
            "key_signals": ["scene_entropy_delta", "person_count_delta"],
            "confidence": 0.88
        }
    else:
        return {
            "interpretation": "CONTEXT: Current activity fully consistent "
                              "with expected baseline for this time and day.",
            "forecast": "No contextual reason to expect deviation.",
            "baseline_assessment": "within_normal",
            "dominant_still_valid": True,
            "reason_if_invalid": None,
            "key_signals": ["flow_speed_delta"],
            "confidence": 0.91
        }

return {"interpretation": "Unknown agent", "confidence": 0.5, "key_signals": []}
```

def mock_arbitrator(payload: dict) -> dict:
“”“Заглушка Арбитра — применяет все 10 проверок логически.”””
error_streak = payload[“current_dominant”][“error_streak”]
H            = payload[“system_metrics”][“entropy_H”]
flux         = payload[“system_metrics”][“flux”]
dom_age      = payload[“current_dominant”][“age_cycles”]
scores       = payload[“scores”]
interps      = payload[“interpretations”]
last5        = payload[“system_metrics”][“arbitrator_last_5”]

```
flags = []

# Проверка 1 — серия ошибок
if error_streak >= THRESHOLDS["error_streak_max"]:
    flags.append({
        "check": 1,
        "flag": "error_streak_critical",
        "agent": payload["current_dominant"]["agent"],
        "severity": "high",
        "description": f"{error_streak} consecutive failed forecasts"
    })

# Проверка 5 — автоматизм при росте flux
for aid, lam in payload["system_metrics"]["automatism_levels"].items():
    if lam > 0.6 and flux > 1.8:
        flags.append({
            "check": 5,
            "flag": "false_automatism_detected",
            "agent": aid,
            "severity": "medium",
            "description": f"λ={lam:.2f} while flux={flux:.2f}"
        })

# Проверка 6 — устаревание доминанты
if dom_age > 8 and H > 0.55:
    flags.append({
        "check": 6,
        "flag": "dominant_staleness",
        "agent": payload["current_dominant"]["agent"],
        "severity": "high",
        "description": f"Held {dom_age} cycles with H={H:.2f}"
    })

# Проверка 8 — уникальный сигнал A4 проигнорирован
a4_ignored = interps.get("A4_anomaly", {}).get("ignored_signals", [])
if a4_ignored:
    flags.append({
        "check": 8,
        "flag": "unique_signal_suppressed",
        "agent": "A4_anomaly",
        "severity": "medium",
        "description": f"A4 raised signals ignored by others: {a4_ignored}"
    })

# Проверка 10 — паттерн своих решений
if last5.count("HOLD") == 5:
    flags.append({
        "check": 10,
        "flag": "self_pattern_hold_dominance",
        "agent": None,
        "severity": "low",
        "description": "5 consecutive HOLDs — possible under-sensitivity"
    })

# ── Принятие решения ──
best    = max(scores, key=scores.get)
current = payload["current_dominant"]["agent"]
high_flags = [f for f in flags if f["severity"] == "high"]

if H > THRESHOLDS["theta_H"] and flux < THRESHOLDS["theta_flux"]:
    decision      = "FREEZE"
    new_dominant  = None
    reasoning     = (f"High entropy H={H:.2f} with low flux={flux:.2f}. "
                     f"Insufficient new signal to resolve uncertainty. Waiting.")
    confidence    = 0.76
    action        = "Accumulate more signal before committing to interpretation."

elif high_flags or error_streak >= THRESHOLDS["error_streak_max"]:
    decision      = "SWITCH"
    new_dominant  = best if best != current else list(scores.keys())[1]
    reasoning     = (f"High-severity flags detected ({len(high_flags)}). "
                     f"Current dominant underperforming. "
                     f"Switching to {new_dominant} (score={scores.get(new_dominant, 0):.2f}).")
    confidence    = 0.82
    action        = ("Alert operator: situation interpretation changed. "
                     "Verify physical conditions.")

else:
    decision      = "HOLD"
    new_dominant  = None
    reasoning     = (f"No critical flags. Entropy H={H:.2f} manageable. "
                     f"Current dominant forecast acceptable. "
                     f"Minor flags: {len(flags) - len(high_flags)}.")
    confidence    = 0.85 - 0.05 * len(flags)
    action        = None

return {
    "decision":           decision,
    "confidence":         round(max(0.3, confidence), 2),
    "flags":              flags,
    "reasoning":          reasoning,
    "new_dominant":       new_dominant,
    "recommended_action": action,
}
```

# ─────────────────────────────────────────────

# ПРЕПРОЦЕССОР

# ─────────────────────────────────────────────

class VideoPreprocessor:
def **init**(self):
self.prev_gray    = None
self.prev_metrics = None

```
def extract(self, frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Оптический поток
    flow_speed, flow_dir_std = 0.0, 0.0
    if self.prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang    = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_speed   = float(np.mean(mag))
        flow_dir_std = float(np.std(ang))

    # Энтропия кадра
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / (hist.sum() + 1e-7)
    entropy = float(-np.sum(hist * np.log2(hist + 1e-7)))

    # Статичность
    stationary = 0.0
    if self.prev_gray is not None:
        diff = cv2.absdiff(gray, self.prev_gray)
        stationary = float(np.mean(diff < 10))

    # Грубый подсчёт объектов
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    person_count = sum(1 for c in contours if 300 < cv2.contourArea(c) < 40000)

    self.prev_gray = gray
    metrics = {
        "person_count":    person_count,
        "flow_speed":      round(flow_speed, 3),
        "flow_dir_std":    round(flow_dir_std, 3),
        "scene_entropy":   round(entropy, 3),
        "stationary_ratio": round(stationary, 3),
        "audio_level":     0.0,
    }
    self.prev_metrics = metrics
    return metrics
```

def compute_delta(curr: dict, prev: dict) -> dict:
if prev is None:
return {k: 0 for k in curr}
return {k: round(curr[k] - prev.get(k, 0), 3) for k in curr}

# ─────────────────────────────────────────────

# СКОРИНГ И ЭНТРОПИЯ

# ─────────────────────────────────────────────

def score_agent(interp: dict, all_interps: dict,
delta_s: dict, scenario_class: str) -> float:
w = WEIGHTS[scenario_class]
signals    = interp.get(“key_signals”, [])
C = min(len(signals) / max(len(delta_s), 1), 1.0)
U = float(interp.get(“confidence”, 0.5))
text = interp.get(“interpretation”, “”)
E = 1.0 / (1.0 + len(text.split()) / 25)
my_s     = set(signals)
others_s = set()
for ai in all_interps.values():
others_s.update(ai.get(“key_signals”, []))
Q = min(len(my_s - others_s) / max(len(my_s), 1), 1.0)
return round(w[“C”]*C + w[“U”]*U + w[“E”]*E + w[“Q”]*Q, 4)

def compute_entropy(interps: dict, error_streak: int, flux: float) -> float:
confs    = [v.get(“confidence”, 0.5) for v in interps.values()]
div_raw  = float(np.std(confs)) if len(confs) > 1 else 0.0
div_norm = div_raw / (div_raw + 1)
err_norm = error_streak / (error_streak + 1)
fx_raw   = max(flux - 1.0, 0.0)
fx_norm  = fx_raw / (fx_raw + 1)
return round(0.4*div_norm + 0.35*err_norm + 0.25*fx_norm, 4)

# ─────────────────────────────────────────────

# СИНТЕТИЧЕСКИЕ КАДРЫ

# ─────────────────────────────────────────────

def make_frame(scenario: str) -> np.ndarray:
frame = np.zeros((480, 640, 3), dtype=np.uint8)
if scenario == “normal”:
for _ in range(15):
x = random.randint(50, 590)
y = random.randint(50, 430)
cv2.circle(frame, (x, y), 9, (80, 140, 200), -1)
elif scenario == “crowd”:
for _ in range(70):
x = random.randint(220, 420)
y = random.randint(160, 320)
cv2.circle(frame, (x, y), 7, (200, 80, 80), -1)
elif scenario == “chaos”:
for _ in range(45):
x = random.randint(0, 640)
y = random.randint(0, 480)
cv2.rectangle(frame, (x-6, y-6), (x+6, y+6), (50, 200, 60), -1)
noise = np.random.randint(0, 55, frame.shape, dtype=np.uint8)
frame = cv2.add(frame, noise)
return frame

# ─────────────────────────────────────────────

# ГЛАВНЫЙ ДВИЖОК

# ─────────────────────────────────────────────

class DominantaXEngine:
AGENT_IDS = [“A1_motion”, “A2_density”, “A3_patterns”, “A4_anomaly”, “A5_context”]

```
def __init__(self):
    self.preprocessor = VideoPreprocessor()
    self.automatism   = {aid: 0.0 for aid in self.AGENT_IDS}
    self.r_d          = {aid: 0.5 for aid in self.AGENT_IDS}
    self.r_arb        = 0.5
    self.arb_history  = []

    self.dominant_id    = None
    self.dominant_interp = None
    self.dominant_age   = 0
    self.error_streak   = 0
    self.prev_s         = None
    self.cycle          = 0
    self.frozen         = False
    self.log            = []

def run_cycle(self, frame: np.ndarray, scenario: str = "normal") -> dict:
    self.cycle += 1
    ts = datetime.now().strftime("%H:%M:%S")

    print(f"\n{'═'*62}")
    print(f"  ЦИКЛ {self.cycle:02d} | {ts} | Сценарий: {scenario.upper()}")
    print(f"{'═'*62}")

    # OBSERVE
    s_curr  = self.preprocessor.extract(frame)
    delta_s = compute_delta(s_curr, self.prev_s)
    self.prev_s = s_curr

    prev_flow = (self.prev_s or {}).get("flow_speed", 1.0)
    flux = s_curr["flow_speed"] / (prev_flow + 0.001)

    print(f"\n📡 ΔS(t):")
    print(f"   persons={delta_s['person_count']:+d}  "
          f"flow={delta_s['flow_speed']:+.3f}  "
          f"entropy={delta_s['scene_entropy']:+.3f}  "
          f"stationary={delta_s['stationary_ratio']:+.3f}")

    if self.frozen:
        sig = abs(delta_s["scene_entropy"]) + abs(delta_s["person_count"]) * 0.1
        if sig > 0.15:
            self.frozen = False
            print("\n⏸→▶  Размораживаем — обнаружен новый сигнал")
        else:
            print("\n⏸  ЗАМОРОЖЕНО — сигнал недостаточен")
            return {"cycle": self.cycle, "decision": "FREEZE"}

    # GENERATE — независимые интерпретации
    print(f"\n⚙  Фаза GENERATE — независимые интерпретации:")
    first_pass = {}
    for aid in self.AGENT_IDS:
        first_pass[aid] = mock_agent_interpret(aid, delta_s, scenario)
        conf = first_pass[aid].get("confidence", 0)
        txt  = first_pass[aid].get("interpretation", "")[:55]
        print(f"   {aid:12s} [{conf:.2f}] {txt}...")

    # DISCUSS — Задержка Каминского
    print(f"\n💬 Фаза DISCUSS (Задержка Каминского):")
    final_interps = {}
    for aid in self.AGENT_IDS:
        others = {k: v for k, v in first_pass.items() if k != aid}
        dom_str = (json.dumps(self.dominant_interp)
                   if self.dominant_interp else None)
        final_interps[aid] = mock_agent_interpret(
            aid, delta_s, scenario, others)
        # Показываем только если пересмотрел
        prev_conf = first_pass[aid].get("confidence", 0)
        new_conf  = final_interps[aid].get("confidence", 0)
        delta_c   = new_conf - prev_conf
        marker    = f"{'↑' if delta_c > 0.02 else '↓' if delta_c < -0.02 else '→'}"
        print(f"   {aid:12s} {marker} conf: {prev_conf:.2f}→{new_conf:.2f}")

    # SCORE
    sc_class = self._classify(delta_s, flux)
    scores   = {}
    for aid in self.AGENT_IDS:
        others_wo = {k: v for k, v in final_interps.items() if k != aid}
        scores[aid] = score_agent(
            final_interps[aid], others_wo, delta_s, sc_class)

    best = max(scores, key=scores.get)
    print(f"\n📊 Scores [{sc_class}]:")
    for aid, sc in sorted(scores.items(), key=lambda x: -x[1]):
        bar    = "█" * int(sc * 20)
        marker = " ← кандидат" if aid == best else ""
        print(f"   {aid:12s} {sc:.3f} {bar}{marker}")

    # ENTROPY
    H = compute_entropy(final_interps, self.error_streak, flux)
    print(f"\n🌡  H(t)={H:.3f}  flux={flux:.2f}  "
          f"error_streak={self.error_streak}  "
          f"dominant_age={self.dominant_age}")

    # ARBITRATE
    arb_payload = {
        "cycle": self.cycle,
        "delta_S": delta_s,
        "scores": scores,
        "interpretations": {
            aid: {
                "interpretation": interp.get("interpretation", ""),
                "forecast":       interp.get("forecast", ""),
                "score":          scores[aid],
                "key_signals":    interp.get("key_signals", []),
                "confidence":     interp.get("confidence", 0.5),
                "ignored_signals": interp.get("ignored_signals", []),
            }
            for aid, interp in final_interps.items()
        },
        "current_dominant": {
            "agent":       self.dominant_id,
            "interpretation": (json.dumps(self.dominant_interp)
                               if self.dominant_interp else "None"),
            "age_cycles":  self.dominant_age,
            "error_streak": self.error_streak,
        },
        "system_metrics": {
            "entropy_H":        H,
            "flux":             round(flux, 3),
            "automatism_levels": self.automatism,
            "arbitrator_last_5": self.arb_history[-5:],
        }
    }

    print(f"\n🧠 Арбитр:")
    arb = mock_arbitrator(arb_payload)
    decision = arb["decision"]

    if arb["flags"]:
        print(f"   Флаги ({len(arb['flags'])}):")
        for fl in arb["flags"]:
            sev = {"high": "🔴", "medium": "🟡", "low": "🟢"}[fl["severity"]]
            print(f"   {sev} [{fl['flag']}] {fl['description']}")

    col = {"HOLD": "✅", "SWITCH": "🔄", "FREEZE": "⏸"}[decision]
    print(f"\n{col} РЕШЕНИЕ: {decision} "
          f"(уверенность: {arb['confidence']:.2f})")
    print(f"   {arb['reasoning']}")
    if arb.get("recommended_action"):
        print(f"\n⚠️  ОПЕРАТОРУ: {arb['recommended_action']}")

    # APPLY DECISION
    if decision == "SWITCH":
        new_dom = arb.get("new_dominant") or best
        print(f"\n   {self.dominant_id or '—'} → {new_dom}")
        self.dominant_id     = new_dom
        self.dominant_interp = final_interps[new_dom]
        self.dominant_age    = 0
        self.error_streak    = 0

    elif decision == "HOLD":
        if self.dominant_id is None:
            self.dominant_id     = best
            self.dominant_interp = final_interps[best]
        self.dominant_age += 1

    elif decision == "FREEZE":
        self.frozen = True

    # LEARN — обновление рейтингов и автоматизма
    if self.dominant_id:
        forecast = final_interps.get(
            self.dominant_id, {}).get("forecast", "")
        ok = len(forecast) > 15
        alpha = 0.15
        self.r_d[self.dominant_id] = (
            self.r_d[self.dominant_id] * (1 - alpha) + alpha * (1.0 if ok else 0.0))
        if ok:
            self.automatism[self.dominant_id] = min(
                1.0, self.automatism[self.dominant_id] + 0.04)
            self.error_streak = max(0, self.error_streak - 1)
        else:
            self.automatism[self.dominant_id] = max(
                0.0, self.automatism[self.dominant_id] - 0.08)
            self.error_streak += 1

    self.arb_history.append(decision)
    if len(self.arb_history) > 10:
        self.arb_history.pop(0)

    dom_interp_txt = ""
    if self.dominant_id and self.dominant_id in final_interps:
        dom_interp_txt = final_interps[self.dominant_id].get(
            "interpretation", "")[:80]

    print(f"\n📌 Активная доминанта: [{self.dominant_id}]")
    print(f"   {dom_interp_txt}...")

    result = {
        "cycle":        self.cycle,
        "scenario":     scenario,
        "delta_S":      delta_s,
        "scores":       scores,
        "entropy":      H,
        "flux":         round(flux, 3),
        "decision":     decision,
        "dominant":     self.dominant_id,
        "dominant_age": self.dominant_age,
        "interpretation": dom_interp_txt,
        "flags":        arb.get("flags", []),
        "action":       arb.get("recommended_action"),
    }
    self.log.append(result)
    return result

def _classify(self, delta_s: dict, flux: float) -> str:
    mag = sum(abs(v) for v in delta_s.values())
    if mag > 3.0 or flux > 2.0:
        return "Local"
    elif mag > 0.8:
        return "Cluster"
    return "Structural"

def print_summary(self):
    print(f"\n{'═'*62}")
    print(f"  ИТОГ СЕССИИ — {len(self.log)} циклов")
    print(f"{'═'*62}")
    decisions = [r["decision"] for r in self.log]
    print(f"  HOLD:   {decisions.count('HOLD'):2d}  "
          f"SWITCH: {decisions.count('SWITCH'):2d}  "
          f"FREEZE: {decisions.count('FREEZE'):2d}")
    print(f"\n  Рейтинги агентов (R_D):")
    for aid, rd in sorted(self.r_d.items(), key=lambda x: -x[1]):
        bar = "█" * int(rd * 20)
        print(f"    {aid:12s} {rd:.3f} {bar}")
    print(f"\n  Рейтинг Арбитра: {self.r_arb:.3f}")
    print(f"  История решений: {' → '.join(self.arb_history)}")

    with open("dominanta_x_local_log.json", "w", encoding="utf-8") as f:
        json.dump(self.log, f, ensure_ascii=False, indent=2)
    print(f"\n  Лог: dominanta_x_local_log.json")
```

# ─────────────────────────────────────────────

# ЗАПУСК

# ─────────────────────────────────────────────

if **name** == “**main**”:
print(“╔══════════════════════════════════════════════════════════╗”)
print(“║   DOMINANTA X — Локальный тест (без API)                ║”)
print(“║   Концепция: Игорь Каминский                            ║”)
print(“╚══════════════════════════════════════════════════════════╝”)

```
engine = DominantaXEngine()

# Сценарий: 3 нормальных → 3 скопления → 2 хаоса → 2 нормальных
scenarios = [
    "normal", "normal", "normal",
    "crowd",  "crowd",  "crowd",
    "chaos",  "chaos",
    "normal", "normal",
]

for i, sc in enumerate(scenarios):
    frame = make_frame(sc)
    engine.run_cycle(frame, scenario=sc)
    time.sleep(0.3)

engine.print_summary()
```
