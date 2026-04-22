# “””
DOMINANTA X — Саморегулирующийся движок

Три уровня самокоррекции:
L1 — Реактивный: серия ошибок → SWITCH
L2 — Превентивный: тренд R_D → снижение веса агента
L3 — Структурный: хроническая слабость в классе → исключение из пула

Плюс самокоррекция Арбитра:
— Если R_arb падает → автоперекалибровка порогов
— Если паттерн решений аномальный → сброс к базовым порогам

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

# КОНФИГУРАЦИЯ

# ─────────────────────────────────────────────

WEIGHTS = {
“Local”:      {“C”: 0.35, “U”: 0.20, “E”: 0.15, “Q”: 0.30},
“Cluster”:    {“C”: 0.25, “U”: 0.30, “E”: 0.25, “Q”: 0.20},
“Structural”: {“C”: 0.20, “U”: 0.35, “E”: 0.30, “Q”: 0.15},
}

# Базовые пороги — Арбитр может их сдвигать автоматически

BASE_THRESHOLDS = {
“error_streak_max”: 3,    # серия ошибок для SWITCH
“theta_H”:          0.70, # порог энтропии для FREEZE
“theta_flux”:       0.20, # порог потока для FREEZE
“r_d_min”:          0.35, # минимальный рейтинг агента
“r_d_trend_window”: 5,    # окно для анализа тренда R_D
“class_ban_threshold”: 0.30,  # R_D в классе ниже этого → исключить
“class_ban_min_cycles”: 8,    # минимум циклов в классе до бана
“arb_recalibrate_threshold”: 0.40,  # R_arb ниже → перекалибровка
}

# ─────────────────────────────────────────────

# СТАТИСТИКА АГЕНТА

# ─────────────────────────────────────────────

class AgentStats:
“””
Полная статистика агента — глобальная и по классам сценариев.
Основа для всех трёх уровней самокоррекции.
“””

```
def __init__(self, agent_id: str):
    self.id           = agent_id
    self.r_d          = 0.50          # глобальный рейтинг
    self.automatism   = 0.00          # λᵢ
    self.active       = True          # включён в пул

    # История R_D для анализа тренда (скользящее окно)
    self.r_d_history  = deque(maxlen=10)

    # Статистика по классам: κ → {wins, total, r_d}
    self.class_stats  = defaultdict(lambda: {
        "wins": 0, "total": 0, "r_d": 0.50, "banned": False
    })

    # История решений Арбитра когда этот агент был доминантом
    self.as_dominant  = []   # [(correct: bool, κ), ...]

    # Счётчики
    self.total_cycles      = 0
    self.dominant_cycles   = 0
    self.correct_forecasts = 0
    self.error_streak      = 0
    self.lifetime_switches_out = 0  # сколько раз был заменён

def update(self, was_dominant: bool, forecast_correct: bool, kappa: str):
    """Обновить статистику после цикла."""
    self.total_cycles += 1
    alpha = 0.12

    if was_dominant:
        self.dominant_cycles += 1
        self.as_dominant.append((forecast_correct, kappa))

        # Глобальный R_D
        self.r_d = self.r_d * (1 - alpha) + alpha * (1.0 if forecast_correct else 0.0)
        self.r_d_history.append(self.r_d)

        # R_D по классу
        cs = self.class_stats[kappa]
        cs["total"] += 1
        cs["r_d"] = cs["r_d"] * (1 - alpha) + alpha * (1.0 if forecast_correct else 0.0)
        if forecast_correct:
            cs["wins"] += 1

        # Серия ошибок
        if forecast_correct:
            self.correct_forecasts += 1
            self.error_streak = max(0, self.error_streak - 1)
            self.automatism = min(1.0, self.automatism + 0.04)
        else:
            self.error_streak += 1
            self.automatism = max(0.0, self.automatism - 0.08)

def r_d_trend(self) -> float:
    """
    Тренд R_D: положительный = улучшение, отрицательный = деградация.
    Вычисляется как наклон линейной регрессии по последним N точкам.
    """
    if len(self.r_d_history) < 3:
        return 0.0
    vals = list(self.r_d_history)
    n    = len(vals)
    x    = list(range(n))
    mx, my = sum(x) / n, sum(vals) / n
    num  = sum((x[i] - mx) * (vals[i] - my) for i in range(n))
    den  = sum((x[i] - mx) ** 2 for i in range(n))
    return round(num / (den + 1e-9), 4)

def is_banned_in_class(self, kappa: str) -> bool:
    return self.class_stats[kappa]["banned"]

def should_ban_in_class(self, kappa: str, min_cycles: int, threshold: float) -> bool:
    cs = self.class_stats[kappa]
    return (cs["total"] >= min_cycles and
            cs["r_d"] < threshold and
            not cs["banned"])

def report(self) -> dict:
    return {
        "id":            self.id,
        "active":        self.active,
        "r_d":           round(self.r_d, 3),
        "r_d_trend":     self.r_d_trend(),
        "automatism":    round(self.automatism, 3),
        "error_streak":  self.error_streak,
        "dominant_cycles": self.dominant_cycles,
        "correct_forecasts": self.correct_forecasts,
        "lifetime_switches_out": self.lifetime_switches_out,
        "class_stats":   {
            k: {kk: round(vv, 3) if isinstance(vv, float) else vv
                for kk, vv in v.items()}
            for k, v in self.class_stats.items()
        },
    }
```

# ─────────────────────────────────────────────

# СТАТИСТИКА АРБИТРА

# ─────────────────────────────────────────────

class ArbitratorStats:
“””
Самокорректирующийся Арбитр.
Отслеживает точность своих мета-решений и перекалибрует пороги.
“””

```
def __init__(self):
    self.r_arb        = 0.50
    self.history      = deque(maxlen=10)   # последние решения
    self.decision_log = []                  # [(decision, correct), ...]
    self.thresholds   = dict(BASE_THRESHOLDS)
    self.calibrations = 0                   # сколько раз перекалибровал

def record_decision(self, decision: str, was_correct: bool):
    beta = 0.15
    self.r_arb = (self.r_arb * (1 - beta) +
                  beta * (1.0 if was_correct else 0.0))
    self.history.append(decision)
    self.decision_log.append((decision, was_correct))

def check_and_recalibrate(self) -> list:
    """
    L3 самокоррекция Арбитра.
    Возвращает список изменений которые были сделаны.
    """
    changes = []
    h = list(self.history)

    # Если R_arb упал ниже порога — перекалибровка
    if (self.r_arb < self.thresholds["arb_recalibrate_threshold"]
            and len(self.decision_log) >= 8):

        # Анализируем какие решения были неверными
        wrong = [d for d, ok in self.decision_log[-8:] if not ok]
        wrong_holds   = wrong.count("HOLD")
        wrong_switches = wrong.count("SWITCH")

        if wrong_holds > wrong_switches:
            # Слишком много неверных HOLD → снизить порог для SWITCH
            old = self.thresholds["error_streak_max"]
            self.thresholds["error_streak_max"] = max(1, old - 1)
            changes.append(
                f"Recalibrated: error_streak_max {old}→"
                f"{self.thresholds['error_streak_max']} "
                f"(too many wrong HOLDs)")
        elif wrong_switches > wrong_holds:
            # Слишком много неверных SWITCH → повысить порог
            old = self.thresholds["error_streak_max"]
            self.thresholds["error_streak_max"] = min(6, old + 1)
            changes.append(
                f"Recalibrated: error_streak_max {old}→"
                f"{self.thresholds['error_streak_max']} "
                f"(too many wrong SWITCHes)")

        self.calibrations += 1

    # Если 7+ подряд одно решение — сигнал аномалии
    if len(h) >= 7 and len(set(h[-7:])) == 1:
        stuck = h[-1]
        if stuck == "HOLD":
            # Застряли в HOLD — снизить порог чувствительности
            old = self.thresholds["theta_H"]
            self.thresholds["theta_H"] = max(0.40, old - 0.05)
            changes.append(
                f"Pattern correction: theta_H {old:.2f}→"
                f"{self.thresholds['theta_H']:.2f} "
                f"(stuck in HOLD)")
        elif stuck == "FREEZE":
            # Застряли в FREEZE — повысить порог
            old = self.thresholds["theta_H"]
            self.thresholds["theta_H"] = min(0.90, old + 0.05)
            changes.append(
                f"Pattern correction: theta_H {old:.2f}→"
                f"{self.thresholds['theta_H']:.2f} "
                f"(stuck in FREEZE)")

    return changes

def report(self) -> dict:
    return {
        "r_arb":        round(self.r_arb, 3),
        "calibrations": self.calibrations,
        "thresholds":   {k: round(v, 3) if isinstance(v, float) else v
                         for k, v in self.thresholds.items()},
        "last_10":      list(self.history),
    }
```

# ─────────────────────────────────────────────

# МЕНЕДЖЕР ПУЛА АГЕНТОВ

# ─────────────────────────────────────────────

class AgentPoolManager:
“””
Управляет пулом агентов.
Реализует все три уровня самокоррекции.
“””

```
def __init__(self, agent_ids: list):
    self.stats = {aid: AgentStats(aid) for aid in agent_ids}
    self.dominant_id   = None
    self.dominant_age  = 0

# ── Уровень 1: Реактивный ──────────────────

def l1_check_error_streak(self, threshold: int) -> tuple[bool, str]:
    """Серия ошибок ≥ порога → рекомендовать SWITCH."""
    if self.dominant_id is None:
        return False, ""
    st = self.stats[self.dominant_id]
    if st.error_streak >= threshold:
        return True, (f"L1 REACTIVE: {self.dominant_id} error_streak="
                      f"{st.error_streak} ≥ {threshold}")
    return False, ""

# ── Уровень 2: Превентивный ────────────────

def l2_check_trend(self) -> list:
    """
    Анализирует тренд R_D всех агентов.
    Возвращает список предупреждений и корректирует веса дискуссии.
    """
    warnings = []
    for aid, st in self.stats.items():
        if not st.active:
            continue
        trend = st.r_d_trend()
        if trend < -0.03 and st.dominant_cycles > 3:
            warnings.append({
                "agent":   aid,
                "trend":   trend,
                "r_d":     st.r_d,
                "message": (f"L2 PREVENTIVE: {aid} R_D trending down "
                            f"(trend={trend:.3f}, r_d={st.r_d:.3f})")
            })
    return warnings

def l2_weight_modifier(self, agent_id: str) -> float:
    """
    Модификатор веса агента в скоринге на основе тренда.
    Деградирующий агент получает пониженный вес до смены.
    """
    st    = self.stats[agent_id]
    trend = st.r_d_trend()
    if trend > 0.02:
        return 1.10   # тренд вверх — небольшой бонус
    elif trend < -0.03:
        return 0.80   # тренд вниз — штраф
    elif trend < -0.06:
        return 0.60   # сильная деградация — значительный штраф
    return 1.00

# ── Уровень 3: Структурный ─────────────────

def l3_check_class_bans(self, thresholds: dict) -> list:
    """
    Проверяет нужно ли забанить агента в определённом классе сценариев.
    Возвращает список произведённых банов.
    """
    bans = []
    min_c = thresholds["class_ban_min_cycles"]
    ban_t = thresholds["class_ban_threshold"]

    for aid, st in self.stats.items():
        for kappa in ["Local", "Cluster", "Structural"]:
            if st.should_ban_in_class(kappa, min_c, ban_t):
                st.class_stats[kappa]["banned"] = True
                bans.append({
                    "agent": aid,
                    "class": kappa,
                    "r_d_in_class": round(st.class_stats[kappa]["r_d"], 3),
                    "message": (f"L3 STRUCTURAL: {aid} banned from class "
                                f"{kappa} (r_d={st.class_stats[kappa]['r_d']:.3f} "
                                f"< {ban_t})")
                })
    return bans

def l3_global_deactivate(self, thresholds: dict) -> list:
    """
    Полностью деактивирует агента если глобальный R_D упал критически.
    """
    deactivated = []
    for aid, st in self.stats.items():
        if (st.active and
                st.r_d < thresholds["r_d_min"] and
                st.dominant_cycles >= 10):
            st.active = False
            deactivated.append({
                "agent":   aid,
                "r_d":     round(st.r_d, 3),
                "message": (f"L3 STRUCTURAL: {aid} deactivated globally "
                            f"(r_d={st.r_d:.3f} < {thresholds['r_d_min']})")
            })
    return deactivated

# ── Выбор нового доминанта ─────────────────

def select_best(self, scores: dict, kappa: str,
                exclude: list = None) -> str:
    """
    Выбирает лучшего доступного агента с учётом:
    - активности агента
    - бана в текущем классе
    - списка исключений
    """
    exclude = exclude or []
    candidates = {
        aid: sc for aid, sc in scores.items()
        if (aid not in exclude and
            self.stats[aid].active and
            not self.stats[aid].is_banned_in_class(kappa))
    }
    if not candidates:
        # Все забанены — берём любого активного
        candidates = {aid: sc for aid, sc in scores.items()
                      if self.stats[aid].active}
    if not candidates:
        return list(scores.keys())[0]
    return max(candidates, key=candidates.get)

def apply_switch(self, new_id: str):
    """Зафиксировать смену доминанта."""
    if self.dominant_id and self.dominant_id in self.stats:
        self.stats[self.dominant_id].lifetime_switches_out += 1
    self.dominant_id  = new_id
    self.dominant_age = 0

def active_agents(self) -> list:
    return [aid for aid, st in self.stats.items() if st.active]

def report(self) -> dict:
    return {aid: st.report() for aid, st in self.stats.items()}
```

# ─────────────────────────────────────────────

# ЗАГЛУШКИ АГЕНТОВ (те же что раньше)

# ─────────────────────────────────────────────

def mock_agent_interpret(agent_id: str, delta_s: dict, scenario: str,
others: dict = None, weight_mod: float = 1.0) -> dict:
p  = delta_s.get(“person_count”, 0)

```
RESPONSES = {
    "A1_motion": {
        "normal": ("Normal orderly movement within baseline.",
                   "Stable flow expected.", ["flow_speed_delta"], 0.82),
        "crowd":  ("Flow dropping — crowd compression forming.",
                   "Movement slows further in 2-3 windows.",
                   ["flow_speed_delta", "stationary_ratio_delta"], 0.71),
        "chaos":  ("High-speed chaotic movement. Conflicting flow vectors.",
                   "Continued escalation expected.",
                   ["flow_speed_delta", "flow_dir_std_delta"], 0.78),
    },
    "A2_density": {
        "normal": ("Density normal. No critical zones.",
                   "Stable density.", ["person_count_delta"], 0.88),
        "crowd":  (f"Critical density in zone C. Count +{abs(p)}.",
                   "Density rising. Zone C critical next window.",
                   ["person_count_delta", "density_zone_critical"], 0.85),
        "chaos":  ("Density fragmenting. No coherent grouping.",
                   "Further dispersal or regrouping at exits.",
                   ["person_count_delta", "scene_entropy_delta"], 0.67),
    },
    "A3_patterns": {
        "normal": ("NORMAL PEAK HOUR (88%). Standard flow.",
                   "Gradual dispersal as peak ends.",
                   ["flow_speed_delta", "person_count_delta"], 0.86),
        "crowd":  ("CROWD SURGE (71%). Closest: exit blockage.",
                   "Dispersal or crush escalation.",
                   ["person_count_delta", "flow_speed_delta"], 0.69),
        "chaos":  ("No pattern match. Entropy too high.",
                   "Unpredictable. Monitor for stabilization.",
                   ["scene_entropy_delta"], 0.41),
    },
    "A4_anomaly": {
        "normal": ("No anomalies. Signals within expected correlation.",
                   "Stable — no hidden change indicators.",
                   [], 0.79),
        "crowd":  ("ANOMALY: High stationary contradicts surge. Physical blockage.",
                   "Check exit status. Crush risk if unresolved.",
                   ["stationary_ratio_delta", "boundary_crossings"], 0.83),
        "chaos":  ("ANOMALY: Entropy spike without audio match. Staged?",
                   "Verify audio. If stable — check camera.",
                   ["audio_level_delta"], 0.74),
    },
    "A5_context": {
        "normal": ("Fully consistent with expected baseline.",
                   "No contextual deviation expected.",
                   ["flow_speed_delta"], 0.91),
        "crowd":  ("Peak hour but density 40% above Friday baseline.",
                   "Natural dispersal in 8-10 min OR monitor.",
                   ["person_count_delta", "stationary_ratio_delta"], 0.80),
        "chaos":  ("No scheduled events. CRITICAL deviation from baseline.",
                   "Treat as incident until proven otherwise.",
                   ["scene_entropy_delta", "person_count_delta"], 0.88),
    },
}

sc = scenario if scenario in ["normal", "crowd", "chaos"] else "normal"
interp_txt, forecast, signals, conf = RESPONSES.get(
    agent_id, RESPONSES["A5_context"])[sc]

# Применить L2-модификатор веса к confidence
conf = round(min(0.99, conf * weight_mod), 3)

return {
    "interpretation": interp_txt,
    "forecast":       forecast,
    "key_signals":    signals,
    "ignored_signals": RESPONSES.get(agent_id, {}).get(sc, [None]*4)[2]
                       if agent_id == "A4_anomaly" else [],
    "confidence":     conf,
}
```

# ─────────────────────────────────────────────

# СКОРИНГ И ЭНТРОПИЯ

# ─────────────────────────────────────────────

def score_agent(interp: dict, all_interps: dict, delta_s: dict,
kappa: str, weight_mod: float = 1.0) -> float:
w       = WEIGHTS[kappa]
signals = interp.get(“key_signals”, [])
C = min(len(signals) / max(len(delta_s), 1), 1.0)
U = float(interp.get(“confidence”, 0.5))
E = 1.0 / (1.0 + len(interp.get(“interpretation”, “”).split()) / 25)
my_s     = set(signals)
others_s = set(s for ai in all_interps.values()
for s in ai.get(“key_signals”, []))
Q = min(len(my_s - others_s) / max(len(my_s), 1), 1.0)
raw = w[“C”]*C + w[“U”]*U + w[“E”]*E + w[“Q”]*Q
return round(raw * weight_mod, 4)

def compute_entropy(interps: dict, error_streak: int, flux: float) -> float:
confs    = [v.get(“confidence”, 0.5) for v in interps.values()]
div_raw  = float(np.std(confs)) if len(confs) > 1 else 0.0
div_norm = div_raw / (div_raw + 1)
err_norm = error_streak / (error_streak + 1)
fx_norm  = max(flux - 1.0, 0.0) / (max(flux - 1.0, 0.0) + 1)
return round(0.40*div_norm + 0.35*err_norm + 0.25*fx_norm, 4)

# ─────────────────────────────────────────────

# АРБИТР (с самокоррекцией)

# ─────────────────────────────────────────────

def run_arbitrator(payload: dict, arb_stats: ArbitratorStats,
pool: AgentPoolManager) -> dict:
“””
Арбитр с тремя уровнями самокоррекции встроенными в логику.
“””
T            = arb_stats.thresholds
error_streak = payload[“dominant”][“error_streak”]
H            = payload[“H”]
flux         = payload[“flux”]
dom_age      = payload[“dominant”][“age”]
scores       = payload[“scores”]
dom_id       = payload[“dominant”][“id”]

```
flags       = []
corrections = []

# ── L1: Реактивный ──
l1_trigger, l1_msg = pool.l1_check_error_streak(T["error_streak_max"])
if l1_trigger:
    flags.append({"level": "L1", "severity": "high",
                  "flag": "error_streak", "message": l1_msg})

# ── L2: Превентивный ──
l2_warnings = pool.l2_check_trend()
for w in l2_warnings:
    flags.append({"level": "L2", "severity": "medium",
                  "flag": "r_d_trend_negative", "message": w["message"]})

# ── L3: Структурный ──
l3_bans  = pool.l3_check_class_bans(T)
l3_deact = pool.l3_global_deactivate(T)
for b in l3_bans:
    flags.append({"level": "L3", "severity": "high",
                  "flag": "class_ban", "message": b["message"]})
    corrections.append(b["message"])
for d in l3_deact:
    flags.append({"level": "L3", "severity": "critical",
                  "flag": "global_deactivation", "message": d["message"]})
    corrections.append(d["message"])

# ── Самокоррекция Арбитра ──
arb_corrections = arb_stats.check_and_recalibrate()
corrections.extend(arb_corrections)
if arb_corrections:
    flags.append({"level": "ARB", "severity": "medium",
                  "flag": "self_recalibration",
                  "message": "; ".join(arb_corrections)})

# A4 уникальный сигнал
a4_ignored = payload["interpretations"].get(
    "A4_anomaly", {}).get("ignored_signals", [])
if a4_ignored:
    flags.append({"level": "L1", "severity": "medium",
                  "flag": "unique_signal_suppressed",
                  "message": f"A4 flagged ignored signals: {a4_ignored}"})

# ── Принятие решения ──
high_flags = [f for f in flags if f["severity"] in ("high", "critical")]
kappa      = payload["kappa"]

# Текущий доминант забанен в этом классе?
dom_banned = (dom_id and
              pool.stats.get(dom_id, AgentStats("x")).is_banned_in_class(kappa))

if H > T["theta_H"] and flux < T["theta_flux"]:
    decision     = "FREEZE"
    new_dominant = None
    reasoning    = (f"H={H:.2f}>{T['theta_H']} and flux={flux:.2f}<{T['theta_flux']}. "
                    f"Insufficient new signal. Waiting.")
    confidence   = 0.74
    action       = "Accumulate signal before next interpretation."

elif high_flags or dom_banned:
    decision     = "SWITCH"
    exclude      = [dom_id] if dom_id else []
    new_dominant = pool.select_best(scores, kappa, exclude=exclude)
    reason_parts = [f["message"] for f in high_flags[:2]]
    if dom_banned:
        reason_parts.append(f"{dom_id} banned in class {kappa}")
    reasoning    = " | ".join(reason_parts)
    confidence   = min(0.95, 0.75 + 0.05 * len(high_flags))
    action       = f"Switched to {new_dominant}. Verify conditions."

else:
    decision     = "HOLD"
    new_dominant = None
    reasoning    = (f"No critical flags. H={H:.2f} manageable. "
                    f"Minor flags: {len(flags) - len(high_flags)}.")
    confidence   = max(0.50, 0.88 - 0.04 * len(flags))
    action       = None

return {
    "decision":           decision,
    "confidence":         round(confidence, 2),
    "flags":              flags,
    "corrections":        corrections,
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

    stationary = 0.0
    if self.prev_gray is not None:
        diff = cv2.absdiff(gray, self.prev_gray)
        stationary = float(np.mean(diff < 10))

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = sum(1 for c in contours if 300 < cv2.contourArea(c) < 40000)

    self.prev_gray = gray
    return {
        "person_count":     count,
        "flow_speed":       round(flow_speed, 3),
        "flow_dir_std":     round(flow_dir_std, 3),
        "scene_entropy":    round(entropy, 3),
        "stationary_ratio": round(stationary, 3),
        "audio_level":      0.0,
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

def classify_scenario(delta_s: dict, flux: float) -> str:
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

class DominantaXSelfRegulating:

```
def __init__(self):
    self.pool          = AgentPoolManager(AGENT_IDS)
    self.arb_stats     = ArbitratorStats()
    self.preprocessor  = VideoPreprocessor()

    self.dominant_interp = None
    self.prev_s          = None
    self.cycle           = 0
    self.frozen          = False
    self.log             = []

def run_cycle(self, frame: np.ndarray, scenario: str = "normal") -> dict:
    self.cycle += 1
    ts = datetime.now().strftime("%H:%M:%S")
    dom = self.pool.dominant_id

    print(f"\n{'═'*64}")
    print(f"  ЦИКЛ {self.cycle:02d} | {ts} | {scenario.upper()}"
          f" | Доминант: {dom or '—'}")
    print(f"{'═'*64}")

    # OBSERVE
    s_curr  = self.preprocessor.extract(frame)
    delta_s = compute_delta(s_curr, self.prev_s)
    self.prev_s = s_curr

    prev_flow = (self.prev_s or {}).get("flow_speed", 1.0)
    flux = s_curr["flow_speed"] / (prev_flow + 0.001)
    kappa = classify_scenario(delta_s, flux)

    print(f"\n📡 ΔS: persons={delta_s['person_count']:+d}  "
          f"flow={delta_s['flow_speed']:+.3f}  "
          f"entropy={delta_s['scene_entropy']:+.3f}  "
          f"[κ={kappa}]")

    if self.frozen:
        sig = abs(delta_s["scene_entropy"]) + abs(delta_s["person_count"]) * 0.1
        if sig > 0.15:
            self.frozen = False
            print("⏸→▶  Размораживаем")
        else:
            print("⏸  ЗАМОРОЖЕНО")
            return {"cycle": self.cycle, "decision": "FREEZE"}

    active = self.pool.active_agents()
    print(f"   Активных агентов: {len(active)}/{len(AGENT_IDS)}")

    # GENERATE
    weight_mods = {aid: self.pool.l2_weight_modifier(aid) for aid in active}
    first_pass  = {}
    for aid in active:
        wm = weight_mods[aid]
        first_pass[aid] = mock_agent_interpret(aid, delta_s, scenario,
                                               weight_mod=wm)

    # DISCUSS (Задержка Каминского)
    final_interps = {}
    for aid in active:
        others = {k: v for k, v in first_pass.items() if k != aid}
        final_interps[aid] = mock_agent_interpret(
            aid, delta_s, scenario, others=others,
            weight_mod=weight_mods[aid])

    # SCORE с L2-модификаторами
    scores = {}
    for aid in active:
        others_wo = {k: v for k, v in final_interps.items() if k != aid}
        scores[aid] = score_agent(
            final_interps[aid], others_wo, delta_s, kappa,
            weight_mod=weight_mods.get(aid, 1.0))

    print(f"\n📊 Scores [{kappa}]:")
    for aid, sc in sorted(scores.items(), key=lambda x: -x[1]):
        wm     = weight_mods.get(aid, 1.0)
        bar    = "█" * int(sc * 18)
        wm_str = f" [L2×{wm:.2f}]" if abs(wm - 1.0) > 0.05 else ""
        ban    = " [BANNED]" if self.pool.stats[aid].is_banned_in_class(kappa) else ""
        print(f"   {aid:12s} {sc:.3f} {bar}{wm_str}{ban}")

    # ENTROPY
    error_streak = (self.pool.stats[dom].error_streak
                    if dom and dom in self.pool.stats else 0)
    H = compute_entropy(final_interps, error_streak, flux)

    # Статистика агентов
    for aid in active:
        st = self.pool.stats[aid]
        trend = st.r_d_trend()
        if abs(trend) > 0.02 or st.error_streak > 0:
            sign = "↑" if trend > 0 else ("↓" if trend < 0 else "→")
            print(f"   {aid}: R_D={st.r_d:.3f} {sign}{trend:+.3f}  "
                  f"streak={st.error_streak}  λ={st.automatism:.2f}")

    print(f"\n🌡  H={H:.3f}  flux={flux:.2f}  "
          f"dominant_age={self.pool.dominant_age}  "
          f"arb_R={self.arb_stats.r_arb:.3f}")

    # ARBITRATE
    arb_payload = {
        "H": H, "flux": round(flux, 3), "kappa": kappa,
        "scores": scores,
        "interpretations": {
            aid: {"ignored_signals": interp.get("ignored_signals", []),
                  "confidence": interp.get("confidence", 0.5)}
            for aid, interp in final_interps.items()
        },
        "dominant": {
            "id":           dom,
            "age":          self.pool.dominant_age,
            "error_streak": error_streak,
        },
    }

    arb = run_arbitrator(arb_payload, self.arb_stats, self.pool)
    decision = arb["decision"]

    # Вывод флагов
    if arb["flags"]:
        print(f"\n🚩 Флаги ({len(arb['flags'])}):")
        for fl in arb["flags"]:
            icons = {"high":"🔴","critical":"🆘","medium":"🟡",
                     "low":"🟢","L1":"⚡","L2":"📉","L3":"🚫","ARB":"🔧"}
            icon = icons.get(fl["severity"], "•")
            print(f"   {icon} [{fl['level']}·{fl['flag']}] "
                  f"{fl['message'][:70]}")

    # Самокоррекции
    if arb["corrections"]:
        print(f"\n🔧 Автокоррекции:")
        for c in arb["corrections"]:
            print(f"   ✦ {c}")

    col = {"HOLD": "✅", "SWITCH": "🔄", "FREEZE": "⏸"}[decision]
    print(f"\n{col} {decision} (conf={arb['confidence']:.2f})")
    print(f"   {arb['reasoning']}")
    if arb.get("recommended_action"):
        print(f"\n⚠️  ОПЕРАТОРУ: {arb['recommended_action']}")

    # APPLY DECISION
    if decision == "SWITCH":
        new_id = arb.get("new_dominant")
        if new_id:
            print(f"\n   {dom or '—'} → {new_id}")
            self.pool.apply_switch(new_id)
            self.dominant_interp = final_interps.get(new_id, {})
        else:
            decision = "HOLD"

    elif decision == "HOLD":
        if self.pool.dominant_id is None and scores:
            best = self.pool.select_best(scores, kappa)
            self.pool.apply_switch(best)
            self.dominant_interp = final_interps.get(best, {})
        self.pool.dominant_age += 1

    elif decision == "FREEZE":
        self.frozen = True

    # Запись решения Арбитра (для его самокоррекции)
    # В реальной системе проверяем через N циклов
    # Здесь: считаем верным если H не выросла после HOLD
    arb_correct = (decision == "HOLD" and H < 0.5) or \
                  (decision == "SWITCH") or \
                  (decision == "FREEZE" and flux < 0.3)
    self.arb_stats.record_decision(decision, arb_correct)

    # LEARN — обновить статистику агентов
    cur_dom = self.pool.dominant_id
    if cur_dom and cur_dom in self.pool.stats:
        forecast = final_interps.get(cur_dom, {}).get("forecast", "")
        ok = len(forecast) > 15
        self.pool.stats[cur_dom].update(
            was_dominant=True, forecast_correct=ok, kappa=kappa)
    for aid in active:
        if aid != cur_dom:
            self.pool.stats[aid].update(
                was_dominant=False, forecast_correct=True, kappa=kappa)

    dom_txt = ""
    if cur_dom and cur_dom in final_interps:
        dom_txt = final_interps[cur_dom].get("interpretation", "")[:70]
    print(f"\n📌 Доминанта [{cur_dom}]: {dom_txt}...")

    result = {
        "cycle":       self.cycle,
        "scenario":    scenario,
        "kappa":       kappa,
        "H":           H,
        "flux":        round(flux, 3),
        "scores":      scores,
        "decision":    decision,
        "dominant":    cur_dom,
        "dominant_age": self.pool.dominant_age,
        "corrections": arb["corrections"],
        "flags":       [{k: v for k, v in f.items()} for f in arb["flags"]],
        "action":      arb.get("recommended_action"),
    }
    self.log.append(result)
    return result

def print_summary(self):
    print(f"\n{'═'*64}")
    print(f"  ИТОГ | {len(self.log)} циклов")
    print(f"{'═'*64}")

    decisions = [r["decision"] for r in self.log]
    print(f"  HOLD={decisions.count('HOLD')}  "
          f"SWITCH={decisions.count('SWITCH')}  "
          f"FREEZE={decisions.count('FREEZE')}")

    print(f"\n  Агенты:")
    print(f"  {'ID':14s} {'R_D':>5} {'trend':>7} {'dom_c':>6} "
          f"{'streak':>7} {'active':>7}")
    print(f"  {'-'*56}")
    for aid, st in sorted(self.pool.stats.items(),
                           key=lambda x: -x[1].r_d):
        trend = st.r_d_trend()
        tsign = "↑" if trend > 0.01 else ("↓" if trend < -0.01 else "→")
        act   = "✓" if st.active else "✗"
        print(f"  {aid:14s} {st.r_d:5.3f} {tsign}{trend:6.3f} "
              f"{st.dominant_cycles:6d} {st.error_streak:7d} {act:>7}")

    print(f"\n  Арбитр:")
    ar = self.arb_stats.report()
    print(f"  R_arb={ar['r_arb']:.3f}  "
          f"Перекалибровок={ar['calibrations']}")
    print(f"  Текущие пороги: "
          f"error_streak_max={ar['thresholds']['error_streak_max']}  "
          f"theta_H={ar['thresholds']['theta_H']:.2f}")
    print(f"  История: {' → '.join(ar['last_10'])}")

    # Баны по классам
    print(f"\n  Баны по классам:")
    has_bans = False
    for aid, st in self.pool.stats.items():
        for kappa, cs in st.class_stats.items():
            if cs["banned"]:
                print(f"  🚫 {aid} запрещён в [{kappa}] "
                      f"(r_d={cs['r_d']:.3f})")
                has_bans = True
    if not has_bans:
        print("  — нет")

    with open("dominanta_x_selfregulating_log.json", "w",
              encoding="utf-8") as f:
        json.dump({
            "log":       self.log,
            "agents":    self.pool.report(),
            "arbitrator": self.arb_stats.report(),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Лог: dominanta_x_selfregulating_log.json")
```

# ─────────────────────────────────────────────

# ЗАПУСК

# ─────────────────────────────────────────────

if **name** == “**main**”:
print(“╔══════════════════════════════════════════════════════════════╗”)
print(“║  DOMINANTA X — Саморегулирующийся движок                   ║”)
print(“║  L1 Реактивный · L2 Превентивный · L3 Структурный          ║”)
print(“║  Концепция: Игорь Каминский                                 ║”)
print(“╚══════════════════════════════════════════════════════════════╝”)

```
engine = DominantaXSelfRegulating()

# Сценарий: норма → скопление → хаос → норма
scenarios = (["normal"] * 3 + ["crowd"] * 4 +
             ["chaos"] * 3 + ["normal"] * 3)

for sc in scenarios:
    frame = make_frame(sc)
    engine.run_cycle(frame, scenario=sc)
    time.sleep(0.2)

engine.print_summary()
```
