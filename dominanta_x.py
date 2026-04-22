# “””
DOMINANTA X — Движок конкурентной интерпретации

Автор концепции: Игорь Каминский
Реализация: Claude (Anthropic)

Запуск:
pip install anthropic opencv-python numpy
python dominanta_x.py –video путь/к/файлу.mp4
python dominanta_x.py –test   # без видео, синтетические данные

API ключ:
export ANTHROPIC_API_KEY=sk-ant-…
или вставить в строку API_KEY ниже
“””

import os
import cv2
import json
import time
import math
import argparse
import numpy as np
from datetime import datetime
from typing import Optional
import anthropic

# ─────────────────────────────────────────────

# КОНФИГУРАЦИЯ

# ─────────────────────────────────────────────

API_KEY = os.environ.get(“ANTHROPIC_API_KEY”, “ВСТАВЬТЕ_КЛЮЧ_СЮДА”)
MODEL   = “claude-sonnet-4-6”

# Тактовая частота: один цикл каждые N секунд видео

WINDOW_SECONDS = 2.0

# Пороги Арбитра (калибруются под домен)

THRESHOLDS = {
“rho_low”:         0.15,
“rho_high”:        0.45,
“theta_err”:       0.40,
“theta_H”:         0.70,
“theta_flux”:      0.20,
“error_streak_max”: 3,
}

# Веса оценки аргументов по классу сценария

WEIGHTS = {
“Local”:      {“C”: 0.35, “U”: 0.20, “E”: 0.15, “Q”: 0.30},
“Cluster”:    {“C”: 0.25, “U”: 0.30, “E”: 0.25, “Q”: 0.20},
“Structural”: {“C”: 0.20, “U”: 0.35, “E”: 0.30, “Q”: 0.15},
}

# ─────────────────────────────────────────────

# СИСТЕМНЫЕ ПРОМПТЫ АГЕНТОВ

# ─────────────────────────────────────────────

AGENT_PROMPTS = {

```
"A1_motion": """You are Agent A1 — Motion Analyst in the Dominanta X system.
```

ICE Core: Algorithmic. You analyze movement patterns, optical flow, speed, and direction.

You receive a JSON with delta_S (what CHANGED in the last window) from a video stream.
Focus on: flow_speed_delta, flow_direction_std_delta, boundary_crossings_delta.

Output JSON only:
{
“interpretation”: “1-2 sentences describing what the movement pattern means”,
“forecast”: “what you expect to happen in the next 1-3 windows”,
“key_signals”: [“list of signals that drove your interpretation”],
“confidence”: 0.0-1.0
}”””,

```
"A2_density": """You are Agent A2 — Density Analyst in the Dominanta X system.
```

ICE Core: Algorithmic. You analyze crowd density, distribution, and grouping patterns.

You receive a JSON with delta_S (what CHANGED in the last window) from a video stream.
Focus on: person_count_delta, density_zone_critical, stationary_ratio_delta.

Output JSON only:
{
“interpretation”: “1-2 sentences describing what the density pattern means”,
“forecast”: “what you expect in the next 1-3 windows”,
“key_signals”: [“signals that drove your interpretation”],
“confidence”: 0.0-1.0
}”””,

```
"A3_patterns": """You are Agent A3 — Pattern Matcher in the Dominanta X system.
```

ICE Core: Social. You match current observations against known historical scenarios.

You receive a JSON with delta_S AND the interpretations of A1 and A2.
Compare the combined picture to known patterns: crowd surge, evacuation, normal peak hour,
fight, abandoned object, panic, planned event dispersal.

Output JSON only:
{
“interpretation”: “1-2 sentences: which pattern this most resembles and confidence”,
“forecast”: “expected evolution based on the matched pattern”,
“pattern_match”: “name of closest historical pattern”,
“match_confidence”: 0.0-1.0,
“key_signals”: [“signals that drove your interpretation”],
“confidence”: 0.0-1.0
}”””,

```
"A4_anomaly": """You are Agent A4 — Anomaly Detector in the Dominanta X system.
```

ICE Core: Existential. You specifically look for what does NOT fit, what contradicts,
what is suspiciously absent, and what appears staged or too deliberate.

You receive a JSON with delta_S AND interpretations from A1, A2, A3.
Look for contradictions between agents. Find signals others ignored.
Ask: what is wrong with the picture? What is missing that should be there?

Output JSON only:
{
“interpretation”: “1-2 sentences: what anomaly or contradiction you detected”,
“forecast”: “what this anomaly suggests about the near future”,
“anomaly_type”: “absence | contradiction | suspicious_regularity | ignored_signal”,
“ignored_signals”: [“signals other agents did not address”],
“key_signals”: [“signals that drove your interpretation”],
“confidence”: 0.0-1.0
}”””,

```
"A5_context": """You are Agent A5 — Context Keeper in the Dominanta X system.
```

ICE Core: Social. You hold the long-term temporal and situational context.
You know what is NORMAL for this place and time.

You receive delta_S, all agent interpretations, AND the current dominant interpretation D*.
Assess: is this situation within normal baseline? Does it fit the time of day and day of week?
Does the current dominant D* still make sense given accumulated history?

Output JSON only:
{
“interpretation”: “1-2 sentences: is this normal, abnormal, or transitional for this context”,
“forecast”: “expected evolution based on contextual knowledge”,
“baseline_assessment”: “within_normal | mild_deviation | significant_deviation | critical”,
“dominant_still_valid”: true/false,
“reason_if_invalid”: “why D* may no longer apply, or null”,
“key_signals”: [“signals that drove your interpretation”],
“confidence”: 0.0-1.0
}”””,
}

# Системный промпт Арбитра (сокращённая версия для скорости)

ARBITRATOR_PROMPT = “”“You are the Arbitrator of the Dominanta X system.
You do NOT interpret the environment. You evaluate the system’s state and health.

Run these checks before deciding:

1. Forecast accuracy — did current dominant’s forecast come true? Error streak?
1. Post-discussion divergence — did agents converge or remain split?
1. Revision quality — did revisions have substantive reasons?
1. Hallucination — does any interpretation contradict the input data?
1. Automatism vs flux — high automatism in rapidly changing environment?
1. Dominant staleness — held too long with rising entropy?
1. Amplitude mismatch — forecast horizon matches scenario type?
1. Suppressed uniqueness — did A4 raise something all others ignored?

Decision rules:

- HOLD: accurate forecasts, converging agents, stable entropy, no major flags
- SWITCH: error streak ≥ 3, better alternative exists, hallucination detected, staleness high
- FREEZE: high entropy + low new signal, agents cannot converge, genuine uncertainty

Output JSON only:
{
“decision”: “HOLD” | “SWITCH” | “FREEZE”,
“confidence”: 0.0-1.0,
“flags”: [{“check”: int, “flag”: “name”, “agent”: “id or null”, “severity”: “low|medium|high”}],
“reasoning”: “2-3 sentences explaining decision”,
“new_dominant”: “agent_id if SWITCH else null”,
“recommended_action”: “one sentence for operator or null”
}”””

# ─────────────────────────────────────────────

# ПРЕПРОЦЕССОР ВИДЕО

# ─────────────────────────────────────────────

class VideoPreprocessor:
“”“Извлекает S(t) из видеокадров.”””

```
def __init__(self):
    self.prev_frame   = None
    self.prev_gray    = None
    self.prev_metrics = None

def extract(self, frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Число людей — упрощённая эвристика (контуры движущихся объектов)
    person_count = self._estimate_persons(frame, gray)

    # Оптический поток
    flow_speed, flow_dir_std = 0.0, 0.0
    if self.prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_speed   = float(np.mean(mag))
        flow_dir_std = float(np.std(ang))

    # Энтропия кадра (мера хаотичности)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = float(-np.sum(hist * np.log2(hist + 1e-7)))

    # Яркость
    brightness = float(np.mean(gray))

    # Доля статичных пикселей
    stationary = 0.0
    if self.prev_gray is not None:
        diff = cv2.absdiff(gray, self.prev_gray)
        stationary = float(np.mean(diff < 10))

    metrics = {
        "person_count": person_count,
        "flow_speed":   round(flow_speed, 3),
        "flow_dir_std": round(flow_dir_std, 3),
        "scene_entropy": round(entropy, 3),
        "brightness":   round(brightness, 1),
        "stationary_ratio": round(stationary, 3),
    }

    self.prev_gray    = gray
    self.prev_frame   = frame
    self.prev_metrics = metrics
    return metrics

def _estimate_persons(self, frame, gray) -> int:
    """Грубая оценка числа людей через детектор контуров."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Фильтруем по размеру — грубая эвристика
    persons = sum(1 for c in contours if 500 < cv2.contourArea(c) < 50000)
    return persons
```

def compute_delta(s_curr: dict, s_prev: Optional[dict]) -> dict:
“”“ΔS(t) = S(t) − S(t−1)”””
if s_prev is None:
return {k: 0.0 for k in s_curr}
return {k: round(s_curr[k] - s_prev.get(k, 0), 3) for k in s_curr}

# ─────────────────────────────────────────────

# АГЕНТЫ

# ─────────────────────────────────────────────

class Agent:
def **init**(self, agent_id: str, client: anthropic.Anthropic):
self.id         = agent_id
self.client     = client
self.memory     = []        # история успешных интерпретаций
self.automatism = 0.0       # λᵢ
self.r_d        = 0.5       # рейтинг доминанта

```
def interpret(self, delta_s: dict, others: dict = None, dominant: str = None) -> dict:
    """Генерирует интерпретацию. others — словарь интерпретаций других агентов."""
    context = {"delta_S": delta_s}
    if others:
        context["other_agents"] = others
    if dominant:
        context["current_dominant_D*"] = dominant

    user_msg = json.dumps(context, ensure_ascii=False)

    try:
        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=AGENT_PROMPTS[self.id],
            messages=[{"role": "user", "content": user_msg}]
        )
        raw = resp.content[0].text.strip()
        # Извлечь JSON
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        return json.loads(raw[start:end])
    except Exception as e:
        return {
            "interpretation": f"[Error: {e}]",
            "forecast": "",
            "confidence": 0.0,
            "key_signals": []
        }

def update_rating(self, forecast_correct: bool):
    alpha = 0.15
    self.r_d = self.r_d * (1 - alpha) + alpha * (1.0 if forecast_correct else 0.0)
    # Обновить автоматизм
    if forecast_correct:
        self.automatism = min(1.0, self.automatism + 0.05)
    else:
        self.automatism = max(0.0, self.automatism - 0.10)
```

# ─────────────────────────────────────────────

# АРБИТР

# ─────────────────────────────────────────────

class Arbitrator:
def **init**(self, client: anthropic.Anthropic):
self.client   = client
self.r_arb    = 0.5
self.history  = []   # последние 5 решений

```
def decide(self, payload: dict) -> dict:
    user_msg = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=ARBITRATOR_PROMPT,
            messages=[{"role": "user", "content": user_msg}]
        )
        raw = resp.content[0].text.strip()
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        result = json.loads(raw[start:end])
    except Exception as e:
        result = {
            "decision": "FREEZE",
            "confidence": 0.3,
            "flags": [],
            "reasoning": f"Arbitrator error: {e}",
            "new_dominant": None,
            "recommended_action": "Check system logs."
        }

    self.history.append(result.get("decision", "FREEZE"))
    if len(self.history) > 5:
        self.history.pop(0)
    return result

def update_rating(self, decision_correct: bool):
    beta = 0.15
    self.r_arb = self.r_arb * (1 - beta) + beta * (1.0 if decision_correct else 0.0)
```

# ─────────────────────────────────────────────

# SCORING

# ─────────────────────────────────────────────

def score_agent(interp: dict, all_interps: dict, delta_s: dict,
scenario_class: str = “Local”) -> float:
“”“Вычисляет Score(Iᵢ’) по 4 осям.”””
w = WEIGHTS[scenario_class]

```
# Coverage — сколько key_signals покрывает интерпретация
signals = interp.get("key_signals", [])
total   = max(len(delta_s), 1)
C = min(len(signals) / total, 1.0)

# Stability — уверенность агента (confidence как прокси)
U = float(interp.get("confidence", 0.5))

# Parsimony — обратное к числу допущений (упрощение: длина интерпретации)
text = interp.get("interpretation", "")
E = 1.0 / (1.0 + len(text.split()) / 30)

# Uniqueness — сигналы которые другие не упомянули
my_signals = set(interp.get("key_signals", []))
others_signals = set()
for aid, ai in all_interps.items():
    others_signals.update(ai.get("key_signals", []))
unique = my_signals - others_signals
Q = min(len(unique) / max(len(my_signals), 1), 1.0)

score = w["C"]*C + w["U"]*U + w["E"]*E + w["Q"]*Q
return round(score, 4)
```

def compute_entropy(interps: dict, error_streak: int, flux: float) -> float:
“”“H(t) = α·D̂iv + β·Êrr + γ·F̂lux — взвешенная сумма.”””
# Дивергенция через разброс confidence
confidences = [v.get(“confidence”, 0.5) for v in interps.values()]
div_raw  = float(np.std(confidences)) if len(confidences) > 1 else 0.0
div_norm = div_raw / (div_raw + 1)

```
err_norm  = error_streak / (error_streak + 1)

flux_raw  = max(flux - 1.0, 0.0)
flux_norm = flux_raw / (flux_raw + 1)

H = 0.4 * div_norm + 0.35 * err_norm + 0.25 * flux_norm
return round(H, 4)
```

# ─────────────────────────────────────────────

# ГЛАВНЫЙ ДВИЖОК

# ─────────────────────────────────────────────

class DominantaXEngine:

```
def __init__(self, api_key: str):
    self.client    = anthropic.Anthropic(api_key=api_key)
    self.agents    = {aid: Agent(aid, self.client) for aid in AGENT_PROMPTS}
    self.arbitrator = Arbitrator(self.client)
    self.preprocessor = VideoPreprocessor()

    # Состояние системы
    self.dominant_id    = None   # текущий доминирующий агент
    self.dominant_interp = None  # текущая доминанта D*
    self.dominant_age   = 0
    self.error_streak   = 0
    self.prev_s         = None
    self.cycle          = 0
    self.frozen         = False

    self.log = []

def run_cycle(self, frame: np.ndarray) -> dict:
    """Один полный цикл движка."""
    self.cycle += 1
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"Цикл {self.cycle} | {ts}")

    # ── Фаза OBSERVE ──
    s_curr   = self.preprocessor.extract(frame)
    delta_s  = compute_delta(s_curr, self.prev_s)
    self.prev_s = s_curr

    flux = s_curr["flow_speed"] / (self.prev_s.get("flow_speed", 1.0) + 0.001) \
           if self.prev_s else 1.0

    print(f"ΔS: persons={delta_s['person_count']:+d}  "
          f"flow={delta_s['flow_speed']:+.2f}  "
          f"entropy={delta_s['scene_entropy']:+.3f}")

    if self.frozen:
        print("⏸  СИСТЕМА ЗАМОРОЖЕНА — ждём накопления сигнала")
        if abs(delta_s["scene_entropy"]) > 0.1 or abs(delta_s["person_count"]) > 5:
            self.frozen = False
            print("   Размораживаем — обнаружен новый сигнал")
        else:
            return {"cycle": self.cycle, "decision": "FREEZE", "dominant": self.dominant_id}

    # ── Фаза GENERATE — независимые интерпретации ──
    print("⚙  Генерация интерпретаций...")
    first_pass = {}
    for aid, agent in self.agents.items():
        first_pass[aid] = agent.interpret(delta_s)

    # ── Фаза DISCUSS (Задержка Каминского) ──
    print("💬 Дискуссия агентов...")
    final_interps = {}
    for aid, agent in self.agents.items():
        others = {k: v for k, v in first_pass.items() if k != aid}
        dominant_str = json.dumps(self.dominant_interp) if self.dominant_interp else None
        final_interps[aid] = agent.interpret(delta_s, others, dominant_str)

    # ── Фаза SCORE ──
    scenario_class = self._classify_scenario(delta_s, flux)
    scores = {}
    for aid, interp in final_interps.items():
        others_without = {k: v for k, v in final_interps.items() if k != aid}
        scores[aid] = score_agent(interp, others_without, delta_s, scenario_class)

    best_agent = max(scores, key=scores.get)
    print(f"📊 Scores: " + " | ".join(f"{k}={v:.2f}" for k, v in scores.items()))
    print(f"   Кандидат на доминанту: {best_agent} ({scores[best_agent]:.2f})")

    # ── Фаза ARBITRATE ──
    H = compute_entropy(final_interps, self.error_streak, flux)

    arb_payload = {
        "cycle": self.cycle,
        "delta_S": delta_s,
        "scores": scores,
        "interpretations": {
            aid: {
                "interpretation": interp.get("interpretation", ""),
                "forecast": interp.get("forecast", ""),
                "score": scores[aid],
                "key_signals": interp.get("key_signals", []),
                "confidence": interp.get("confidence", 0.5),
            }
            for aid, interp in final_interps.items()
        },
        "current_dominant": {
            "agent": self.dominant_id,
            "interpretation": json.dumps(self.dominant_interp) if self.dominant_interp else "None",
            "age_cycles": self.dominant_age,
            "error_streak": self.error_streak,
        },
        "system_metrics": {
            "entropy_H": H,
            "flux": round(flux, 3),
            "automatism_levels": {aid: round(ag.automatism, 2)
                                  for aid, ag in self.agents.items()},
            "arbitrator_last_5": self.arbitrator.history,
        }
    }

    print(f"🧠 Арбитр анализирует... (H={H:.2f}, flux={flux:.2f})")
    arb_result = self.arbitrator.decide(arb_payload)
    decision   = arb_result.get("decision", "FREEZE")

    print(f"⚖️  Решение: {decision} | Уверенность: {arb_result.get('confidence', 0):.2f}")
    print(f"   {arb_result.get('reasoning', '')}")

    if arb_result.get("recommended_action"):
        print(f"⚠️  ОПЕРАТОРУ: {arb_result['recommended_action']}")

    # ── Фаза STABILIZE / SWITCH / FREEZE ──
    if decision == "SWITCH":
        new_dom = arb_result.get("new_dominant") or best_agent
        print(f"🔄 СМЕНА ДОМИНАНТЫ: {self.dominant_id} → {new_dom}")
        self.dominant_id    = new_dom
        self.dominant_interp = final_interps[new_dom]
        self.dominant_age   = 0
        self.error_streak   = 0

    elif decision == "HOLD":
        if self.dominant_id is None:
            self.dominant_id    = best_agent
            self.dominant_interp = final_interps[best_agent]
        self.dominant_age += 1
        print(f"✅ УДЕРЖАНИЕ: {self.dominant_id} (возраст: {self.dominant_age} циклов)")

    elif decision == "FREEZE":
        self.frozen = True
        print("⏸  ЗАМОРОЗКА — накапливаем сигнал")

    # ── Фаза LEARN ──
    if self.dominant_id and self.dominant_id in self.agents:
        dom_agent  = self.agents[self.dominant_id]
        dom_interp = final_interps.get(self.dominant_id, {})
        forecast   = dom_interp.get("forecast", "")
        # Упрощённая проверка: если прогноз непустой — считаем что есть шанс
        # В реальной системе сравниваем с S(t+1)
        forecast_ok = len(forecast) > 20
        dom_agent.update_rating(forecast_ok)
        if not forecast_ok:
            self.error_streak += 1
        else:
            self.error_streak = max(0, self.error_streak - 1)

    # Итог цикла
    result = {
        "cycle":     self.cycle,
        "timestamp": ts,
        "delta_S":   delta_s,
        "scores":    scores,
        "entropy":   H,
        "decision":  decision,
        "dominant":  self.dominant_id,
        "dominant_age": self.dominant_age,
        "interpretation": final_interps.get(self.dominant_id, {}).get("interpretation", ""),
        "flags":     arb_result.get("flags", []),
        "recommended_action": arb_result.get("recommended_action"),
    }
    self.log.append(result)
    return result

def _classify_scenario(self, delta_s: dict, flux: float) -> str:
    """Классифицирует класс сценария κ."""
    magnitude = sum(abs(v) for v in delta_s.values())
    if magnitude > 5.0 or flux > 2.0:
        return "Local"
    elif magnitude > 1.0:
        return "Cluster"
    else:
        return "Structural"

def save_log(self, path: str = "dominanta_x_log.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(self.log, f, ensure_ascii=False, indent=2)
    print(f"\n📁 Лог сохранён: {path}")
```

# ─────────────────────────────────────────────

# СИНТЕТИЧЕСКИЙ ТЕСТ (без видео)

# ─────────────────────────────────────────────

def make_synthetic_frame(scenario: str) -> np.ndarray:
“”“Создаёт синтетический кадр для теста.”””
frame = np.zeros((480, 640, 3), dtype=np.uint8)

```
if scenario == "normal":
    # Несколько движущихся точек — нормальный поток
    for _ in range(20):
        x, y = np.random.randint(50, 590), np.random.randint(50, 430)
        cv2.circle(frame, (x, y), 8, (100, 150, 200), -1)

elif scenario == "crowd":
    # Много точек в одной зоне — скопление
    for _ in range(80):
        x = np.random.randint(200, 440)
        y = np.random.randint(150, 330)
        cv2.circle(frame, (x, y), 6, (200, 100, 100), -1)

elif scenario == "chaos":
    # Хаотичное движение — высокая энтропия
    for _ in range(50):
        x, y = np.random.randint(0, 640), np.random.randint(0, 480)
        cv2.rectangle(frame, (x-5, y-5), (x+5, y+5), (50, 200, 50), -1)
    # Случайный шум
    noise = np.random.randint(0, 60, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)

return frame
```

# ─────────────────────────────────────────────

# ТОЧКА ВХОДА

# ─────────────────────────────────────────────

def main():
parser = argparse.ArgumentParser(description=“Dominanta X Engine”)
parser.add_argument(”–video”, type=str, help=“Путь к видеофайлу (mp4, avi, …)”)
parser.add_argument(”–rtsp”,  type=str, help=“RTSP URL камеры”)
parser.add_argument(”–test”,  action=“store_true”, help=“Запуск с синтетическими данными”)
parser.add_argument(”–cycles”, type=int, default=10, help=“Число циклов (по умолчанию 10)”)
args = parser.parse_args()

```
print("=" * 60)
print("  DOMINANTA X — Движок конкурентной интерпретации")
print("  Концепция: Игорь Каминский")
print("=" * 60)

engine = DominantaXEngine(API_KEY)

# ── Режим: синтетический тест ──
if args.test:
    print("\n🧪 Режим синтетического теста")
    scenarios = ["normal"] * 3 + ["crowd"] * 3 + ["chaos"] * 2 + ["normal"] * 2
    for i, scenario in enumerate(scenarios[:args.cycles]):
        print(f"\n[Сценарий: {scenario.upper()}]")
        frame = make_synthetic_frame(scenario)
        time.sleep(0.5)  # имитация задержки потока
        engine.run_cycle(frame)

# ── Режим: видеофайл ──
elif args.video:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео: {args.video}")
        return

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    step     = int(fps * WINDOW_SECONDS)
    frame_n  = 0
    cycles   = 0

    print(f"\n🎬 Видео: {args.video} | FPS: {fps:.1f} | Окно: {WINDOW_SECONDS}с")

    while cycles < args.cycles:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        if frame_n % step != 0:
            continue
        engine.run_cycle(frame)
        cycles += 1

    cap.release()

# ── Режим: RTSP ──
elif args.rtsp:
    cap = cv2.VideoCapture(args.rtsp)
    if not cap.isOpened():
        print(f"❌ Не удалось подключиться к: {args.rtsp}")
        return

    print(f"\n📡 RTSP поток: {args.rtsp}")
    cycles = 0
    last_t = time.time()

    while cycles < args.cycles:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        if time.time() - last_t >= WINDOW_SECONDS:
            engine.run_cycle(frame)
            last_t = time.time()
            cycles += 1

    cap.release()

else:
    print("❌ Укажите источник: --video файл.mp4  |  --rtsp url  |  --test")
    return

engine.save_log()
print("\n✅ Готово.")
```

if **name** == “**main**”:
main()
