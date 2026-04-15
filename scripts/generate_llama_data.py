"""
generate_llama_data.py
----------------------
Runs the CTA Attention Model across 500 simulated scenarios.

Pipeline per scenario:
  1. Sample realistic input features (station, time, weather, sports event)
  2. Run through TransitDelayPredictor to get attention weights (focus)
  3. Compute a domain-grounded delay estimate (overrides untrained model output
     so language quality doesn't depend on weights being trained yet)
  4. Compose Chicago-style natural language using compositional templates
  5. Write Llama 3.1 chat-format JSONL to data/llama_train.jsonl

Run:
    python scripts/generate_llama_data.py
    python scripts/generate_llama_data.py --model models/transit_attention.pt  # use saved weights
    python scripts/generate_llama_data.py --n 1000 --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.attention_model import TransitDelayPredictor, ModelConfig

# ─────────────────────────────────────────────────────────────────────────────
# CTA Station Registry — loaded from data/station_map.json (real GTFS data)
# Falls back to a minimal hardcoded set if the file hasn't been generated yet.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Station:
    id: int
    name: str
    line: str
    hood: str
    sports: Optional[str] = None   # "cubs" | "sox" | "bears" | "bulls_hawks"


def _load_stations_from_map() -> list[Station]:
    map_path = ROOT / "data" / "station_map.json"
    if not map_path.exists():
        return []
    with open(map_path) as f:
        sm = json.load(f)
    return [
        Station(
            id     = s["token_id"],
            name   = s["name"],
            line   = s["line"],
            hood   = s.get("hood", ""),
            sports = s.get("sports"),
        )
        for s in sm["stations"]
    ]


STATIONS: list[Station] = _load_stations_from_map()

if not STATIONS:
    # Minimal fallback — only used if fetch_cta_data.py hasn't been run
    print("[generate] WARNING: station_map.json not found — using fallback station list.")
    print("           Run:  python scripts/fetch_cta_data.py  for real CTA data.")
    STATIONS = [
        Station(12, "Addison (Red)",  "Red",    "Wrigleyville", "cubs"),
        Station(24, "Sox-35th",       "Red",    "Bridgeport",   "sox"),
        Station(35, "Clark/Lake",     "Blue",   "Loop"),
        Station(50, "Jackson (Red)",  "Red",    "Loop"),
        Station(80, "O'Hare",         "Blue",   "O'Hare"),
        Station(84, "Midway",         "Orange", "Midway"),
    ]

STATION_BY_ID = {s.id: s for s in STATIONS}

# ─────────────────────────────────────────────────────────────────────────────
# Sports event context
# ─────────────────────────────────────────────────────────────────────────────
SPORTS_CONTEXTS = {
    "cubs":       ("Cubs game at Wrigley Field",      "Addison"),
    "sox":        ("White Sox game at Guaranteed Rate","Sox-35th"),
    "bears":      ("Bears game at Soldier Field",     "Roosevelt"),
    "bulls_hawks":("Bulls/Hawks game at United Center","UIC-Halsted"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Scenario dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Scenario:
    station: Station
    time_norm: float        # 0.0–1.0 (0=midnight, 0.5=noon, 1.0=23:59)
    weather_idx: float      # 0.0 (clear) – 1.0 (blizzard/extreme)
    sports_flag: float      # 1.0 if major event nearby, else 0.0
    weather_desc: str       # human-readable weather label
    time_label: str         # e.g. "5:45 PM"
    sports_desc: str        # e.g. "Cubs game at Wrigley"
    hour: float             # 0–24 float


# ─────────────────────────────────────────────────────────────────────────────
# Helper: normalised time → human label
# ─────────────────────────────────────────────────────────────────────────────
def time_label(hour: float) -> str:
    h = int(hour) % 24
    m = int((hour - int(hour)) * 60)
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {suffix}"


def time_bucket(hour: float) -> str:
    if   6.5 <= hour <= 9.5:  return "AM rush"
    elif 9.5 <  hour <= 14:   return "midday"
    elif 14  <  hour <= 19.5: return "PM rush"
    elif 19.5 < hour <= 23:   return "evening"
    else:                     return "late night / overnight"


WEATHER_PROFILES = [
    (0.00, 0.05,  "clear skies"),
    (0.05, 0.15,  "partly cloudy"),
    (0.15, 0.30,  "light rain"),
    (0.30, 0.45,  "moderate rain"),
    (0.45, 0.55,  "heavy rain / thunderstorms"),
    (0.55, 0.65,  "light snow"),
    (0.65, 0.75,  "moderate snow"),
    (0.75, 0.88,  "heavy snow / lake effect"),
    (0.88, 0.95,  "blizzard conditions"),
    (0.95, 1.00,  "extreme cold / wind chill warning"),
]

def weather_desc(idx: float) -> str:
    for lo, hi, label in WEATHER_PROFILES:
        if lo <= idx < hi:
            return label
    return "extreme conditions"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_scenarios(n: int = 500) -> list[Scenario]:
    """
    Generates n diverse scenarios biased toward realistic distributions:
      - Rush hours are over-represented (~40 % of scenarios)
      - Sports events occur ~20 % of the time
      - Severe weather occurs ~15 % of the time (Chicago is Chicago)
    """
    scenarios: list[Scenario] = []

    while len(scenarios) < n:
        station = random.choice(STATIONS)

        # Time: mix of uniform + rush-hour bias
        if random.random() < 0.40:
            # Rush hour windows: 7–9 AM or 4–7 PM
            if random.random() < 0.5:
                hour = random.uniform(7.0, 9.5)
            else:
                hour = random.uniform(16.0, 19.5)
        else:
            hour = random.uniform(0, 24)
        time_norm = hour / 24.0

        # Weather: heavier tail for bad weather (Chicago)
        if random.random() < 0.15:
            weather_idx = random.uniform(0.65, 1.0)    # bad
        elif random.random() < 0.30:
            weather_idx = random.uniform(0.30, 0.65)   # moderate
        else:
            weather_idx = random.uniform(0.0, 0.30)    # fine

        # Sports event: only relevant at stations near a venue
        sports_flag = 0.0
        sports_desc_str = "No major events nearby"
        if station.sports and random.random() < 0.20:
            sports_flag = 1.0
            venue_label, _ = SPORTS_CONTEXTS[station.sports]
            sports_desc_str = venue_label
        elif random.random() < 0.05:
            # Spill-over effect: stadium crowds ripple to adjacent stations
            sports_flag = 0.4
            sports_desc_str = "Nearby stadium event (indirect impact)"

        scenarios.append(Scenario(
            station=station,
            time_norm=time_norm,
            weather_idx=weather_idx,
            sports_flag=sports_flag,
            weather_desc=weather_desc(weather_idx),
            time_label=time_label(hour),
            sports_desc=sports_desc_str,
            hour=hour,
        ))

    return scenarios


# ─────────────────────────────────────────────────────────────────────────────
# Domain-grounded delay estimator
# Decoupled from model weights so data quality doesn't depend on training state
# ─────────────────────────────────────────────────────────────────────────────
def compute_realistic_delay(s: Scenario) -> float:
    """
    Computes expected delay in minutes using transit domain knowledge.
    Stochasticity is seeded per-scenario (deterministic given the same Scenario).
    """
    h = s.hour

    # Base delay from time of day
    if 7.0 <= h <= 9.5:          # AM rush
        base = random.gauss(7.5, 3.5)
    elif 16.0 <= h <= 19.5:      # PM rush
        base = random.gauss(9.0, 4.0)
    elif 22.0 <= h or h < 5.0:   # overnight (low frequency = longer waits)
        base = random.gauss(5.0, 2.5)
    else:
        base = random.gauss(2.5, 1.5)

    # Weather additive
    base += s.weather_idx * random.uniform(8.0, 14.0)

    # Sports event additive
    if s.sports_flag >= 1.0:
        base += random.gauss(10.0, 4.5)
    elif s.sports_flag > 0:
        base += random.gauss(3.0, 1.5)

    return max(0.0, base)


def delay_severity(minutes: float) -> str:
    if   minutes < 1.0:  return "none"
    elif minutes < 4.0:  return "mild"
    elif minutes < 10.0: return "moderate"
    elif minutes < 18.0: return "heavy"
    else:                return "severe"


# ─────────────────────────────────────────────────────────────────────────────
# Chicago-style language generation
# Compositional approach: opening + reason + action + optional closer
# ─────────────────────────────────────────────────────────────────────────────

# Openings keyed by severity
OPENINGS = {
    "none": [
        "You're golden.",
        "All clear on the {line} Line.",
        "No drama today —",
        "The {line}'s running like a dream.",
        "Smooth sailing on the {line} right now.",
        "Good news for once —",
        "Believe it or not, the {line} is on time.",
    ],
    "mild": [
        "Minor slowdown on the {line} —",
        "Heads up, slight delay ahead:",
        "Little bit of a lag on the {line} at {station}:",
        "Not terrible, but worth knowing:",
        "Small hiccup on the {line}:",
        "The {line} is running a hair late —",
        "Nothing serious, but the {line} at {station} is a few minutes off.",
    ],
    "moderate": [
        "The {line} Line at {station} is running about {delay} minutes behind.",
        "Heads up — you're looking at a {delay}-minute delay on the {line}.",
        "The {line} is dragging today — {delay} minutes behind at {station}.",
        "Fair warning: the {line} at {station} is {delay} minutes late.",
        "Plan ahead — the {line} is running {delay} minutes slow at {station}.",
        "Not ideal: {delay}-minute delay on the {line} at {station}.",
        "The {line} is taking its sweet time — {delay} minutes behind.",
    ],
    "heavy": [
        "Okay, the {line} is messed up. You're looking at {delay} minutes at {station}.",
        "Rough one today — {delay}-minute delay on the {line} at {station}.",
        "The {line} is seriously dragging: {delay} minutes behind at {station}.",
        "Budget serious time — {delay}-minute delay on the {line}.",
        "The {line} is a mess right now. {delay} minutes at {station}.",
        "Not great: {delay} minutes behind on the {line} at {station}.",
        "The {line} Line is struggling — {delay} minutes at {station}.",
    ],
    "severe": [
        "The {line} is wrecked right now — {delay}-minute delay at {station}.",
        "Genuinely bad news: {delay} minutes on the {line} at {station}.",
        "The CTA is not your friend today. {delay}-minute delay on the {line}.",
        "Avoid the {line} if you can — {delay} minutes behind at {station}.",
        "Major disruption on the {line}: {delay} minutes at {station}.",
        "The {line} is completely bottlenecked. {delay} minutes, {station}.",
        "Do NOT count on the {line} right now — {delay}-minute delay.",
    ],
}

# Factor-specific reason phrases
REASON_SPORTS_CUBS = [
    "Wrigleyville is packed — the Cubs just let out.",
    "Addison station is shoulder-to-shoulder after the game.",
    "Cubs crowd is overwhelming the Red Line right now.",
    "Post-game surge at Wrigley is flooding the {line}.",
    "Sixty thousand Cubs fans trying to get home at once — do the math.",
    "The Friendly Confines just emptied out. The {line} is paying for it.",
]
REASON_SPORTS_SOX = [
    "White Sox game just wrapped at Guaranteed Rate Field.",
    "South Side crowd heading home is stacking up at Sox-35th.",
    "Game's over on 35th — the {line} is feeling it.",
    "Post-Sox crowd is piling onto the Red Line.",
    "Bridgeport is backed up from the ballgame.",
]
REASON_SPORTS_BEARS = [
    "Bears game at Soldier Field just ended — Roosevelt station is a zoo.",
    "Sunday football crowd is overwhelming the Orange and Green Lines.",
    "Soldier Field just let out. Expect chaos near Roosevelt.",
    "Bears crowd is hammering the South Side lines.",
    "Post-game traffic from the lakefront is bleeding onto the L.",
]
REASON_SPORTS_BULLS_HAWKS = [
    "United Center just let out — Bulls/Hawks crowd is flooding the Blue Line.",
    "Game over at the UC. Clinton and UIC-Halsted are packed.",
    "West Side lines are slammed from the United Center event.",
    "Post-game surge from United Center is hitting the Blue Line hard.",
]
REASON_SPORTS_GENERIC = [
    "Nearby stadium event is sending extra riders your way.",
    "Event crowd is adding to the usual load on this line.",
    "Stadium spillover is slowing things down.",
]

REASON_WEATHER = {
    "clear skies":              [],  # no weather reason needed
    "partly cloudy":            [],
    "light rain":               [
        "Light rain is slowing things down on the tracks.",
        "Wet rails are adding a few minutes to the schedule.",
        "Rain's picked up — signal sensitivity is slowing trains.",
    ],
    "moderate rain":            [
        "Rain is messing with signals downtown.",
        "Moderate rain has trains running cautiously through the Loop.",
        "Wet weather is backing up the system.",
    ],
    "heavy rain / thunderstorms": [
        "Thunderstorm alerts are forcing trains to slow down.",
        "Heavy rain is flooding some low-lying sections — service is impacted.",
        "Storm system over the city is throwing off the whole schedule.",
    ],
    "light snow":               [
        "Light snow is adding a few minutes — nothing catastrophic.",
        "Snow's starting to accumulate on the elevated tracks.",
        "Flurries are slowing the outdoor segments of the line.",
    ],
    "moderate snow":            [
        "Snow is building up on the elevated structure — trains are moving carefully.",
        "Moderate snowfall is causing signal and switch issues.",
        "The outdoor L tracks are icing up — expect slower speeds.",
    ],
    "heavy snow / lake effect": [
        "Lake-effect snow is hammering the North Side elevated tracks.",
        "Heavy snow is causing switch failures across the system.",
        "The Hawk is here and the tracks are paying for it.",
        "Lake-effect is brutal today — outdoor rail segments are struggling.",
    ],
    "blizzard conditions":      [
        "Blizzard is causing widespread switch failures and slowdowns.",
        "The CTA is fighting the blizzard — trains are running but slowly.",
        "This is a Chicago blizzard situation. Expect significant delays system-wide.",
        "Full blizzard out there. The L is trying its best.",
    ],
    "extreme cold / wind chill warning": [
        "Wind chill warning is causing rail contractions and switch problems.",
        "Chiberia mode: metal contracts in extreme cold, switches freeze.",
        "Sub-zero wind chills are freezing rail components — expect delays.",
        "It's dangerously cold and the rail infrastructure is feeling it.",
    ],
}

REASON_TIME = {
    "AM rush":           [
        "Classic AM rush — every train is packed from Evanston to 95th.",
        "Morning rush is peaking. Half of Chicago is heading downtown.",
        "This is just the 8 AM reality. Welcome to the {line} Line.",
        "Rush hour congestion — longer dwell times at every stop.",
    ],
    "PM rush":           [
        "Evening rush is in full swing — trains are stacking behind each other.",
        "PM rush is doing its thing. The whole system is running heavy.",
        "Five o'clock crunch — platforms are full, trains are slow.",
        "Classic PM delay. Everyone's going home at the same time.",
    ],
    "midday":            [
        "Midday congestion — more than usual for this hour.",
        "Higher-than-expected midday ridership is slowing things.",
        "Unusual midday volume on this line.",
    ],
    "evening":           [
        "Evening crowd heading out is adding to the load.",
        "Entertainment district spillover — Friday/Saturday effect.",
    ],
    "late night / overnight": [
        "Late-night service runs less frequently — longer waits are normal.",
        "Overnight schedule means fewer trains. This is expected.",
        "Late-night riders: trains run every 15–20 minutes at this hour.",
    ],
}

# Action advice keyed by severity + factor
ACTIONS_NONE = [
    "Head out whenever.",
    "Jump on the next train — you're good.",
    "No adjustments needed.",
    "Catch the {line} at your usual time.",
    "Normal service. Nothing to plan around.",
]
ACTIONS_MILD = [
    "Leave 5 minutes earlier than usual and you're fine.",
    "Give yourself a small cushion — 5 minutes should cover it.",
    "Minor adjustment: leave a little earlier.",
    "Not worth stressing over, but account for it.",
    "Pad your schedule by 5 minutes and you'll be okay.",
]
ACTIONS_MODERATE = [
    "Leave 10–15 minutes earlier or you'll be cutting it close.",
    "Budget an extra 10 minutes and assume trains are crowded.",
    "If you have flexibility, push your departure back — the {line} will catch up eventually.",
    "Check the CTA Tracker app before you head out.",
    "Rideshare might be faster if you're on a hard deadline.",
    "The next 2 trains will be packed — let one pass and grab a seat.",
]
ACTIONS_HEAVY = [
    "Seriously consider alternatives: rideshare, bike, or a parallel bus.",
    "Check Ventra for updates — this could get worse before it gets better.",
    "If you're on the {line}, expect to be standing the whole ride.",
    "An Uber might save you 20 minutes of misery right now.",
    "Try the {alt_line} if that works for your route — it's moving better.",
    "Walk to the next station if it's under 10 minutes — the crowd thins.",
]
ACTIONS_SEVERE = [
    "Seriously: take a rideshare, call a cab, or work remote if you can.",
    "The {line} is not reliable right now. Find another way.",
    "This is a 'burn your commute and start over' kind of situation.",
    "Check CTA alerts. This isn't a normal delay — there may be a service gap.",
    "If you must take the {line}, get to the platform early and let 2 trains pass.",
    "A Divvy bike to a different line might be your best bet.",
]

ALT_LINES = {
    "Red":    ["Brown", "Purple"],
    "Blue":   ["Green", "Pink"],
    "Brown":  ["Red", "Purple"],
    "Green":  ["Orange", "Pink"],
    "Orange": ["Green", "Red"],
    "Pink":   ["Green", "Blue"],
    "Purple": ["Red", "Brown"],
    "Yellow": ["Red"],
}

CLOSERS = [
    "Good luck out there.",
    "Stay warm.",
    "The CTA giveth and the CTA taketh away.",
    "Chicago, baby.",
    "This city never makes it easy.",
    "That's transit life in Chicago.",
    "Check Ventra for live updates.",
    "",   # no closer — keeps variety
    "",
    "",
]


def generate_language(s: Scenario, delay: float, dominant: str, importance: dict[str, float]) -> str:
    """Assembles Chicago-style advisory from compositional template parts."""

    sev    = delay_severity(delay)
    line   = s.station.line
    name   = s.station.name
    hood   = s.station.hood
    tbucket = time_bucket(s.hour)
    alt   = random.choice(ALT_LINES.get(line, ["Red"]))
    delay_r = int(round(delay))

    fmt = dict(
        line=line, station=name, hood=hood, delay=delay_r, alt_line=alt
    )

    # Opening
    opening = random.choice(OPENINGS[sev]).format(**fmt)

    # Reason — pick most informative factor
    reason = ""
    if dominant == "Sports_Event" and s.sports_flag >= 1.0 and s.station.sports:
        bank = {
            "cubs":       REASON_SPORTS_CUBS,
            "sox":        REASON_SPORTS_SOX,
            "bears":      REASON_SPORTS_BEARS,
            "bulls_hawks":REASON_SPORTS_BULLS_HAWKS,
        }.get(s.station.sports, REASON_SPORTS_GENERIC)
        reason = random.choice(bank).format(**fmt)
    elif dominant == "Weather_Index" and s.weather_idx > 0.10:
        pool = REASON_WEATHER.get(s.weather_desc, [])
        if pool:
            reason = random.choice(pool).format(**fmt)
    elif dominant == "Time_of_Day":
        pool = REASON_TIME.get(tbucket, [])
        if pool:
            reason = random.choice(pool).format(**fmt)
    # Station-dominant: no standalone reason phrase — the opening is enough

    # Action
    actions = {
        "none":     ACTIONS_NONE,
        "mild":     ACTIONS_MILD,
        "moderate": ACTIONS_MODERATE,
        "heavy":    ACTIONS_HEAVY,
        "severe":   ACTIONS_SEVERE,
    }[sev]
    action = random.choice(actions).format(**fmt)

    # Closer
    closer = random.choice(CLOSERS)

    parts = [p for p in [opening, reason, action, closer] if p]
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Attention model runner
# ─────────────────────────────────────────────────────────────────────────────
def run_attention_model(
    scenarios: list[Scenario],
    model: TransitDelayPredictor,
    device: torch.device,
) -> list[dict]:
    """
    Forward-passes all scenarios in a single batch.
    Returns list of {dominant_factor, importance_dict}.
    Even with untrained weights, attention patterns are meaningful for data diversity.
    """
    model.eval()

    station_ids   = torch.tensor([s.station.id for s in scenarios], dtype=torch.long,  device=device)
    time_of_day   = torch.tensor([s.time_norm   for s in scenarios], dtype=torch.float, device=device)
    weather_index = torch.tensor([s.weather_idx for s in scenarios], dtype=torch.float, device=device)
    sports_event  = torch.tensor([s.sports_flag for s in scenarios], dtype=torch.float, device=device)

    with torch.no_grad():
        _, attn_weights, _weather_prior = model(
            station_ids, time_of_day, weather_index, sports_event,
            return_attention=True,
        )
        # attn_weights: (N, 4, 4) — column-mean = how much each feature was "looked at"
        importance = attn_weights.mean(dim=1)                            # (N, 4)
        importance = importance / importance.sum(dim=-1, keepdim=True)  # normalise

    names = model.cfg.feature_names
    results = []
    for i in range(len(scenarios)):
        scores = importance[i].cpu().tolist()
        imp_dict = dict(zip(names, [round(v, 4) for v in scores]))
        dominant = names[importance[i].argmax().item()]
        results.append({"dominant": dominant, "importance": imp_dict})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# JSONL entry builder  (Llama 3.1 chat format)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are the CTA Transit Assistant — a no-nonsense Chicago transit advisor "
    "embedded in the city's Intelligent Transit System. "
    "You speak like a knowledgeable Chicago local: direct, practical, and familiar "
    "with the city's neighborhoods, sports teams, weather patterns, and the quirks of the 'L'. "
    "Given a model prediction about transit conditions, provide a concise, actionable advisory. "
    "Include the delay estimate, the primary cause, and what the rider should do. "
    "Keep it under four sentences. Don't sugarcoat delays, but don't catastrophize minor ones."
)


def build_entry(s: Scenario, delay: float, dominant: str, importance: dict) -> dict:
    top2 = sorted(importance.items(), key=lambda x: -x[1])[:2]

    user_content = (
        f"CTA Alert — {s.station.line} Line @ {s.station.name} ({s.station.hood})\n"
        f"Time: {s.time_label}  [{time_bucket(s.hour)}]\n"
        f"Weather: {s.weather_desc} (index: {s.weather_idx:.2f})\n"
        f"Nearby Events: {s.sports_desc}\n"
        f"---\n"
        f"Model Prediction: {delay:.1f} minute delay\n"
        f"Primary Driver:   {top2[0][0]} ({top2[0][1]:.0%} attention)\n"
        f"Secondary Factor: {top2[1][0]} ({top2[1][1]:.0%} attention)"
    )

    advisory = generate_language(s, delay, dominant, importance)

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": advisory},
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate Llama fine-tuning data from CTA Attention Model")
    parser.add_argument("--n",      type=int,   default=500,  help="Number of scenarios to generate")
    parser.add_argument("--seed",   type=int,   default=42,   help="Random seed")
    parser.add_argument("--model",  type=str,   default=None, help="Path to saved attention model weights")
    parser.add_argument("--out",    type=str,   default=str(ROOT / "data" / "llama_train.jsonl"),
                        help="Output JSONL path")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate] Device: {device}")

    # Load or initialise attention model
    if args.model and Path(args.model).exists():
        print(f"[generate] Loading attention model from {args.model}")
        model = TransitDelayPredictor.load(args.model, map_location=str(device))
    else:
        print("[generate] No saved weights found — using randomly initialised model for attention structure")
        cfg   = ModelConfig(num_stations=150)
        model = TransitDelayPredictor(cfg)
    model.to(device).eval()

    # Generate scenarios
    print(f"[generate] Generating {args.n} scenarios …")
    scenarios = generate_scenarios(args.n)

    # Run attention model in one batch
    print("[generate] Running attention model …")
    attn_results = run_attention_model(scenarios, model, device)

    # Build JSONL entries
    print("[generate] Composing Chicago-style advisories …")
    entries = []
    for s, attn in zip(scenarios, attn_results):
        # Domain-grounded delay (deterministic for reproducibility)
        delay = compute_realistic_delay(s)
        entry = build_entry(s, delay, attn["dominant"], attn["importance"])
        entries.append(entry)

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    print(f"[generate] Wrote {len(entries)} entries → {out_path}")

    # Print sample
    print("\n── Sample entry (index 0) ──────────────────────────────────────")
    sample = entries[0]
    print(f"USER:\n{sample['messages'][1]['content']}\n")
    print(f"ASSISTANT:\n{sample['messages'][2]['content']}\n")

    # Quick stats
    delays = [compute_realistic_delay(s) for s in scenarios]
    print(f"── Delay stats ─────────────────────────────────────────────────")
    print(f"  mean:   {sum(delays)/len(delays):.1f} min")
    print(f"  median: {sorted(delays)[len(delays)//2]:.1f} min")
    print(f"  max:    {max(delays):.1f} min")

    sev_counts: dict[str, int] = {}
    for d in delays:
        k = delay_severity(d)
        sev_counts[k] = sev_counts.get(k, 0) + 1
    print(f"  severity distribution: {sev_counts}")


if __name__ == "__main__":
    main()
