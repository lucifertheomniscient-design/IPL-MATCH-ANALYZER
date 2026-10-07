from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ART = joblib.load(Path(__file__).with_name("ipl_artifacts.pkl"))
MODEL, FEATS = ART["model"], ART["feats"]
RATING, SQUADS, VENUES = ART["latest"], ART["squads"], ART["valid_venues"]
ELO, VREC, VENUE_BFF = ART["elo"], ART["vrec"], ART["venue_bff"]
SMOOTH_K = 5

app = FastAPI(title="IPL Match Predictor")


def smooth(w, n):
    return (w + 0.5 * SMOOTH_K) / (n + SMOOTH_K)


def features(a, b, venue, toss_winner, toss_decision, players_a, players_b):
    """One feature row, with team `a` playing the role of team1 (same definitions as training)."""
    a_bats_first = int((toss_winner == a) == (toss_decision == "bat"))
    bf_rate = VENUE_BFF.get(venue, 0.5)
    mq = lambda ps: sum(RATING.get(p, 0.0) for p in ps) / len(ps)
    return {
        "elo_diff": ELO.get(a, 1500.0) - ELO.get(b, 1500.0),
        "venue_diff": smooth(*VREC.get((a, venue), [0, 0])) - smooth(*VREC.get((b, venue), [0, 0])),
        "mq_diff": mq(players_a) - mq(players_b),
        "t1_bats_first": a_bats_first,
        "t1_venue_bat_edge": (bf_rate - 0.5) * (1 if a_bats_first else -1),
    }


def prob_first_wins(*args):
    row = pd.DataFrame([features(*args)])[FEATS]
    return float(MODEL.predict_proba(row)[0, 1])


class PredictRequest(BaseModel):
    team1: str
    team2: str
    toss_winner: str
    toss_decision: Literal["bat", "field"]
    venue: str
    team1_players: list[str] = Field(min_length=3, max_length=3)
    team2_players: list[str] = Field(min_length=3, max_length=3)


def check(r: PredictRequest):
    errors = []
    for t in (r.team1, r.team2):
        if t not in SQUADS:
            errors.append(f"unknown team: {t}")
    if r.team1 == r.team2:
        errors.append("team1 and team2 must be different")
    if r.toss_winner not in (r.team1, r.team2):
        errors.append("toss_winner must be team1 or team2")
    if r.venue not in VENUES:
        errors.append(f"unknown venue: {r.venue}")
    for team, players in ((r.team1, r.team1_players), (r.team2, r.team2_players)):
        if len(set(players)) != 3:
            errors.append(f"{team}: pick 3 different players")
        bad = [p for p in players if p not in SQUADS.get(team, [])]
        if bad:
            errors.append(f"{team}: not in squad: {bad}")
    if errors:
        raise HTTPException(status_code=422, detail=errors)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options")
def options():
    return {"teams": sorted(SQUADS), "venues": VENUES, "squads": SQUADS,
            "toss_decisions": ["bat", "field"]}


@app.post("/predict")
def predict(r: PredictRequest):
    check(r)
    args = (r.venue, r.toss_winner, r.toss_decision)
    p_a = prob_first_wins(r.team1, r.team2, *args, r.team1_players, r.team2_players)
    p_b = prob_first_wins(r.team2, r.team1, *args, r.team2_players, r.team1_players)
    p1 = (p_a + (1 - p_b)) / 2          # average both orderings so the order of the teams doesn't matter
    winner = r.team1 if p1 >= 0.5 else r.team2
    return {
        "team1": r.team1, "team2": r.team2,
        "prob_team1": round(p1, 3), "prob_team2": round(1 - p1, 3),
        "predicted_winner": winner,
        "features": features(r.team1, r.team2, *args, r.team1_players, r.team2_players),
        "note": "IPL outcomes are close to a coin flip; treat this as a weak-edge probability.",
    }