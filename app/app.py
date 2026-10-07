import os
import time

import requests
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="IPL Match Predictor")


def wake_api(max_wait=180):
    """Ping /health until the API answers (a free-tier service can take 1-2 min to wake)."""
    start = time.time()
    while time.time() - start < max_wait:
        try:
            if requests.get(f"{API}/health", timeout=15).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


@st.cache_data(ttl=3600, show_spinner=False)
def get_options():
    if not wake_api():
        raise RuntimeError("API did not wake up in time")
    r = requests.get(f"{API}/options", timeout=30)
    r.raise_for_status()
    return r.json()


st.title("IPL Match Predictor")
st.caption("Pick the teams, toss, venue, and the 3 most impactful players on each side.")

with st.spinner("Waking up the prediction server (free tier, can take 1-2 minutes)..."):
    try:
        opt = get_options()
    except (requests.RequestException, RuntimeError):
        st.error("Can't reach the API right now. Refresh the page in a minute.")
        st.stop()

teams, squads = opt["teams"], opt["squads"]

c1, c2 = st.columns(2)
team1 = c1.selectbox("Team 1", teams)
team2 = c2.selectbox("Team 2", [t for t in teams if t != team1])
p1 = c1.multiselect(f"{team1}: 3 key players", squads[team1], max_selections=3, key=f"p1_{team1}")
p2 = c2.multiselect(f"{team2}: 3 key players", squads[team2], max_selections=3, key=f"p2_{team2}")

toss_winner = st.radio("Toss won by", [team1, team2], horizontal=True)
toss_decision = st.radio("Toss decision", opt["toss_decisions"], horizontal=True)
venue = st.selectbox("Venue", opt["venues"])

if st.button("Predict", type="primary"):
    if len(p1) != 3 or len(p2) != 3:
        st.warning("Pick exactly 3 players for each team.")
    else:
        payload = {"team1": team1, "team2": team2, "toss_winner": toss_winner,
                   "toss_decision": toss_decision, "venue": venue,
                   "team1_players": p1, "team2_players": p2}
        r = None
        with st.spinner("Predicting (the server may need a moment to wake up)..."):
            if wake_api():
                try:
                    r = requests.post(f"{API}/predict", json=payload, timeout=60)
                except requests.RequestException:
                    r = None
        if r is None:
            st.error("Can't reach the API. Try again in a minute.")
        elif r.status_code == 200:
            res = r.json()
            a, b = st.columns(2)
            a.metric(team1, f"{res['prob_team1']:.0%}")
            b.metric(team2, f"{res['prob_team2']:.0%}")
            st.progress(res["prob_team1"])
            st.success(f"Slight edge: {res['predicted_winner']}")
            st.caption(res["note"])
        else:
            st.error(str(r.json().get("detail")))

st.divider()
st.markdown(
    "<div style='text-align:center; opacity:0.6; font-size:0.85rem'>"
    "KEY FACTS:RCB once got all out for 49 · CSK have 7 trophies (5 IPL + 2 CLT20) · Thala for a reason 💛"
    "</div>",
    unsafe_allow_html=True,
)