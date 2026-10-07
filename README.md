# IPL Match Predictor 🏏

Predict the winner of an IPL match from the two teams, the toss, the venue and the three most impactful players on each side. Built with scikit-learn, FastAPI and Streamlit.

## Try it live

- App: https://ipl-predictor-app-ukt9.onrender.com/
- API docs: https://ipl-predictor-qosd.onrender.com/docs

Both run on Render's free tier and go to sleep when nobody is using them. The first visit after a break can take 1 to 2 minutes while they wake up. After that it is instant.

## What it does

You choose:

- the two teams
- who won the toss and what they chose (bat or field)
- the venue
- the 3 most impactful players for each team, picked from the current squads

The app shows each team's win probability and which side has the slight edge.

## How it works

**1. Data**
The Kaggle IPL Complete Dataset (matches.csv and deliveries.csv, 2008 to 2024). Matches with no result are removed, and team and venue names are cleaned (renamed franchises and duplicate stadium names are merged). That leaves 1,090 matches.

**2. Features**
Every feature is calculated using only information available before the match starts, so nothing from the result leaks in.

- Elo difference: overall team strength, updated after every match
- Venue record difference: how well each team has done at that ground
- Marquee difference: the rating of the 3 chosen players for team 1 minus team 2
- Bats first: whether team 1 bats first, worked out from the toss
- Venue bat-first edge: whether the side batting first tends to win at that ground

**3. Player ratings**
Each player gets a score for every match from ball-by-ball data: batting runs above the average, plus bowling runs saved and wickets. A player's rating is a blend of recent form (the last 10 or so matches count most) and career average. Players with very few matches are pulled towards average so one lucky game does not make a star.

**4. Model**
Logistic regression with strong regularisation. With only about 1,000 matches, simple models work better than complex ones.

**5. Prediction**
The API predicts twice, once with each team as "team 1", and averages the two. This way the order of the teams never changes the answer.

## Results

Tested the honest way: for each season from 2015 to 2024, train on all earlier seasons and predict that season (633 matches in total).

| Model | Accuracy | Log loss |
|---|---|---|
| Always predict 50% | 50% | 0.693 |
| Final model (with marquee players) | 52.3% | 0.692 |
| Same model without marquee players | 50.7% | 0.693 |

What this means:

- IPL matches are very close to a coin flip before they start. Treat the output as a weak edge, not a confident prediction.
- The marquee player feature has a small positive link with the winner (correlation of about 0.07 across all matches), which is why it stays in the model.
- Random forest and XGBoost were also tried. They did worse because they overfit on this little data.
- Weather was left out because the dataset does not contain it.

## API

Base URL: https://ipl-predictor-qosd.onrender.com

| Endpoint | What it does |
|---|---|
| GET /health | Check that the API is awake |
| GET /options | Teams, venues and squads for the dropdowns |
| POST /predict | Returns win probabilities for a match |

Example request to POST /predict:

```json
{
  "team1": "Royal Challengers Bengaluru",
  "team2": "Chennai Super Kings",
  "toss_winner": "Chennai Super Kings",
  "toss_decision": "field",
  "venue": "M Chinnaswamy Stadium",
  "team1_players": ["V Kohli", "Mohammed Siraj", "RM Patidar"],
  "team2_players": ["RA Jadeja", "M Pathirana", "DL Chahar"]
}
```

Example response (shortened):

```json
{
  "team1": "Royal Challengers Bengaluru",
  "team2": "Chennai Super Kings",
  "prob_team1": 0.436,
  "prob_team2": 0.564,
  "predicted_winner": "Chennai Super Kings"
}
```

Invalid input (an unknown team, a player who is not in that squad, a toss winner who is not playing) returns a 422 error with a list of what is wrong.

## Project structure

```
IPL-PREDICTOR/
├── api/
│   ├── main.py               FastAPI service
│   ├── ipl_artifacts.pkl     trained model and lookups
│   └── requirements.txt
├── app/
│   ├── app.py                Streamlit app
│   └── requirements.txt
└── .gitignore
```

## Run it locally

```bash
git clone https://github.com/SarthakSaxena12/IPL-PREDICTOR.git
cd IPL-PREDICTOR
```

Start the API:

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

In a second terminal, start the app:

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

The app talks to http://127.0.0.1:8000 by default. To point it somewhere else, set the `API_URL` environment variable.

## Deployment

Both parts are separate free web services on Render, deployed from this repo.

- API: Root Directory `api`, start command `uvicorn main:app --host 0.0.0.0 --port $PORT`
- App: Root Directory `app`, start command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- Both services have the environment variable `PYTHON_VERSION` set to `3.12.8`
- The app also has `API_URL` set to the API's public address

The package versions in `api/requirements.txt` are pinned to the ones used for training, because a saved model only loads reliably with the same scikit-learn version.

Because free services sleep, the app pings the API's /health endpoint and waits for it to wake up before asking for anything.

## Limitations

- The data ends with the 2024 season, so the squads are the ones from the last season in the data. Players who joined or moved teams after that will not appear in the dropdowns.
- No weather, pitch, injury or playing XI information.
- Player ratings come from ball-by-ball data only, so fielding is not counted.
- The model has a small edge at best. Please do not use it for betting.

## Tech stack

Python, pandas, scikit-learn, FastAPI, Streamlit, Render.

## Data

IPL Complete Dataset by patrickb1912 on Kaggle: https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020

Built by Sarthak as a learning project.
