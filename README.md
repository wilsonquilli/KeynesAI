# KeynesAI

KeynesAI is a stock market web app with a Flask backend and a React frontend. It combines live market snapshots, a stock prediction workflow, portfolio tracking, sector browsing, and Supabase-backed authentication with 65% prediction accuracy.

## What’s In The App

- `Trending Stocks`: Loads gainers, losers, and market movers from a cached multi-provider market data pipeline.
- `Predictions`: Shows momentum-based stock signals with confidence, horizon, rationale, and projected target prices.
- `Portfolio`: Lets authenticated users manage owned stocks, watchlists, and wishlists.
- `Authentication`: Login, registration, session handling, and portfolio persistence using Supabase.

## Current Stack

- Backend: Python, Flask
- Frontend: React
- Data processing: pandas, numpy
- Machine learning: scikit-learn
- Database/Auth: Supabase
- Market data: Nasdaq, Finnhub, Yahoo Finance fallbacks

## Project Structure

```text
KeynesAI/
├── backend/
│   ├── app.py
│   ├── stock.py
│   ├── boomCrash.py
│   ├── chart.py
│   ├── stock_tree.py
│   ├── requirements.txt
│   ├── supabase_schema.sql
│   └── .env
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Backend Features

- Flask API routes for:
  - auth status
  - login / register / logout
  - portfolio CRUD
  - trending market data
  - market predictions
- Cached market snapshot loading to keep the UI faster after the first request
- Parallel market-provider fetches for trending/predictions pages
- Prediction pipeline in `backend/stock.py` with:
  - technical features
  - horizon-based features
  - RandomForest-based signal generation
  - future prediction generation for the predictions page

## Frontend Features

- React SPA served separately through Vite during development
- API proxy from Vite to Flask for:
  - `/api`
  - `/login`
  - `/register`
  - `/logout`
- Pages:
  - Home
  - Predictions
  - Trending
  - Portfolio

## Installation

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

## Running The App

Start the backend:

```bash
cd backend
python3 app.py
```

Start the frontend in another terminal:

```bash
cd frontend
npm run dev
```

Then open the Vite app in the browser, usually:

```text
http://127.0.0.1:5173
```

The frontend proxies API traffic to the Flask backend at:

```text
http://127.0.0.1:5000
```

## Production Build

Frontend production build:

```bash
cd frontend
npm run build
```

## Contributors

- Mostafa Amer
- Nicholas Shvelidze
- Wilson Quilli

## Contact

For questions or support:

`wilo240105@gmail.com`