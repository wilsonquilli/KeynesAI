import { useEffect, useState } from "react";
import { Link, Navigate, Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar.jsx";
import Footer from "./components/Footer.jsx";
import LoginModal from "./components/LoginModal.jsx";
import Portfolio from "./pages/Portfolio.jsx";
import Predictions from "./pages/Predictions.jsx";
import Trending from "./pages/Trending.jsx";
import { readResponsePayload } from "./lib/api.js";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

const TICKERS = [
  { sym: "AAPL", name: "Apple Inc.", price: "213.49", chg: "+1.24%", up: true },
  { sym: "MSFT", name: "Microsoft Corp.", price: "421.07", chg: "+0.87%", up: true },
  { sym: "NVDA", name: "Nvidia Corp.", price: "875.39", chg: "+3.12%", up: true },
  { sym: "TSLA", name: "Tesla Inc.", price: "174.61", chg: "-1.88%", up: false },
  { sym: "GOOGL", name: "Alphabet Inc.", price: "178.20", chg: "+0.55%", up: true },
  { sym: "SPY", name: "S&P 500 ETF", price: "523.14", chg: "+0.42%", up: true },
  { sym: "IBM", name: "Intl. Bus. Machines", price: "168.92", chg: "-0.31%", up: false },
];

const MARQUEE_ITEMS = [...TICKERS, ...TICKERS];

function SparkBars({ up }) {
  const heights = up
    ? [6, 8, 5, 9, 7, 11, 10, 14, 12, 16]
    : [16, 13, 14, 10, 12, 9, 8, 7, 5, 6];
  const color = up ? "var(--green)" : "var(--red)";

  return (
    <div className="spark">
      {heights.map((height, index) => (
        <div
          key={`${up ? "up" : "down"}-${index}`}
          className="spark-bar"
          style={{ height: `${height}px`, background: color }}
        />
      ))}
    </div>
  );
}

function HomePage() {
  const handlePredictionSearch = (event) => {
    event.preventDefault();
    window.location.href = "/predictions";
  };

  return (
    <>
      <section className="hero">
        <div className="hero-eyebrow">AI-Powered Market Intelligence</div>

        <div className="hero-layout">
          <div>
            <h1 className="hero-headline">
              Predict Markets
              <br />
              with <em>Precision.</em>
            </h1>
            <p className="hero-sub">
              AI analysis inspired by John Maynard Keynes, blending classical economic thinking
              with modern machine learning to surface high-conviction trade ideas.
            </p>

            <form className="search-wrap" onSubmit={handlePredictionSearch}>
              <input type="text" placeholder="Search ticker, sector or keyword..." />
              <button type="submit" className="search-btn">
                🔍
              </button>
            </form>

            <div className="hero-ctas">
              <Link to="/portfolio" className="cta-card">
                <span className="cta-icon">📁</span>
                Your Portfolio
              </Link>
              <Link to="/predictions" className="cta-card">
                <span className="cta-icon">🤖</span>
                AI Predictions
              </Link>
              <Link to="/trending" className="cta-card">
                <span className="cta-icon">📈</span>
                Trending Now
              </Link>
            </div>
          </div>

          <div className="data-panel">
            <div className="panel-header">
              <span className="panel-title">Featured Stocks</span>
              <span className="panel-live">LIVE</span>
            </div>
            <ul className="ticker-list">
              {TICKERS.map((ticker) => (
                <li key={ticker.sym} className="ticker-row">
                  <span className="ticker-sym">{ticker.sym}</span>
                  <div>
                    <div className="ticker-name">{ticker.name}</div>
                    <SparkBars up={ticker.up} />
                  </div>
                  <span className="ticker-price">${ticker.price}</span>
                  <span className={`ticker-change ${ticker.up ? "up" : "down"}`}>{ticker.chg}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <div className="stats-bar">
        <div className="stat-cell">
          <div className="stat-label">Predictions Made</div>
          <div className="stat-value">14,892</div>
          <div className="stat-delta up">↑ 238 today</div>
        </div>
        <div className="stat-cell">
          <div className="stat-label">Accuracy Rate</div>
          <div className="stat-value">70%</div>
          <div className="stat-delta up">↑ vs 71% avg</div>
        </div>
        <div className="stat-cell">
          <div className="stat-label">Markets Covered</div>
          <div className="stat-value">42</div>
          <div className="stat-delta" style={{ color: "var(--text-3)" }}>
            Across 12 sectors
          </div>
        </div>
        <div className="stat-cell">
          <div className="stat-label">Active Users</div>
          <div className="stat-value">3,100+</div>
          <div className="stat-delta up">↑ 18% this month</div>
        </div>
      </div>

      <div className="marquee-wrap">
        <div className="marquee-label">MARKETS</div>
        <div className="marquee-track-wrap">
          <div className="marquee-track">
            {MARQUEE_ITEMS.map((ticker, index) => (
              <span key={`${ticker.sym}-${index}`} className="marquee-item">
                <span className="sym">{ticker.sym}</span>
                <span className="price">${ticker.price}</span>
                <span className={`chg ${ticker.up ? "up" : "down"}`}>{ticker.chg}</span>
                {index < MARQUEE_ITEMS.length - 1 ? <span className="marquee-sep">|</span> : null}
              </span>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

function App() {
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authMode, setAuthMode] = useState("login");
  const [user, setUser] = useState(null);

  useEffect(() => {
    const loadAuthStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/auth/status`, {
          credentials: "include",
          headers: { Accept: "application/json" },
        });

        const data = await readResponsePayload(response);
        if (!response.ok) {
          throw new Error(data.message || `Auth status request failed with ${response.status}.`);
        }

        if (data.logged_in) {
          setUser(data.user);
        }
      } catch (error) {
        console.error("Unable to read auth status", error);
      }
    };

    loadAuthStatus();
  }, []);

  const openAuthModal = (mode) => {
    setAuthMode(mode);
    setAuthModalOpen(true);
  };

  const closeAuthModal = () => {
    setAuthModalOpen(false);
  };

  const handleAuthSuccess = (nextUser) => {
    setUser(nextUser);
  };

  const handleLogout = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/logout`, {
        method: "POST",
        credentials: "include",
        headers: { Accept: "application/json" },
      });

      if (!response.ok) {
        throw new Error("Logout failed");
      }

      setUser(null);
    } catch (error) {
      console.error("Unable to log out", error);
    }
  };

  return (
    <>
      <Navbar
        user={user}
        onOpenLogin={() => openAuthModal("login")}
        onOpenSignup={() => openAuthModal("signup")}
        onLogout={handleLogout}
      />

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/portfolio" element={<Portfolio user={user} onOpenLogin={() => openAuthModal("login")} />} />
        <Route path="/predictions" element={<Predictions />} />
        <Route path="/trending" element={<Trending />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>

      <LoginModal
        key={`${authMode}-${authModalOpen ? "open" : "closed"}`}
        isOpen={authModalOpen}
        mode={authMode}
        onClose={closeAuthModal}
        onModeChange={setAuthMode}
        onAuthSuccess={handleAuthSuccess}
      />

      <Footer />
    </>
  );
}

export default App;
