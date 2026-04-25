import { useEffect, useState } from "react";
import { apiUrl, readResponsePayload } from "../lib/api.js";
const HORIZONS = ["All", "7d", "14d", "30d"];
const DIRECTIONS = ["All", "up", "down"];

function ConfidenceBar({ value, direction }) {
  const color = direction === "up" ? "var(--green)" : "var(--red)";
  const tier = value >= 80 ? "HIGH" : value >= 65 ? "MED" : "LOW";
  const tierColor = value >= 80 ? "var(--green)" : value >= 65 ? "var(--gold)" : "var(--red)";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <div style={{ flex: 1, height: 4, borderRadius: 4, background: "rgba(255,255,255,0.06)", overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${value}%`, background: color, borderRadius: 4, transition: "width 1s ease" }} />
      </div>
      <span style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.72rem", color: "var(--text-3)", minWidth: 28, textAlign: "right" }}>
        {value}%
      </span>
      <span
        style={{
          padding: "2px 6px",
          borderRadius: 4,
          background: `color-mix(in srgb, ${tierColor} 12%, transparent)`,
          border: `1px solid color-mix(in srgb, ${tierColor} 25%, transparent)`,
          fontFamily: "'Space Mono',monospace",
          fontSize: "0.62rem",
          fontWeight: 700,
          letterSpacing: "0.1em",
          color: tierColor,
        }}
      >
        {tier}
      </span>
    </div>
  );
}

function PredictionCard({ item, idx }) {
  const [expanded, setExpanded] = useState(false);
  const up = item.direction === "up";
  const current = parseFloat(item.current);
  const target = parseFloat(item.target);
  const move = current ? (((target - current) / current) * 100).toFixed(1) : "0.0";
  const moveUp = parseFloat(move) >= 0;

  return (
    <div
      style={{
        border: `1px solid ${up ? "rgba(61,255,160,0.12)" : "rgba(255,92,92,0.12)"}`,
        borderRadius: 14,
        background: "var(--surface)",
        overflow: "hidden",
        transition: "all .25s",
        animation: "fadeUp 0.4s ease both",
        animationDelay: `${idx * 60}ms`,
      }}
      onMouseEnter={(event) => {
        event.currentTarget.style.transform = "translateY(-2px)";
        event.currentTarget.style.boxShadow = "0 12px 40px rgba(0,0,0,0.4)";
      }}
      onMouseLeave={(event) => {
        event.currentTarget.style.transform = "translateY(0)";
        event.currentTarget.style.boxShadow = "none";
      }}
    >
      <div style={{ height: 2, background: up ? "var(--green)" : "var(--red)", opacity: 0.6 }} />

      <div style={{ padding: "20px 22px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontFamily: "'Space Mono',monospace", fontWeight: 700, fontSize: "1rem" }}>{item.sym}</span>
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 5,
                padding: "3px 10px",
                borderRadius: 20,
                fontFamily: "'Space Mono',monospace",
                fontWeight: 700,
                fontSize: "0.7rem",
                letterSpacing: "0.06em",
                background: up ? "rgba(61,255,160,0.1)" : "rgba(255,92,92,0.1)",
                border: `1px solid ${up ? "rgba(61,255,160,0.25)" : "rgba(255,92,92,0.25)"}`,
                color: up ? "var(--green)" : "var(--red)",
              }}
            >
              {up ? "▲" : "▼"} {up ? "BULLISH" : "BEARISH"}
            </span>
          </div>
          <span
            style={{
              fontFamily: "'Space Mono',monospace",
              fontSize: "0.68rem",
              color: "var(--text-3)",
              padding: "3px 8px",
              borderRadius: 4,
              border: "1px solid var(--border)",
            }}
          >
            {item.horizon}
          </span>
        </div>

        <div style={{ color: "var(--text-2)", fontSize: "0.82rem", marginBottom: 14 }}>{item.name}</div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 16 }}>
          {[
            { label: "Current", val: `$${item.current}`, color: "var(--text)" },
            { label: "Target", val: `$${item.target}`, color: up ? "var(--green)" : "var(--red)" },
            { label: "Expected Move", val: `${moveUp ? "+" : ""}${move}%`, color: moveUp ? "var(--green)" : "var(--red)" },
          ].map(({ label, val, color }) => (
            <div key={label} style={{ padding: "10px 12px", borderRadius: 8, background: "rgba(255,255,255,0.03)", border: "1px solid var(--border)" }}>
              <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.58rem", color: "var(--text-3)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 4 }}>{label}</div>
              <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.88rem", fontWeight: 700, color }}>{val}</div>
            </div>
          ))}
        </div>

        <div style={{ marginBottom: 14 }}>
          <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.62rem", color: "var(--text-3)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6 }}>
            Confidence
          </div>
          <ConfidenceBar value={item.confidence} direction={item.direction} />
        </div>

        <button
          onClick={() => setExpanded((currentValue) => !currentValue)}
          style={{
            border: "none",
            background: "none",
            padding: 0,
            cursor: "pointer",
            fontFamily: "'Space Mono',monospace",
            fontSize: "0.7rem",
            color: "var(--gold)",
            letterSpacing: "0.06em",
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          {expanded ? "▼" : "▶"} {expanded ? "Hide rationale" : "View rationale"}
        </button>

        {expanded ? (
          <div
            style={{
              marginTop: 12,
              padding: "12px 14px",
              borderRadius: 8,
              background: "rgba(245,200,66,0.04)",
              border: "1px solid rgba(245,200,66,0.1)",
              color: "var(--text-2)",
              fontSize: "0.83rem",
              lineHeight: 1.65,
              fontStyle: "italic",
            }}
          >
            {item.rationale}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function Predictions() {
  const [predictions, setPredictions] = useState([]);
  const [horizon, setHorizon] = useState("All");
  const [direction, setDirection] = useState("All");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("confidence");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [metadata, setMetadata] = useState({ updatedAt: null, symbolsAnalyzed: 0, failures: [] });

  useEffect(() => {
    const loadPredictions = async () => {
      setIsLoading(true);
      setError("");

      try {
        const response = await fetch(apiUrl("/api/market/predictions"), {
          credentials: "include",
          headers: { Accept: "application/json" },
        });
        const data = await readResponsePayload(response);

        if (!response.ok || !data.success) {
          throw new Error(data.message || "Unable to load predictions.");
        }

        setPredictions(data.predictions || []);
        setMetadata({
          updatedAt: data.updated_at || null,
          symbolsAnalyzed: data.symbols_analyzed || 0,
          failures: data.failures || [],
          source: data.source || "Finnhub",
        });
      } catch (loadError) {
        setError(loadError.message || "Unable to load predictions.");
      } finally {
        setIsLoading(false);
      }
    };

    loadPredictions();
  }, []);
  const filtered = predictions
    .filter((item) => horizon === "All" || item.horizon === horizon)
    .filter((item) => direction === "All" || item.direction === direction)
    .filter((item) => !search || item.sym.includes(search.toUpperCase()) || item.name.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => (sort === "confidence" ? b.confidence - a.confidence : a.sym.localeCompare(b.sym)));

  const bullCount = filtered.filter((item) => item.direction === "up").length;
  const bearCount = filtered.filter((item) => item.direction === "down").length;
  const avgConf = filtered.length ? Math.round(filtered.reduce((total, item) => total + item.confidence, 0) / filtered.length) : 0;

  const FilterPill = ({ value, active, onClick, color }) => (
    <button
      onClick={onClick}
      style={{
        padding: "7px 14px",
        border: `1px solid ${active ? color || "var(--gold)" : "var(--border)"}`,
        borderRadius: 6,
        background: active ? `color-mix(in srgb, ${color || "var(--gold)"} 12%, transparent)` : "transparent",
        color: active ? color || "var(--gold)" : "var(--text-2)",
        cursor: "pointer",
        fontFamily: "'Space Mono',monospace",
        fontSize: "0.72rem",
        fontWeight: 700,
        letterSpacing: "0.06em",
        transition: "all .15s",
      }}
    >
      {value === "up" ? "▲ Bullish" : value === "down" ? "▼ Bearish" : value}
    </button>
  );

  return (
    <>
      <div className="page-wrap">
        <div style={{ marginBottom: 40 }}>
          <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.72rem", letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--gold)", marginBottom: 10 }}>
            ● AI Predictions
          </div>
          <h1 className="page-title">Market Intelligence</h1>
          <p className="page-subtitle">Finnhub market data plus momentum scoring to estimate which stocks may move up or down next.</p>
        </div>

        {error ? (
          <div style={{ marginBottom: 24, padding: "14px 16px", borderRadius: 10, border: "1px solid rgba(255,92,92,0.25)", background: "rgba(255,92,92,0.08)", color: "var(--red)" }}>
            {error}
          </div>
        ) : null}

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12, marginBottom: 32 }}>
          {[
            { label: "Total Signals", val: filtered.length, color: "var(--text)" },
            { label: "Bullish", val: bullCount, color: "var(--green)" },
            { label: "Bearish", val: bearCount, color: "var(--red)" },
            { label: "Avg Confidence", val: `${avgConf}%`, color: "var(--gold)" },
          ].map(({ label, val, color }) => (
            <div key={label} style={{ padding: "16px 20px", border: "1px solid var(--border)", borderRadius: 10, background: "var(--surface)" }}>
              <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.62rem", letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--text-3)", marginBottom: 6 }}>{label}</div>
              <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: "1.6rem", color }}>{val}</div>
            </div>
          ))}
        </div>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 28, alignItems: "center" }}>
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search ticker..."
            style={{
              padding: "8px 14px",
              borderRadius: 6,
              border: "1px solid var(--border)",
              background: "var(--ink-3)",
              color: "var(--text)",
              fontFamily: "'Space Mono',monospace",
              fontSize: "0.78rem",
              outline: "none",
              width: 160,
            }}
          />
          <div style={{ display: "flex", gap: 6 }}>
            {HORIZONS.map((value) => <FilterPill key={value} value={value} active={horizon === value} onClick={() => setHorizon(value)} />)}
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            {DIRECTIONS.map((value) => (
              <FilterPill
                key={value}
                value={value}
                active={direction === value}
                onClick={() => setDirection(value)}
                color={value === "up" ? "var(--green)" : value === "down" ? "var(--red)" : undefined}
              />
            ))}
          </div>
          <select
            value={sort}
            onChange={(event) => setSort(event.target.value)}
            style={{
              marginLeft: "auto",
              padding: "8px 12px",
              borderRadius: 6,
              border: "1px solid var(--border)",
              background: "var(--ink-3)",
              color: "var(--text-2)",
              fontFamily: "'Space Mono',monospace",
              fontSize: "0.72rem",
              outline: "none",
              cursor: "pointer",
            }}
          >
            <option value="confidence">Sort: Confidence</option>
            <option value="alpha">Sort: A-Z</option>
          </select>
        </div>

        {isLoading ? (
          <div style={{ textAlign: "center", padding: "80px 0", color: "var(--text-3)", fontFamily: "'Space Mono',monospace" }}>
            Loading live market predictions...
          </div>
        ) : filtered.length === 0 ? (
          <div style={{ textAlign: "center", padding: "80px 0", color: "var(--text-3)", fontFamily: "'Space Mono',monospace" }}>
            No predictions match your filters.
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: 16 }}>
            {filtered.map((item, index) => <PredictionCard key={item.sym} item={item} idx={index} />)}
          </div>
        )}

        <p style={{ marginTop: 40, color: "var(--text-3)", fontSize: "0.78rem", fontFamily: "'Space Mono',monospace", textAlign: "center", lineHeight: 1.7 }}>
          {metadata.updatedAt ? `Updated ${new Date(metadata.updatedAt * 1000).toLocaleString()}. ` : ""}
          Analyzed {metadata.symbolsAnalyzed} symbols from {metadata.source || "Finnhub"}.
          {metadata.failures.length ? ` ${metadata.failures.length} symbols were skipped due to API/data issues.` : ""}
        </p>
      </div>

      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </>
  );
}

export default Predictions;
