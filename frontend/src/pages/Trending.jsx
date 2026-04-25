import { useEffect, useState } from "react";
import { apiUrl, readResponsePayload } from "../lib/api.js";

function HeatBar({ chg }) {
  const val = Math.abs(parseFloat(chg));
  const up = !chg.startsWith("-");
  const pct = Math.min(val / 10, 1) * 100;

  return (
    <div style={{ width: "100%", height: 3, borderRadius: 3, background: "rgba(255,255,255,0.05)", overflow: "hidden", marginTop: 4 }}>
      <div style={{ height: "100%", width: `${pct}%`, background: up ? "var(--green)" : "var(--red)", borderRadius: 3, transition: "width 1s ease" }} />
    </div>
  );
}

function GainerRow({ item, rank }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "32px 72px 1fr 80px 80px 70px auto",
        alignItems: "center",
        gap: 12,
        padding: "14px 20px",
        borderBottom: "1px solid rgba(61,255,160,0.05)",
        transition: "background .15s",
      }}
      onMouseEnter={(event) => {
        event.currentTarget.style.background = "rgba(61,255,160,0.04)";
      }}
      onMouseLeave={(event) => {
        event.currentTarget.style.background = "transparent";
      }}
    >
      <span style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.72rem", color: "var(--text-3)", textAlign: "right" }}>#{rank}</span>
      <span style={{ fontFamily: "'Space Mono',monospace", fontWeight: 700, fontSize: "0.88rem" }}>{item.sym}</span>
      <div>
        <div style={{ fontSize: "0.82rem", color: "var(--text-2)" }}>{item.name}</div>
        <div style={{ fontSize: "0.7rem", color: "var(--text-3)", marginTop: 2 }}>{item.sector}</div>
      </div>
      <span style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.85rem", textAlign: "right" }}>${item.price}</span>
      <span style={{ fontFamily: "'Space Mono',monospace", fontWeight: 700, fontSize: "0.85rem", color: "var(--green)", textAlign: "right" }}>{item.chg}</span>
      <span style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.7rem", color: "var(--text-3)", textAlign: "right" }}>{item.vol}</span>
      <span style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.7rem", color: "var(--text-3)", textAlign: "right" }}>{item.mktCap}</span>
    </div>
  );
}

function LoserRow({ item, rank }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "32px 72px 1fr 80px 80px 1fr",
        alignItems: "center",
        gap: 12,
        padding: "14px 20px",
        borderBottom: "1px solid rgba(255,92,92,0.05)",
        transition: "background .15s",
      }}
      onMouseEnter={(event) => {
        event.currentTarget.style.background = "rgba(255,92,92,0.03)";
      }}
      onMouseLeave={(event) => {
        event.currentTarget.style.background = "transparent";
      }}
    >
      <span style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.72rem", color: "var(--text-3)", textAlign: "right" }}>#{rank}</span>
      <span style={{ fontFamily: "'Space Mono',monospace", fontWeight: 700, fontSize: "0.88rem" }}>{item.sym}</span>
      <div>
        <div style={{ fontSize: "0.82rem", color: "var(--text-2)" }}>{item.name}</div>
        <div style={{ fontSize: "0.7rem", color: "var(--text-3)", marginTop: 2 }}>{item.sector}</div>
      </div>
      <span style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.85rem", textAlign: "right" }}>${item.price}</span>
      <span style={{ fontFamily: "'Space Mono',monospace", fontWeight: 700, fontSize: "0.85rem", color: "var(--red)", textAlign: "right" }}>{item.chg}</span>
      <div style={{ paddingLeft: 8 }}>
        <div style={{ fontSize: "0.72rem", color: "var(--text-3)", marginBottom: 2 }}>{item.reason}</div>
        <HeatBar chg={item.chg} />
      </div>
    </div>
  );
}

function SectionHeader({ label, color, count }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "14px 20px",
        borderBottom: "1px solid var(--border)",
        background: `color-mix(in srgb, ${color} 4%, var(--surface))`,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{ width: 8, height: 8, borderRadius: "50%", background: color, boxShadow: `0 0 8px ${color}` }} />
        <span style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.72rem", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color }}>
          {label}
        </span>
      </div>
      <span style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.7rem", color: "var(--text-3)", padding: "2px 8px", borderRadius: 4, border: "1px solid var(--border)" }}>
        {count} stocks
      </span>
    </div>
  );
}

function Trending() {
  const [gainers, setGainers] = useState([]);
  const [losers, setLosers] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [metadata, setMetadata] = useState({ updatedAt: null, symbolsAnalyzed: 0, failures: [] });

  useEffect(() => {
    const loadTrending = async () => {
      setIsLoading(true);
      setError("");

      try {
        const response = await fetch(apiUrl("/api/market/trending"), {
          credentials: "include",
          headers: { Accept: "application/json" },
        });
        const data = await readResponsePayload(response);

        if (!response.ok || !data.success) {
          throw new Error(data.message || "Unable to load trending stocks.");
        }

        setGainers(data.gainers || []);
        setLosers(data.losers || []);
        setMetadata({
          updatedAt: data.updated_at || null,
          symbolsAnalyzed: data.symbols_analyzed || 0,
          failures: data.failures || [],
          source: data.source || "Finnhub",
        });
      } catch (loadError) {
        setError(loadError.message || "Unable to load trending stocks.");
      } finally {
        setIsLoading(false);
      }
    };

    loadTrending();
  }, []);

  const topGainer = gainers[0];
  const topLoser = losers[0];

  return (
    <>
      <div className="page-wrap">
        <div style={{ marginBottom: 40 }}>
          <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.72rem", letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--gold)", marginBottom: 10 }}>
            ● Trending Stocks
          </div>
          <h1 className="page-title">Market Movers</h1>
          <p className="page-subtitle">Live gainers and losers based on current Finnhub market data.</p>
        </div>

        {error ? (
          <div style={{ marginBottom: 24, padding: "14px 16px", borderRadius: 10, border: "1px solid rgba(255,92,92,0.25)", background: "rgba(255,92,92,0.08)", color: "var(--red)" }}>
            {error}
          </div>
        ) : null}

        {isLoading ? (
          <div style={{ textAlign: "center", padding: "80px 0", color: "var(--text-3)", fontFamily: "'Space Mono',monospace" }}>
            Loading live market movers...
          </div>
        ) : (
          <>
            {topGainer && topLoser ? (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 40 }}>
                <div style={{ padding: "24px 28px", borderRadius: 16, border: "1px solid rgba(61,255,160,0.2)", background: "linear-gradient(135deg, rgba(61,255,160,0.06), var(--surface))", position: "relative", overflow: "hidden" }}>
                  <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "var(--green)" }} />
                  <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.65rem", letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--green)", marginBottom: 14, display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--green)", display: "inline-block", boxShadow: "0 0 6px var(--green)", animation: "pulseDot 2s ease-in-out infinite" }} />
                    Top Gainer Today
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
                    <div>
                      <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: "2.2rem", lineHeight: 1, marginBottom: 4 }}>{topGainer.sym}</div>
                      <div style={{ color: "var(--text-2)", fontSize: "0.85rem" }}>{topGainer.name}</div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "1.4rem", color: "var(--green)", fontWeight: 700 }}>{topGainer.chg}</div>
                      <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.82rem", color: "var(--text-2)", marginTop: 4 }}>${topGainer.price}</div>
                    </div>
                  </div>
                </div>

                <div style={{ padding: "24px 28px", borderRadius: 16, border: "1px solid rgba(255,92,92,0.2)", background: "linear-gradient(135deg, rgba(255,92,92,0.06), var(--surface))", position: "relative", overflow: "hidden" }}>
                  <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "var(--red)" }} />
                  <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.65rem", letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--red)", marginBottom: 14, display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--red)", display: "inline-block" }} />
                    Biggest Drop Today
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
                    <div>
                      <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: "2.2rem", lineHeight: 1, marginBottom: 4 }}>{topLoser.sym}</div>
                      <div style={{ color: "var(--text-2)", fontSize: "0.85rem" }}>{topLoser.name}</div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "1.4rem", color: "var(--red)", fontWeight: 700 }}>{topLoser.chg}</div>
                      <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.82rem", color: "var(--text-2)", marginTop: 4 }}>${topLoser.price}</div>
                    </div>
                  </div>
                </div>
              </div>
            ) : null}

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
              <div style={{ border: "1px solid rgba(61,255,160,0.12)", borderRadius: 16, overflow: "hidden", background: "var(--surface)" }}>
                <SectionHeader label="Top Gainers" color="var(--green)" count={gainers.length} />
                <div style={{ display: "grid", gap: 12, padding: "10px 20px", borderBottom: "1px solid var(--border)", gridTemplateColumns: "32px 72px 1fr 80px 80px 70px auto" }}>
                  {["#", "Ticker", "Company", "Price", "Chg%", "Volume", "Mkt Cap"].map((label, index) => (
                    <span key={label} style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.6rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--text-3)", textAlign: index >= 3 ? "right" : "left" }}>{label}</span>
                  ))}
                </div>
                {gainers.map((item, index) => <GainerRow key={item.sym} item={item} rank={index + 1} />)}
              </div>

              <div style={{ border: "1px solid rgba(255,92,92,0.12)", borderRadius: 16, overflow: "hidden", background: "var(--surface)" }}>
                <SectionHeader label="Biggest Drops" color="var(--red)" count={losers.length} />
                <div style={{ display: "grid", gap: 12, padding: "10px 20px", borderBottom: "1px solid var(--border)", gridTemplateColumns: "32px 72px 1fr 80px 80px 1fr" }}>
                  {["#", "Ticker", "Company", "Price", "Chg%", "Reason"].map((label, index) => (
                    <span key={label} style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.6rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--text-3)", textAlign: index >= 3 && index < 5 ? "right" : "left" }}>{label}</span>
                  ))}
                </div>
                {losers.map((item, index) => <LoserRow key={item.sym} item={item} rank={index + 1} />)}
              </div>
            </div>
          </>
        )}

        <p style={{ marginTop: 32, color: "var(--text-3)", fontSize: "0.75rem", fontFamily: "'Space Mono',monospace", textAlign: "center" }}>
          {metadata.updatedAt ? `Updated ${new Date(metadata.updatedAt * 1000).toLocaleString()}. ` : ""}
          Scanned {metadata.symbolsAnalyzed} symbols from {metadata.source || "Finnhub"}.
          {metadata.failures.length ? ` ${metadata.failures.length} symbols were skipped due to API/data issues.` : ""}
        </p>
      </div>

      <style>{`
        @keyframes pulseDot {
          0%, 100% { box-shadow: 0 0 0 0 rgba(61,255,160,0.5); }
          50%       { box-shadow: 0 0 0 6px rgba(61,255,160,0); }
        }
      `}</style>
    </>
  );
}

export default Trending;
