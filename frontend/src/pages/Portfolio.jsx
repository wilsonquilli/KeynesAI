import { useEffect, useState } from "react";
import { apiUrl, readResponsePayload } from "../lib/api.js";

const TABS = [
  { id: "owned",    label: "Owned",        icon: "💼", desc: "Stocks you currently hold" },
  { id: "watchlist",label: "Watch List",   icon: "👁️", desc: "Stocks you're monitoring" },
  { id: "wishlist", label: "Want to Buy",  icon: "⭐", desc: "Stocks you plan to purchase" },
];

const EMPTY_PORTFOLIO = {
  owned: [],
  watchlist: [],
  wishlist: [],
};

function GainBadge({ value }) {
  const up = !String(value).startsWith("-");
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "3px 8px", borderRadius: 4,
      background: up ? "rgba(61,255,160,0.08)" : "rgba(255,92,92,0.08)",
      border: `1px solid ${up ? "rgba(61,255,160,0.2)" : "rgba(255,92,92,0.2)"}`,
      color: up ? "var(--green)" : "var(--red)",
      fontFamily: "'Space Mono', monospace", fontSize: "0.75rem", fontWeight: 700,
    }}>
      {up ? "▲" : "▼"} {String(value).replace(/^[+-]/, "")}
    </span>
  );
}

function EmptyState({ tab, onAdd }) {
  const msgs = {
    owned:     { headline: "No stocks tracked yet", sub: "Add your first holding below." },
    watchlist: { headline: "Watch list is empty",    sub: "Start monitoring stocks you care about." },
    wishlist:  { headline: "Wish list is empty",     sub: "Note down stocks you're planning to buy." },
  };
  return (
    <div style={{ textAlign: "center", padding: "64px 20px" }}>
      <div style={{ fontSize: "2.5rem", marginBottom: 16 }}>
        {tab === "owned" ? "💼" : tab === "watchlist" ? "👁️" : "⭐"}
      </div>
      <p style={{ fontFamily: "'DM Serif Display', serif", fontSize: "1.4rem", margin: "0 0 8px", color: "var(--text)" }}>
        {msgs[tab].headline}
      </p>
      <p style={{ color: "var(--text-3)", fontSize: "0.9rem", margin: "0 0 28px" }}>{msgs[tab].sub}</p>
      <button className="btn gold" onClick={onAdd}>+ Add Stock</button>
    </div>
  );
}

function AddModal({ tab, onClose, onAdd, isSaving }) {
  const [form, setForm] = useState({ sym: "", name: "", shares: "", avg: "", targetPrice: "", note: "" });
  const upd = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!form.sym.trim()) return;
    const entry = {
      sym: form.sym.toUpperCase().trim(),
      name: form.name.trim() || form.sym.toUpperCase().trim(),
      price: "—",
      chg: "—",
    };
    if (tab === "owned")     { entry.shares = Number(form.shares) || 0; entry.avg = Number(form.avg) || 0; }
    if (tab === "watchlist") { entry.note = form.note; }
    if (tab === "wishlist")  { entry.targetPrice = form.targetPrice; entry.note = form.note; }
    onAdd(entry);
  };

  const field = (label, key, type = "text", placeholder = "") => (
    <div style={{ display: "grid", gap: 6 }}>
      <label style={{ fontSize: "0.72rem", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--text-3)" }}>
        {label}
      </label>
      <input
        type={type} value={form[key]} placeholder={placeholder}
        onChange={e => upd(key, e.target.value)}
        style={{
          padding: "11px 14px", borderRadius: 8, border: "1px solid var(--border)",
          background: "var(--ink-3)", color: "var(--text)", fontSize: "0.9rem", outline: "none",
          transition: "border-color .2s",
        }}
        onFocus={e => e.target.style.borderColor = "var(--gold)"}
        onBlur={e => e.target.style.borderColor = "var(--border)"}
      />
    </div>
  );

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 200, display: "grid", placeItems: "center",
      padding: 20, background: "rgba(0,5,10,0.85)", backdropFilter: "blur(16px)",
    }} onClick={onClose}>
      <div style={{
        position: "relative", width: "min(100%,460px)", padding: "36px 32px",
        borderRadius: 20, border: "1px solid var(--border-2)", background: "var(--surface)",
        boxShadow: "0 32px 96px rgba(0,0,0,0.7)",
      }} onClick={e => e.stopPropagation()}>
        <button onClick={onClose} style={{
          position: "absolute", top: 14, right: 14, width: 32, height: 32,
          border: "1px solid var(--border)", borderRadius: 6, cursor: "pointer",
          background: "transparent", color: "var(--text-3)", fontSize: "0.85rem",
        }}>✕</button>

        <p style={{ margin: "0 0 6px", fontFamily: "'Space Mono',monospace", fontSize: "0.7rem", letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--gold)" }}>
          Add to {TABS.find(t => t.id === tab)?.label}
        </p>
        <h2 style={{ margin: "0 0 24px", fontFamily: "'DM Serif Display',serif", fontSize: "1.7rem" }}>
          {tab === "owned" ? "Log a holding" : tab === "watchlist" ? "Watch a stock" : "Add to wish list"}
        </h2>

        <form onSubmit={handleSubmit} style={{ display: "grid", gap: 14 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 12 }}>
            {field("Ticker", "sym", "text", "AAPL")}
            {field("Company Name", "name", "text", "Apple Inc.")}
          </div>
          {tab === "owned" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {field("Shares Owned", "shares", "number", "10")}
              {field("Avg. Buy Price ($)", "avg", "number", "178.40")}
            </div>
          )}
          {tab === "wishlist" && field("Target Buy Price ($)", "targetPrice", "number", "440.00")}
          {(tab === "watchlist" || tab === "wishlist") && field("Notes", "note", "text", "Reason for watching…")}

          <button type="submit" disabled={isSaving} style={{
            marginTop: 8, padding: "13px", border: 0, borderRadius: 10,
            background: "var(--gold)", color: "var(--ink)", fontWeight: 800,
            fontSize: "0.85rem", letterSpacing: "0.08em", textTransform: "uppercase",
            cursor: isSaving ? "wait" : "pointer", boxShadow: "0 0 24px rgba(245,200,66,0.2)",
            opacity: isSaving ? 0.7 : 1,
          }}>
            {isSaving ? "Saving..." : "Add Stock"}
          </button>
        </form>
      </div>
    </div>
  );
}

function OwnedTable({ items, onRemove }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid var(--border)" }}>
            {["Ticker", "Company", "Shares", "Avg Cost", "Current", "Change", "P&L", ""].map(h => (
              <th key={h} style={{
                padding: "10px 16px", textAlign: h === "" ? "right" : "left",
                fontFamily: "'Space Mono',monospace", fontSize: "0.65rem",
                letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--text-3)",
                fontWeight: 400,
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.map((item) => {
            const current = parseFloat(item.price) || 0;
            const avg = parseFloat(item.avg) || 0;
            const shares = parseFloat(item.shares) || 0;
            const pl = current !== 0 ? ((current - avg) * shares).toFixed(2) : null;
            const plPct = (current !== 0 && avg !== 0) ? (((current - avg) / avg) * 100).toFixed(2) : null;
            const plUp = pl !== null ? parseFloat(pl) >= 0 : null;
            return (
              <tr key={item.id} style={{ borderBottom: "1px solid rgba(255,215,100,0.05)", transition: "background .15s" }}
                onMouseEnter={e => e.currentTarget.style.background = "rgba(255,215,100,0.03)"}
                onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                <td style={{ padding: "16px", fontFamily: "'Space Mono',monospace", fontWeight: 700, fontSize: "0.88rem" }}>{item.sym}</td>
                <td style={{ padding: "16px", color: "var(--text-2)", fontSize: "0.85rem" }}>{item.name}</td>
                <td style={{ padding: "16px", fontFamily: "'Space Mono',monospace", fontSize: "0.85rem" }}>{item.shares}</td>
                <td style={{ padding: "16px", fontFamily: "'Space Mono',monospace", fontSize: "0.85rem", color: "var(--text-2)" }}>
                  {avg ? `$${avg.toFixed(2)}` : "—"}
                </td>
                <td style={{ padding: "16px", fontFamily: "'Space Mono',monospace", fontSize: "0.85rem" }}>
                  {current ? `$${item.price}` : "—"}
                </td>
                <td style={{ padding: "16px" }}>{item.chg !== "—" ? <GainBadge value={item.chg} /> : <span style={{ color: "var(--text-3)" }}>—</span>}</td>
                <td style={{ padding: "16px" }}>
                  {pl !== null ? (
                    <span style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.82rem", fontWeight: 700, color: plUp ? "var(--green)" : "var(--red)" }}>
                      {plUp ? "+" : ""}${pl} <span style={{ fontWeight: 400, opacity: 0.7 }}>({plPct}%)</span>
                    </span>
                  ) : <span style={{ color: "var(--text-3)" }}>—</span>}
                </td>
                <td style={{ padding: "16px", textAlign: "right" }}>
                  <button onClick={() => onRemove(item.id)} style={{
                    border: "1px solid rgba(255,92,92,0.2)", borderRadius: 6, padding: "5px 10px",
                    background: "transparent", color: "var(--red)", cursor: "pointer", fontSize: "0.75rem",
                    opacity: 0.7, transition: "opacity .2s",
                  }}
                    onMouseEnter={e => e.currentTarget.style.opacity = 1}
                    onMouseLeave={e => e.currentTarget.style.opacity = 0.7}>
                    Remove
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function WatchCard({ item, onRemove }) {
  return (
    <div style={{
      padding: "20px 24px", border: "1px solid var(--border)", borderRadius: 12,
      background: "var(--surface)", transition: "all .2s", position: "relative",
    }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = "var(--border-2)"; e.currentTarget.style.transform = "translateY(-2px)"; }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.transform = "translateY(0)"; }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
        <div>
          <div style={{ fontFamily: "'Space Mono',monospace", fontWeight: 700, fontSize: "1rem", marginBottom: 4 }}>{item.sym}</div>
          <div style={{ color: "var(--text-2)", fontSize: "0.82rem" }}>{item.name}</div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.9rem" }}>{item.price !== "—" ? `$${item.price}` : "—"}</div>
          {item.chg !== "—" && <GainBadge value={item.chg} />}
        </div>
      </div>
      {item.note && (
        <div style={{
          padding: "8px 12px", borderRadius: 6, background: "rgba(245,200,66,0.05)",
          border: "1px solid rgba(245,200,66,0.1)", color: "var(--text-2)",
          fontSize: "0.8rem", fontStyle: "italic", marginBottom: 12,
        }}>
          "{item.note}"
        </div>
      )}
      <button onClick={onRemove} style={{
        border: "1px solid rgba(255,92,92,0.2)", borderRadius: 6, padding: "5px 12px",
        background: "transparent", color: "var(--red)", cursor: "pointer", fontSize: "0.75rem",
        opacity: 0.7,
      }}>Remove</button>
    </div>
  );
}

function WishCard({ item, onRemove }) {
  const current = parseFloat(item.price) || 0;
  const target = parseFloat(item.targetPrice) || 0;
  const belowTarget = target > 0 && current <= target;
  return (
    <div style={{
      padding: "20px 24px", border: `1px solid ${belowTarget ? "rgba(61,255,160,0.3)" : "var(--border)"}`,
      borderRadius: 12, background: "var(--surface)", transition: "all .2s", position: "relative",
    }}
      onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-2px)"; }}
      onMouseLeave={e => { e.currentTarget.style.transform = "translateY(0)"; }}>
      {belowTarget && (
        <div style={{
          position: "absolute", top: -10, right: 14, padding: "2px 10px", borderRadius: 4,
          background: "var(--green)", color: "var(--ink)", fontSize: "0.68rem",
          fontFamily: "'Space Mono',monospace", fontWeight: 700, letterSpacing: "0.08em",
        }}>BUY SIGNAL</div>
      )}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
        <div>
          <div style={{ fontFamily: "'Space Mono',monospace", fontWeight: 700, fontSize: "1rem", marginBottom: 4 }}>{item.sym}</div>
          <div style={{ color: "var(--text-2)", fontSize: "0.82rem" }}>{item.name}</div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.9rem" }}>{item.price !== "—" ? `$${item.price}` : "—"}</div>
          {item.chg !== "—" && <GainBadge value={item.chg} />}
        </div>
      </div>
      {item.targetPrice && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-3)", fontFamily: "'Space Mono',monospace" }}>TARGET:</span>
          <span style={{ fontSize: "0.82rem", fontFamily: "'Space Mono',monospace", color: "var(--gold)", fontWeight: 700 }}>${item.targetPrice}</span>
        </div>
      )}
      {item.note && (
        <div style={{
          padding: "8px 12px", borderRadius: 6, background: "rgba(245,200,66,0.05)",
          border: "1px solid rgba(245,200,66,0.1)", color: "var(--text-2)",
          fontSize: "0.8rem", fontStyle: "italic", marginBottom: 12,
        }}>
          "{item.note}"
        </div>
      )}
      <button onClick={onRemove} style={{
        border: "1px solid rgba(255,92,92,0.2)", borderRadius: 6, padding: "5px 12px",
        background: "transparent", color: "var(--red)", cursor: "pointer", fontSize: "0.75rem",
        opacity: 0.7,
      }}>Remove</button>
    </div>
  );
}

function Portfolio({ user, onOpenLogin }) {
  const [activeTab, setActiveTab] = useState("owned");
  const [stocks, setStocks] = useState(EMPTY_PORTFOLIO);
  const [showAdd, setShowAdd] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!user) {
      return;
    }

    const loadPortfolio = async () => {
      setIsLoading(true);
      setError("");

      try {
        const response = await fetch(apiUrl("/api/portfolio"), {
          credentials: "include",
          headers: {
            Accept: "application/json",
          },
        });

        const data = await readResponsePayload(response);
        if (!response.ok || !data.success) {
          throw new Error(data.message || "Unable to load portfolio.");
        }

        setStocks({
          owned: data.portfolio?.owned || [],
          watchlist: data.portfolio?.watchlist || [],
          wishlist: data.portfolio?.wishlist || [],
        });
      } catch (loadError) {
        setError(loadError.message || "Unable to load portfolio.");
      } finally {
        setIsLoading(false);
      }
    };

    loadPortfolio();
  }, [user]);

  const addStock = async (entry) => {
    setIsSaving(true);
    setError("");

    try {
      const response = await fetch(apiUrl("/api/portfolio/items"), {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          ...entry,
          category: activeTab,
        }),
      });

      const data = await readResponsePayload(response);
      if (!response.ok || !data.success) {
        throw new Error(data.message || "Unable to save portfolio item.");
      }

      setStocks((currentStocks) => ({
        ...currentStocks,
        [activeTab]: [...currentStocks[activeTab], data.item],
      }));
      setShowAdd(false);
    } catch (saveError) {
      setError(saveError.message || "Unable to save portfolio item.");
    } finally {
      setIsSaving(false);
    }
  };

  const removeStock = async (tab, itemId) => {
    setError("");

    try {
      const response = await fetch(apiUrl(`/api/portfolio/items/${itemId}`), {
        method: "DELETE",
        credentials: "include",
        headers: {
          Accept: "application/json",
        },
      });

      const data = await readResponsePayload(response);
      if (!response.ok || !data.success) {
        throw new Error(data.message || "Unable to delete portfolio item.");
      }

      setStocks((currentStocks) => ({
        ...currentStocks,
        [tab]: currentStocks[tab].filter((item) => item.id !== itemId),
      }));
    } catch (deleteError) {
      setError(deleteError.message || "Unable to delete portfolio item.");
    }
  };

  const ownedValue = stocks.owned.reduce((acc, s) => {
    const price = parseFloat(s.price) || 0;
    return acc + price * (s.shares || 0);
  }, 0);

  const ownedCost = stocks.owned.reduce((acc, s) => {
    return acc + (s.avg || 0) * (s.shares || 0);
  }, 0);

  const totalPL = ownedValue - ownedCost;
  const plUp = totalPL >= 0;

  return (
    <>
      <div className="page-wrap">

        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 40, flexWrap: "wrap", gap: 24 }}>
          <div>
            <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.72rem", letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--gold)", marginBottom: 10 }}>
              ● Portfolio
            </div>
            <h1 className="page-title">My Holdings</h1>
            <p className="page-subtitle">{user ? `Logged in as ${user.email}` : "Sign in to save your portfolio"}</p>
          </div>

          {/* Summary cards */}
          {user && (
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              <div style={{ padding: "16px 22px", border: "1px solid var(--border)", borderRadius: 12, background: "var(--surface)", minWidth: 140 }}>
                <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.62rem", letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--text-3)", marginBottom: 6 }}>Portfolio Value</div>
                <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: "1.5rem" }}>${ownedValue.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
              </div>
              <div style={{ padding: "16px 22px", border: `1px solid ${plUp ? "rgba(61,255,160,0.2)" : "rgba(255,92,92,0.2)"}`, borderRadius: 12, background: "var(--surface)", minWidth: 140 }}>
                <div style={{ fontFamily: "'Space Mono',monospace", fontSize: "0.62rem", letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--text-3)", marginBottom: 6 }}>Total P&L</div>
                <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: "1.5rem", color: plUp ? "var(--green)" : "var(--red)" }}>
                  {plUp ? "+" : ""}${totalPL.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Not logged in */}
        {!user && (
          <div style={{
            padding: "80px 40px", textAlign: "center", border: "1px solid var(--border)",
            borderRadius: 20, background: "var(--surface)",
          }}>
            <div style={{ fontSize: "3rem", marginBottom: 20 }}>🔒</div>
            <h2 style={{ fontFamily: "'DM Serif Display',serif", fontSize: "1.8rem", margin: "0 0 12px" }}>Sign in to track your portfolio</h2>
            <p style={{ color: "var(--text-2)", marginBottom: 28 }}>Keep track of stocks you own, watch, and want to buy.</p>
            <button className="btn gold" onClick={onOpenLogin}>Log In to Continue</button>
          </div>
        )}

        {/* Logged in content */}
        {user && (
          <>
            {error && (
              <div style={{
                marginBottom: 20, padding: "14px 16px", borderRadius: 10,
                border: "1px solid rgba(255,92,92,0.25)", background: "rgba(255,92,92,0.08)",
                color: "var(--red)",
              }}>
                {error}
              </div>
            )}

            {/* Tabs */}
            <div style={{ display: "flex", gap: 8, marginBottom: 28, flexWrap: "wrap" }}>
              {TABS.map(t => (
                <button key={t.id} onClick={() => setActiveTab(t.id)} style={{
                  display: "flex", alignItems: "center", gap: 8,
                  padding: "10px 20px", border: `1px solid ${activeTab === t.id ? "var(--gold)" : "var(--border)"}`,
                  borderRadius: 8, background: activeTab === t.id ? "rgba(245,200,66,0.1)" : "var(--surface)",
                  color: activeTab === t.id ? "var(--gold)" : "var(--text-2)",
                  cursor: "pointer", fontWeight: 700, fontSize: "0.82rem",
                  letterSpacing: "0.04em", transition: "all .2s",
                }}>
                  {t.icon} {t.label}
                  <span style={{
                    padding: "1px 7px", borderRadius: 20, background: activeTab === t.id ? "rgba(245,200,66,0.2)" : "rgba(255,255,255,0.06)",
                    fontFamily: "'Space Mono',monospace", fontSize: "0.7rem",
                    color: activeTab === t.id ? "var(--gold)" : "var(--text-3)",
                  }}>{stocks[t.id].length}</span>
                </button>
              ))}
              <button className="btn gold" style={{ marginLeft: "auto" }} onClick={() => setShowAdd(true)}>
                + Add Stock
              </button>
            </div>

            {/* Tab desc */}
            <p style={{ color: "var(--text-3)", fontSize: "0.82rem", marginBottom: 20, fontFamily: "'Space Mono',monospace", letterSpacing: "0.06em" }}>
              {TABS.find(t => t.id === activeTab)?.desc}
            </p>

            {/* Content */}
            <div style={{ border: "1px solid var(--border)", borderRadius: 16, background: "var(--surface)", overflow: "hidden" }}>
              {isLoading
                ? <div style={{ padding: "64px 20px", textAlign: "center", color: "var(--text-3)" }}>Loading your saved portfolio...</div>
                : stocks[activeTab].length === 0
                ? <EmptyState tab={activeTab} onAdd={() => setShowAdd(true)} />
                : activeTab === "owned"
                  ? <OwnedTable items={stocks.owned} onRemove={itemId => removeStock("owned", itemId)} />
                  : activeTab === "watchlist"
                    ? <div style={{ padding: 24, display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16 }}>
                        {stocks.watchlist.map((item) => <WatchCard key={item.id} item={item} onRemove={() => removeStock("watchlist", item.id)} />)}
                      </div>
                    : <div style={{ padding: 24, display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16 }}>
                        {stocks.wishlist.map((item) => <WishCard key={item.id} item={item} onRemove={() => removeStock("wishlist", item.id)} />)}
                      </div>
              }
            </div>
          </>
        )}
      </div>

      {showAdd && <AddModal tab={activeTab} onClose={() => setShowAdd(false)} onAdd={addStock} isSaving={isSaving} />}

    </>
  );
}

export default Portfolio;
