import { Link } from "react-router-dom";

function Navbar({ user, onOpenLogin, onOpenSignup, onLogout }) {
  return (
    <header className="navbar">
      <Link to="/" className="brand-mark">
        <img src="keynes_.png" alt="Keynes AI logo" />
      </Link>

      <nav className="nav-links">
        <Link to="/portfolio">Portfolio</Link>
        <Link to="/predictions">Predictions</Link>
        <Link to="/trending">Trending Stocks</Link>
      </nav>

      <div className="auth-buttons">
        {user ? (
          <>
            <span className="user-pill">{user.email}</span>
            <button type="button" className="btn lime" onClick={onLogout}>
              Log Out
            </button>
          </>
        ) : (
          <>
            <button type="button" className="btn lime ghost" onClick={onOpenSignup}>
              Sign Up
            </button>
            <button type="button" className="btn lime" onClick={onOpenLogin}>
              Log In
            </button>
          </>
        )}
      </div>
    </header>
  );
}

export default Navbar;
