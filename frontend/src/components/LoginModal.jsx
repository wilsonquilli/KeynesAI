import { useEffect, useState } from "react";
import { readResponsePayload } from "../lib/api.js";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

const initialFormState = {
  email: "",
  password: "",
};

function LoginModal({ isOpen, mode, onClose, onModeChange, onAuthSuccess }) {
  const [form, setForm] = useState(initialFormState);
  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (!isOpen) {
      return undefined;
    }

    const handleEscape = (event) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  const isSignup = mode === "signup";

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((currentForm) => ({
      ...currentForm,
      [name]: value,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setSuccessMessage("");

    if (!form.email.trim() || !form.password.trim()) {
      setError("Please enter both your email and password.");
      return;
    }

    setIsSubmitting(true);

    try {
      const response = await fetch(`${API_BASE_URL}/${isSignup ? "register" : "login"}`, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          email: form.email.trim(),
          password: form.password,
        }),
      });

      const data = await readResponsePayload(response);

      if (!response.ok || !data.success) {
        throw new Error(data.message || `Authentication failed with status ${response.status}.`);
      }

      setSuccessMessage(data.message || "Success!");
      onAuthSuccess?.(data.user);

      window.setTimeout(() => {
        onClose();
      }, 500);
    } catch (submitError) {
      setError(submitError.message || "Something went wrong. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="auth-modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="auth-modal"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="auth-modal-title"
      >
        <button type="button" className="auth-modal-close" onClick={onClose} aria-label="Close authentication modal">
          x
        </button>

        <div className="auth-modal-copy">
          <p className="auth-modal-eyebrow">Account Access</p>
          <h2 id="auth-modal-title">{isSignup ? "Create your account" : "Welcome back"}</h2>
          <p className="auth-modal-subtitle">
            {isSignup
              ? "Sign up to save your progress and keep using Keynes AI."
              : "Log in to get back to your saved account."}
          </p>
        </div>

        <div className="auth-modal-toggle" role="tablist" aria-label="Authentication mode">
          <button
            type="button"
            className={!isSignup ? "is-active" : ""}
            onClick={() => {
              setForm(initialFormState);
              setError("");
              setSuccessMessage("");
              setIsSubmitting(false);
              onModeChange("login");
            }}
          >
            Log In
          </button>
          <button
            type="button"
            className={isSignup ? "is-active" : ""}
            onClick={() => {
              setForm(initialFormState);
              setError("");
              setSuccessMessage("");
              setIsSubmitting(false);
              onModeChange("signup");
            }}
          >
            Sign Up
          </button>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label htmlFor="auth-email">Email</label>
          <input
            id="auth-email"
            name="email"
            type="email"
            placeholder="you@example.com"
            value={form.email}
            onChange={handleChange}
            autoComplete="email"
            required
          />

          <label htmlFor="auth-password">Password</label>
          <input
            id="auth-password"
            name="password"
            type="password"
            placeholder={isSignup ? "Create a password" : "Enter your password"}
            value={form.password}
            onChange={handleChange}
            autoComplete={isSignup ? "new-password" : "current-password"}
            minLength={6}
            required
          />

          {error ? <div className="auth-message auth-message-error">{error}</div> : null}
          {successMessage ? <div className="auth-message auth-message-success">{successMessage}</div> : null}

          <button type="submit" className="auth-submit" disabled={isSubmitting}>
            {isSubmitting ? "Please wait..." : isSignup ? "Create Account" : "Log In"}
          </button>
        </form>

        <p className="auth-modal-footer">
          {isSignup ? "Already have an account?" : "Need an account?"}{" "}
          <button
            type="button"
            className="auth-inline-button"
            onClick={() => {
              setForm(initialFormState);
              setError("");
              setSuccessMessage("");
              setIsSubmitting(false);
              onModeChange(isSignup ? "login" : "signup");
            }}
          >
            {isSignup ? "Log in here" : "Sign up here"}
          </button>
        </p>
      </div>
    </div>
  );
}

export default LoginModal;
