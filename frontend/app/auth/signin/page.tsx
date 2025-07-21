"use client";

import { useState } from "react";
import { signIn } from "next-auth/react";
import { useRouter } from "next/navigation";

export default function SignIn() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const result = await signIn("credentials", {
        email,
        password,
        redirect: false,
      });

      if (result?.error) {
        setError("Invalid credentials");
      } else {
        router.push("/");
      }
    } catch {
      setError("An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: '#f9fafb',
      padding: '3rem 1rem'
    }}>
      <div style={{
        maxWidth: '28rem',
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: '2rem'
      }}>
        <div>
          <h2 style={{
            marginTop: '1.5rem',
            textAlign: 'center',
            fontSize: '1.875rem',
            fontWeight: '800',
            color: '#111827'
          }}>
            Sign in to Marvin
          </h2>
        </div>
        
        <form style={{
          marginTop: '2rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.5rem'
        }} onSubmit={handleSubmit}>
          
          <div style={{
            borderRadius: '0.375rem',
            boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
            display: 'flex',
            flexDirection: 'column',
            gap: '-1px'
          }}>
            <div>
              <label htmlFor="email" style={{ display: 'none' }}>
                Email address
              </label>
              <input
                id="email"
                name="email"
                type="email"
                autoComplete="email"
                required
                style={{
                  appearance: 'none',
                  borderRadius: '0.375rem 0.375rem 0 0',
                  position: 'relative',
                  display: 'block',
                  width: '100%',
                  padding: '0.5rem 0.75rem',
                  border: '1px solid #d1d5db',
                  color: '#111827',
                  fontSize: '0.875rem',
                  outline: 'none'
                }}
                placeholder="Email address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="password" style={{ display: 'none' }}>
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                style={{
                  appearance: 'none',
                  borderRadius: '0 0 0.375rem 0.375rem',
                  position: 'relative',
                  display: 'block',
                  width: '100%',
                  padding: '0.5rem 0.75rem',
                  border: '1px solid #d1d5db',
                  borderTop: 'none',
                  color: '#111827',
                  fontSize: '0.875rem',
                  outline: 'none'
                }}
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>

          {error && (
            <div style={{
              color: '#dc2626',
              fontSize: '0.875rem',
              textAlign: 'center'
            }}>
              {error}
            </div>
          )}

          <div>
            <button
              type="submit"
              disabled={loading}
              style={{
                position: 'relative',
                width: '100%',
                display: 'flex',
                justifyContent: 'center',
                padding: '0.5rem 1rem',
                border: '1px solid transparent',
                fontSize: '0.875rem',
                fontWeight: '500',
                borderRadius: '0.375rem',
                color: 'white',
                backgroundColor: '#4f46e5',
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.5 : 1,
                outline: 'none'
              }}
              onMouseEnter={(e) => {
                if (!loading) {
                  e.currentTarget.style.backgroundColor = '#4338ca';
                }
              }}
              onMouseLeave={(e) => {
                if (!loading) {
                  e.currentTarget.style.backgroundColor = '#4f46e5';
                }
              }}
            >
              {loading ? "Signing in..." : "Sign in"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
} 