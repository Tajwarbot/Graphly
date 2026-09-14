import React, { StrictMode, Component } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("Uncaught application error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#FFFFFF', padding: '24px', fontFamily: 'Inter, sans-serif' }}>
          <div style={{ maxWidth: '640px', width: '100%', border: '1px solid #e4e4e7', borderRadius: '12px', padding: '32px', backgroundColor: '#FFFFFF' }}>
            <div style={{ fontSize: '12px', fontFamily: 'JetBrains Mono, monospace', fontWeight: 'bold', textTransform: 'uppercase', marginBottom: '8px', color: '#0044FF' }}>
              Graphly could not display this view
            </div>
            <h1 style={{ fontSize: '20px', fontWeight: 'bold', margin: '0 0 16px 0', color: '#000000' }}>
              Your saved graphs are still here
            </h1>
            <pre style={{ backgroundColor: '#F5F5F5', border: '1px solid #000000', padding: '12px', fontSize: '12px', fontFamily: 'JetBrains Mono, monospace', overflowX: 'auto', marginBottom: '20px', whiteSpace: 'pre-wrap' }}>
              {this.state.error?.toString() || 'Unknown runtime error'}
            </pre>
            <div style={{ display: 'flex', gap: '12px' }}>
              <button
                type="button"
                onClick={() => window.location.reload()}
                style={{ border: '1px solid #e4e4e7', borderRadius: '12px', backgroundColor: '#000000', color: '#FFFFFF', padding: '8px 16px', fontWeight: 'bold', fontSize: '13px', cursor: 'pointer' }}
              >
                Reload Page
              </button>
              <button
                type="button"
                onClick={() => {
                  window.location.href = window.location.pathname;
                }}
                style={{ border: '1px solid #e4e4e7', borderRadius: '12px', backgroundColor: '#FFFFFF', color: '#000000', padding: '8px 16px', fontWeight: 'bold', fontSize: '13px', cursor: 'pointer' }}
              >
                Return to home
              </button>
            </div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
