import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div className="space-landing">
      {/* Header */}
      <header className="space-header">
        <div className="space-logo">
          <div className="planet-logo">
            <div className="planet-core"></div>
            <div className="planet-ring"></div>
          </div>
          <span className="logo-text">EXOPLANET AI</span>
        </div>
        <nav className="space-nav">
          <Link to="/">Home</Link>
          <Link to="/merge-train">Merge & Train</Link>
          <Link to="/start-training">Start Training</Link>
          <Link to="/prediction">Prediction</Link>
          <Link to="/merge-train-advanced">Advanced</Link>
          <div className="search-icon">🔍</div>
        </nav>
      </header>

      {/* Hero Section */}
      <section className="space-hero">
        <div className="hero-content">
          <div className="hero-text">
            <h1 className="hero-title">EXOPLANETS</h1>
            <h2 className="hero-subtitle">THE AI MISSION</h2>
            <p className="hero-description">
              Discover exoplanets using advanced machine learning models trained on Kepler, 
              K2, and TESS data. Explore the cosmos with AI-powered classification and 
              prediction systems.
            </p>
            <div className="hero-buttons">
              <Link to="/merge-train" className="cta-primary">
                Join Mission
              </Link>
              <button className="cta-secondary">
                <span className="cta-icon">🚀</span>
                Features
              </button>
            </div>
          </div>
          <div className="hero-astronaut">
            <div className="astronaut-figure">
              <div className="astronaut-helmet"></div>
              <div className="astronaut-body"></div>
              <div className="astronaut-glow"></div>
            </div>
          </div>
        </div>
      </section>

      {/* Mission Stats */}
      <section className="mission-stats">
        <div className="stat-card">
          <div className="stat-icon insight-icon">📊</div>
          <div className="stat-text">1091 Models Trained</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon orbiter-icon">🛰️</div>
          <div className="stat-text">3 Satellite Missions</div>
        </div>
        <div className="stat-card">
          <div className="stat-data">
            <div className="stat-row">
              <span className="stat-label">KOI:</span>
              <span className="stat-value">8,054</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">K2:</span>
              <span className="stat-value">2,393</span>
            </div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-data">
            <div className="stat-row">
              <span className="stat-label">STATUS:</span>
              <span className="stat-value">Active</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">MODEL:</span>
              <span className="stat-value">XGBoost</span>
            </div>
          </div>
        </div>
      </section>

      {/* Feature Cards */}
      <section className="feature-section">
        <div className="feature-grid">
          <div className="feature-card space-card">
            <div className="card-icon">🚀</div>
            <h3>Merge & Train</h3>
            <p>Combine datasets and train ML models for exoplanet classification</p>
            <Link to="/merge-train" className="card-button">Launch Training</Link>
          </div>

          <div className="feature-card space-card">
            <div className="card-icon">📊</div>
            <h3>Quick Training</h3>
            <p>Rapid model training with real-time progress tracking</p>
            <Link to="/start-training" className="card-button">Start Now</Link>
          </div>

          <div className="feature-card space-card">
            <div className="card-icon">🔮</div>
            <h3>Predictions</h3>
            <p>AI-powered exoplanet classification with confidence scores</p>
            <Link to="/prediction" className="card-button">Make Predictions</Link>
          </div>

          <div className="feature-card space-card">
            <div className="card-icon">⚙️</div>
            <h3>Mission Control</h3>
            <p>Advanced tools for comprehensive model management</p>
            <Link to="/merge-train-advanced" className="card-button">Access Tools</Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="space-footer">
        <div className="footer-content">
          <div className="footer-left">
            <span className="footer-title">EXOPLANET DISCOVERY</span>
          </div>
          <div className="footer-right">
            <div className="footer-data">
              <div className="data-grid">
                <div className="data-square blue"></div>
                <div className="data-square red"></div>
                <div className="data-square green"></div>
                <div className="data-square yellow"></div>
              </div>
              <span className="data-label">Training Data</span>
            </div>
            <div className="footer-mission">
              <span>Kepler Mission Data</span>
              <button className="distance-btn">Accuracy</button>
            </div>
          </div>
        </div>
      </footer>

          {/* Background Elements */}
          <div className="space-background">
            <div className="planet-surface"></div>
            <div className="space-stars"></div>
            <div className="atmosphere-glow"></div>
          </div>

          {/* Floating Particles */}
          <div className="floating-particles">
            {[...Array(20)].map((_, i) => (
              <div
                key={i}
                className="particle"
                style={{
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                  animationDelay: `${Math.random() * 6}s`,
                  animationDuration: `${4 + Math.random() * 4}s`
                }}
              />
            ))}
          </div>
        </div>
      );
    };

export default Home;
