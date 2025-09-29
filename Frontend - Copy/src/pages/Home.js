import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div className="wrap">
      <div className="hero">
        <h1>Hunting for Exoplanets with AI</h1>
        <p>
          Discover exoplanets using advanced machine learning models trained on Kepler, 
          K2, and TESS data. Upload your datasets, train models, and make predictions 
          with our comprehensive AI system.
        </p>
      </div>

      <div className="feature-grid">
        <div className="feature-card">
          <div className="feature-icon">🚀</div>
          <h3>Merge & Train</h3>
          <p>
            Combine multiple CSV datasets and train machine learning models 
            for exoplanet classification using various algorithms.
          </p>
          <Link to="/merge-train" className="cta-button">
            Start Training
          </Link>
        </div>

        <div className="feature-card">
          <div className="feature-icon">📊</div>
          <h3>Quick Training</h3>
          <p>
            Upload a single CSV file and quickly train models with 
            real-time progress tracking and performance metrics.
          </p>
          <Link to="/start-training" className="cta-button">
            Quick Train
          </Link>
        </div>

        <div className="feature-card">
          <div className="feature-icon">🔮</div>
          <h3>Predictions</h3>
          <p>
            Use trained models to predict exoplanet classifications 
            with confidence scores and detailed analysis.
          </p>
          <Link to="/prediction" className="cta-button">
            Make Predictions
          </Link>
        </div>

        <div className="feature-card">
          <div className="feature-icon">⚙️</div>
          <h3>Advanced Tools</h3>
          <p>
            Access advanced merging tools, detailed logs, and 
            comprehensive model management features.
          </p>
          <Link to="/merge-train-advanced" className="cta-button">
            Advanced Tools
          </Link>
        </div>
      </div>

      <div className="card" style={{ marginTop: '40px' }}>
        <h2>About the System</h2>
        <p>
          This AI system is designed for NASA Space Apps Challenge 2025, focusing on 
          exoplanet discovery and classification. It supports multiple satellite datasets 
          including Kepler (KOI), K2, and TESS (TOI) observations.
        </p>
        <div className="row" style={{ marginTop: '20px' }}>
          <div>
            <h3>Supported Models</h3>
            <ul>
              <li>XGBoost (XGB)</li>
              <li>Random Forest (RF)</li>
              <li>Decision Tree (DT)</li>
              <li>Support Vector Machine (SVM)</li>
              <li>Logistic Regression</li>
            </ul>
          </div>
          <div>
            <h3>Features</h3>
            <ul>
              <li>Real-time training progress</li>
              <li>Confusion matrix visualization</li>
              <li>Model performance metrics</li>
              <li>CSV data merging</li>
              <li>Batch predictions</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
