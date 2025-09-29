import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navigation = () => {
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <nav className="nav">
      <div className="nav-content">
        <div className="nav-brand">
          <div className="logo"></div>
          <h1 className="nav-title">Exoplanet AI System</h1>
        </div>
        <div className="nav-links">
          <Link 
            to="/" 
            className={`nav-link ${isActive('/') ? 'active' : ''}`}
          >
            Home
          </Link>
          <Link 
            to="/merge-train" 
            className={`nav-link ${isActive('/merge-train') ? 'active' : ''}`}
          >
            Merge & Train
          </Link>
          <Link 
            to="/start-training" 
            className={`nav-link ${isActive('/start-training') ? 'active' : ''}`}
          >
            Start Training
          </Link>
          <Link 
            to="/prediction" 
            className={`nav-link ${isActive('/prediction') ? 'active' : ''}`}
          >
            Prediction
          </Link>
          <Link 
            to="/merge-train-advanced" 
            className={`nav-link ${isActive('/merge-train-advanced') ? 'active' : ''}`}
          >
            Advanced
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
