import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import apiService from '../services/api';

const StartTraining = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    file: null,
    satellite: 'KOI',
    model: 'rf'
  });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const handleFileChange = (e) => {
    setFormData(prev => ({
      ...prev,
      file: e.target.files[0]
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.file) {
      setMessage('Please select a CSV file');
      return;
    }

    setLoading(true);
    setMessage('Uploading and starting training…');

    try {
      const result = await apiService.startTraining(
        formData.file,
        formData.satellite,
        formData.model
      );

      // Store the result for the progress page
      sessionStorage.setItem("lastStartPayload", JSON.stringify(result));

      // Navigate to progress page
      const url = new URL('/training-progress', window.location.origin);
      url.searchParams.set('job_id', result.job_id);
      if (result.status_url) {
        url.searchParams.set('status_url', result.status_url);
      }
      if (result.logs_url) {
        url.searchParams.set('logs_url', result.logs_url);
      }
      
      navigate(url.pathname + url.search);
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const showMessage = (type, text) => {
    setMessage(text);
  };

  return (
    <div className="wrap">
      <div className="card" style={{ maxWidth: '720px', margin: '0 auto' }}>
        <h1>Train Model</h1>
        <p className="hint">
          Upload a CSV, choose Satellite + Model, then start training. 
          You'll be redirected to a live status page.
        </p>

        <form onSubmit={handleSubmit}>
          <label>CSV File</label>
          <input 
            type="file" 
            accept=".csv" 
            required 
            onChange={handleFileChange}
          />

          <div className="row">
            <div>
              <label>Satellite</label>
              <select 
                value={formData.satellite}
                onChange={(e) => setFormData(prev => ({ ...prev, satellite: e.target.value }))}
                required
              >
                <option value="KOI">KOI</option>
                <option value="K2">K2</option>
                <option value="TOI">TOI</option>
              </select>
            </div>
            <div>
              <label>Model</label>
              <select 
                value={formData.model}
                onChange={(e) => setFormData(prev => ({ ...prev, model: e.target.value }))}
                required
              >
                <option value="xgb">xgb</option>
                <option value="rf">rf</option>
                <option value="dt">dt</option>
                <option value="grdb">grdb</option>
                <option value="logreg">logreg</option>
                <option value="svm">svm</option>
              </select>
            </div>
          </div>

          <button type="submit" disabled={loading}>
            {loading ? 'Starting...' : 'Start Training'}
          </button>
        </form>

        {message && (
          <div className={message.includes('Error') ? 'err' : 'hint'}>
            {message}
          </div>
        )}
      </div>
    </div>
  );
};

export default StartTraining;
