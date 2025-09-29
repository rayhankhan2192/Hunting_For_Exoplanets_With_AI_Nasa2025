import React, { useState, useRef } from 'react';
import apiService from '../services/api';

const Prediction = () => {
  const [formData, setFormData] = useState({
    satellite: 'KOI',
    fromRow: '',
    toRow: '',
    apiBase: 'http://203.190.12.138:8080'
  });
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [pageSize, setPageSize] = useState(20);
  const [currentPage, setCurrentPage] = useState(1);
  const [toast, setToast] = useState({ show: false, message: '', type: 'success' });
  const fileInputRef = useRef(null);

  const showToast = (message, type = 'success') => {
    setToast({ show: true, message, type });
    setTimeout(() => setToast({ show: false, message: '', type: 'success' }), 2500);
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      showToast('File attached');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type === 'text/csv') {
      setFile(droppedFile);
      showToast('File attached');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handlePredict = async () => {
    if (!file) {
      showToast('Please choose a CSV file', 'error');
      return;
    }

    setLoading(true);
    try {
      const options = {};
      if (formData.fromRow) options.from_csv_range = formData.fromRow;
      if (formData.toRow) options.to_csv_range = formData.toRow;

      const result = await apiService.predict(file, formData.satellite, options);
      setResults(result);
      setCurrentPage(1);
      showToast('Prediction complete');
    } catch (error) {
      showToast(`Error: ${error.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setFormData({
      satellite: 'KOI',
      fromRow: '',
      toRow: '',
      apiBase: 'http://203.190.12.138:8080'
    });
    setResults(null);
    setSearchTerm('');
    setCurrentPage(1);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    showToast('Reset');
  };

  const getFilteredResults = () => {
    if (!results?.results) return [];
    
    if (!searchTerm) return results.results;
    
    return results.results.filter(row => {
      return Object.values(row).some(value => 
        value && String(value).toLowerCase().includes(searchTerm.toLowerCase())
      );
    });
  };

  const getPaginatedResults = () => {
    const filtered = getFilteredResults();
    const start = (currentPage - 1) * pageSize;
    return filtered.slice(start, start + pageSize);
  };

  const getTotalPages = () => {
    return Math.ceil(getFilteredResults().length / pageSize);
  };

  const getClassCounts = () => {
    if (!results?.results) return [];
    
    const counts = new Map();
    results.results.forEach(row => {
      const className = row.Predicted_Class || row['Predicted Class'] || row.Predicted || 'Unknown';
      counts.set(className, (counts.get(className) || 0) + 1);
    });
    
    return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  };

  const formatProbability = (value) => {
    if (value == null || isNaN(value)) return '–';
    return (Number(value) * 100).toFixed(2) + '%';
  };

  const getClassBadge = (value) => {
    if (!value) return '';
    
    const className = String(value).toUpperCase();
    let badgeClass = '';
    
    if (className.includes('CONFIRMED')) badgeClass = 'b-confirmed';
    else if (className.includes('CANDIDATE')) badgeClass = 'b-candidate';
    else if (className.includes('FALSE')) badgeClass = 'b-fp';
    
    return `<span class="badge ${badgeClass}">${value}</span>`;
  };

  const getColumns = () => {
    if (!results?.results?.length) return [];
    
    const allColumns = new Set();
    results.results.forEach(row => Object.keys(row).forEach(key => allColumns.add(key)));
    
    const priority = ['Row_Number', 'KEP_ID', 'KOI_Name', 'Kepler_Name', 'Actual_Class', 'Predicted_Class', 'Confidence', 'Match'];
    const probColumns = Array.from(allColumns).filter(col => String(col).startsWith('Prob_')).sort();
    const otherColumns = Array.from(allColumns).filter(col => !priority.includes(col) && !probColumns.includes(col)).sort();
    
    return [...priority.filter(p => allColumns.has(p)), ...probColumns, ...otherColumns];
  };

  return (
    <div className="wrap">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <div style={{
            width: '42px',
            height: '42px',
            borderRadius: '12px',
            background: 'radial-gradient(circle at 30% 30%, #22d3ee, transparent 40%), radial-gradient(circle at 70% 65%, #60a5fa, transparent 35%), #0ea5e9',
            boxShadow: '0 6px 30px rgba(34,211,238,.35)'
          }}></div>
          <div>
            <h1 style={{ fontSize: '22px', margin: 0 }}>Exoplanet Predictor – KOI</h1>
            <div className="hint">Upload a KOI CSV → get predictions, probabilities, and a downloadable merged CSV.</div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <span className="chip">Backend: Django</span>
          <span className="chip">Model: XGB/RF</span>
          <span className="chip">Theme: Dark</span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr .8fr', gap: '16px' }}>
        {/* Left: Form */}
        <section className="card" style={{ padding: '18px' }}>
          <h2 style={{ margin: '0 0 8px' }}>Run Prediction</h2>
          <div className="hint" style={{ marginBottom: '14px' }}>
            Endpoint: <span className="mono">/api/predict</span>
          </div>

          <div className="row">
            <div style={{ flex: 1, minWidth: '160px' }}>
              <label>Satellite</label>
              <select 
                value={formData.satellite}
                onChange={(e) => setFormData(prev => ({ ...prev, satellite: e.target.value }))}
              >
                <option value="KOI">KOI</option>
              </select>
            </div>
            <div style={{ width: '140px' }}>
              <label>From Row (optional)</label>
              <input 
                type="number" 
                min="0" 
                placeholder="e.g. 0"
                value={formData.fromRow}
                onChange={(e) => setFormData(prev => ({ ...prev, fromRow: e.target.value }))}
              />
            </div>
            <div style={{ width: '140px' }}>
              <label>To Row (optional)</label>
              <input 
                type="number" 
                min="0" 
                placeholder="e.g. 99"
                value={formData.toRow}
                onChange={(e) => setFormData(prev => ({ ...prev, toRow: e.target.value }))}
              />
            </div>
            <div style={{ flex: 1, minWidth: '220px' }}>
              <label>API Base</label>
              <input 
                type="text" 
                value={formData.apiBase}
                onChange={(e) => setFormData(prev => ({ ...prev, apiBase: e.target.value }))}
              />
            </div>
          </div>

          <div 
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '12px',
              padding: '20px',
              border: '1px dashed #334155',
              borderRadius: '14px',
              background: '#0b1120',
              cursor: 'pointer'
            }}
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            {loading && <div className="spinner"></div>}
            <div>
              <b>Click to choose or drag & drop a CSV…</b>
              <div className="hint">Only <span className="mono">.csv</span> files</div>
            </div>
            <input 
              ref={fileInputRef}
              type="file" 
              accept=".csv" 
              style={{ display: 'none' }}
              onChange={handleFileSelect}
            />
          </div>
          <div style={{ fontSize: '12px', color: 'var(--muted)', textAlign: 'center', marginTop: '8px' }}>
            Tip: Use range to preview a slice quickly (e.g., 0 → 99).
          </div>

          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'center', marginTop: '12px' }}>
            <button 
              className="btn btn-primary"
              onClick={handlePredict}
              disabled={loading}
            >
              Predict
            </button>
            <button 
              className="btn btn-ghost"
              onClick={handleReset}
            >
              Reset
            </button>
            <span className="hint">
              {loading ? 'Uploading & predicting…' : ''}
            </span>
          </div>
        </section>

        {/* Right: Summary */}
        <section className="card" style={{ padding: '18px' }}>
          <h2 style={{ margin: '0 0 8px' }}>Summary</h2>
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <div style={{ flex: 1, minWidth: '160px', background: '#0b1120', border: '1px solid var(--border)', borderRadius: '12px', padding: '12px' }}>
              <span style={{ color: 'var(--muted)', fontSize: '12px' }}>Rows Predicted</span>
              <b style={{ display: 'block', fontSize: '18px' }}>
                {results ? (results.total_predicted || results.results?.length || 0) : '–'}
              </b>
            </div>
            <div style={{ flex: 1, minWidth: '160px', background: '#0b1120', border: '1px solid var(--border)', borderRadius: '12px', padding: '12px' }}>
              <span style={{ color: 'var(--muted)', fontSize: '12px' }}>Range</span>
              <b style={{ display: 'block', fontSize: '18px' }}>
                {results ? `${results.from_row} – ${results.to_row}` : '–'}
              </b>
            </div>
            <div style={{ flex: 1, minWidth: '160px', background: '#0b1120', border: '1px solid var(--border)', borderRadius: '12px', padding: '12px' }}>
              <span style={{ color: 'var(--muted)', fontSize: '12px' }}>CSV Output</span>
              <b style={{ display: 'block', fontSize: '18px', whiteSpace: 'nowrap' }}>
                {results?.csv_file ? (
                  <a href={results.csv_file} target="_blank" rel="noopener">download</a>
                ) : '–'}
              </b>
            </div>
          </div>
          <div style={{ marginTop: '12px' }}>
            <div style={{ color: 'var(--muted)', fontSize: '12px' }}>Class distribution</div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '8px' }}>
              {getClassCounts().map(([className, count]) => (
                <span 
                  key={className}
                  className={`chip ${
                    String(className).toUpperCase().includes('FALSE') ? 'err' :
                    String(className).toUpperCase().includes('CONFIRMED') ? 'ok' : 'warn'
                  }`}
                >
                  {className}: {count}
                </span>
              ))}
            </div>
          </div>
        </section>
      </div>

      {/* Results Table */}
      {results && (
        <section className="card" style={{ marginTop: '16px' }}>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center', justifyContent: 'space-between', padding: '12px', background: '#0b1120', borderBottom: '1px solid var(--border)' }}>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <input 
                type="text" 
                placeholder="Search KOI, Kepler, class…"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={{ width: '260px' }}
              />
              <button 
                className="btn btn-ghost"
                onClick={() => setSearchTerm('')}
              >
                Clear
              </button>
            </div>
            <div className="row" style={{ margin: 0 }}>
              <label style={{ fontSize: '12px' }}>Page size</label>
              <select 
                value={pageSize}
                onChange={(e) => setPageSize(Number(e.target.value))}
                style={{ width: '90px' }}
              >
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </div>
          </div>

          <div style={{ border: '1px solid var(--border)', borderRadius: '14px', overflow: 'hidden' }}>
            <table>
              <thead>
                <tr>
                  {getColumns().map(col => (
                    <th key={col} style={{ position: 'sticky', top: 0, background: '#0f172a', borderBottom: '1px solid var(--border)', fontWeight: 600, textAlign: 'left', padding: '10px 12px', whiteSpace: 'nowrap', cursor: 'pointer' }}>
                      {col.replaceAll('_', ' ')} ▾
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {getPaginatedResults().map((row, index) => (
                  <tr key={index} style={{ background: index % 2 === 0 ? 'transparent' : '#0b1120' }}>
                    {getColumns().map(col => {
                      const value = row[col];
                      let cellContent = value;
                      
                      if (col === 'Predicted_Class' || col === 'Actual_Class') {
                        cellContent = <span dangerouslySetInnerHTML={{ __html: getClassBadge(value) }} />;
                      } else if (String(col).startsWith('Prob_') || col === 'Confidence') {
                        cellContent = <span style={{ textAlign: 'right' }}>{formatProbability(value)}</span>;
                      } else if (value === null || value === undefined || value === '') {
                        cellContent = <span style={{ color: 'var(--muted)' }}>–</span>;
                      } else if (typeof value === 'number') {
                        cellContent = <span style={{ textAlign: 'right', fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace' }}>{value}</span>;
                      }
                      
                      return (
                        <td key={col} style={{ padding: '10px 12px', borderBottom: '1px solid rgba(255,255,255,.05)', verticalAlign: 'top' }}>
                          {cellContent}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center', justifyContent: 'flex-end', padding: '10px 12px', background: '#0b1120' }}>
              <button 
                className="btn btn-ghost"
                onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                disabled={currentPage <= 1}
              >
                ◀ Prev
              </button>
              <div style={{ color: 'var(--muted)', fontSize: '12px' }}>
                Page {currentPage} / {getTotalPages()}
              </div>
              <button 
                className="btn btn-ghost"
                onClick={() => setCurrentPage(prev => Math.min(getTotalPages(), prev + 1))}
                disabled={currentPage >= getTotalPages()}
              >
                Next ▶
              </button>
            </div>
          </div>
        </section>
      )}

      {/* Toast */}
      <div 
        className={`toast ${toast.show ? 'show' : ''}`}
        style={{
          position: 'fixed',
          right: '16px',
          bottom: '16px',
          background: '#0b1120',
          border: `1px solid ${toast.type === 'error' ? 'rgba(239,68,68,.35)' : 'rgba(16,185,129,.35)'}`,
          padding: '12px 14px',
          borderRadius: '12px',
          boxShadow: 'var(--shadow)',
          opacity: toast.show ? 1 : 0,
          transform: toast.show ? 'translateY(0)' : 'translateY(10px)',
          transition: '.25s'
        }}
      >
        {toast.message}
      </div>
    </div>
  );
};

export default Prediction;
