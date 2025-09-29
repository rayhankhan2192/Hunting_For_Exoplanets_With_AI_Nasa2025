import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import apiService from '../services/api';

const MergeTrain = () => {
  const navigate = useNavigate();
  const [mergedFiles, setMergedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewData, setPreviewData] = useState(null);
  const [mergeForm, setMergeForm] = useState({
    file_a: null,
    file_b: null,
    dedupe: 'true',
    output_name: ''
  });
  const [trainForm, setTrainForm] = useState({
    satellite: 'KOI',
    model: 'rf'
  });
  const [messages, setMessages] = useState({ merge: '', train: '' });

  useEffect(() => {
    fetchExisting();
  }, []);

  const fetchExisting = async () => {
    try {
      const data = await apiService.getMergedFiles();
      setMergedFiles(data.files || []);
    } catch (error) {
      console.error('Failed to fetch merged files:', error);
    }
  };

  const handleFileSelect = (file, type) => {
    setMergeForm(prev => ({
      ...prev,
      [type]: file
    }));
  };

  const handleMerge = async (e) => {
    e.preventDefault();
    if (!mergeForm.file_a || !mergeForm.file_b) {
      setMessages(prev => ({ ...prev, merge: 'Please select both files' }));
      return;
    }

    setLoading(true);
    setMessages(prev => ({ ...prev, merge: 'Merging...' }));

    try {
      const result = await apiService.mergeFiles(
        mergeForm.file_a,
        mergeForm.file_b,
        {
          dedupe: mergeForm.dedupe === 'true',
          output_name: mergeForm.output_name || undefined
        }
      );

      // Select the newly merged file
      setSelectedFile({
        url: result.merged_url,
        filename: result.merged_filename,
        rows: result.merged_rows,
        token: result.merge_token
      });

      setMessages(prev => ({ 
        ...prev, 
        merge: `Merged: ${result.merged_filename}` 
      }));

      // Refresh the list
      fetchExisting();
    } catch (error) {
      setMessages(prev => ({ 
        ...prev, 
        merge: `Merge failed: ${error.message}` 
      }));
    } finally {
      setLoading(false);
    }
  };

  const handleUseFile = async (file) => {
    setSelectedFile({
      url: file.url,
      filename: file.filename,
      rows: file.rows,
      token: file.merge_token
    });

    // Load preview
    try {
      const response = await fetch(file.url, { cache: "no-store" });
      const blob = await response.blob();
      const text = await blob.text();
      setPreviewData(buildPreviewTable(text, 10));
    } catch (error) {
      console.error('Failed to load preview:', error);
    }
  };

  const buildPreviewTable = (csvText, maxRows = 10) => {
    const lines = csvText.trim().split(/\r?\n/);
    if (lines.length === 0) return null;
    
    const header = lines[0].split(",");
    const bodyLines = lines.slice(1, 1 + maxRows);
    
    return {
      header,
      rows: bodyLines.map(line => line.split(","))
    };
  };

  const handleTrain = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setMessages(prev => ({ 
        ...prev, 
        train: 'Please select or merge a CSV first' 
      }));
      return;
    }

    setLoading(true);
    setMessages(prev => ({ ...prev, train: 'Starting training...' }));

    try {
      // Re-upload the selected file
      const response = await fetch(selectedFile.url, { cache: "no-store" });
      const blob = await response.blob();
      const file = new File([blob], selectedFile.filename, { type: "text/csv" });

      const result = await apiService.startTraining(
        file,
        trainForm.satellite,
        trainForm.model
      );

      // Store the result and navigate to progress page
      sessionStorage.setItem("lastStartPayload", JSON.stringify(result));
      navigate(`/training-progress?job_id=${result.job_id}`);
    } catch (error) {
      setMessages(prev => ({ 
        ...prev, 
        train: `Could not start: ${error.message}` 
      }));
    } finally {
      setLoading(false);
    }
  };

  const formatSize = (bytes) => {
    if (!Number.isFinite(bytes)) return "";
    const units = ["B", "KB", "MB", "GB"];
    let i = 0;
    let n = +bytes;
    while (n >= 1024 && i < units.length - 1) {
      n /= 1024;
      i++;
    }
    return `${n.toFixed(1)} ${units[i]}`;
  };

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

      <div className="wrap">
        <div className="card">
        <h1>Merge & Train</h1>
        <p className="hint">
          This page first <strong>fetches merged CSVs</strong> from <code>/api/merge</code>. 
          Pick one to train, or merge two new files, then train.
        </p>
      </div>

      <div className="grid2">
        {/* LEFT: EXISTING MERGES */}
        <div className="card">
          <h2>Existing Merged CSVs</h2>
          <p className="hint">Fetched from <code>GET /api/merge</code></p>
          <div className="hint">
            {mergedFiles.length} merged file(s)
          </div>
          <div style={{
            overflow: 'auto',
            maxHeight: '320px',
            borderRadius: '12px',
            border: '1px solid rgba(255,255,255,.08)',
            padding: '8px'
          }}>
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Rows</th>
                  <th>Size</th>
                  <th>Modified</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {mergedFiles.map((file, index) => (
                  <tr key={index}>
                    <td>
                      <a href={file.url} download>
                        {file.filename}
                      </a>
                    </td>
                    <td>{file.rows || ''}</td>
                    <td>{formatSize(file.size_bytes)}</td>
                    <td>
                      {file.modified ? file.modified.replace('T', ' ').split('.')[0] : ''}
                    </td>
                    <td>
                      <button 
                        className="secondary btn-use"
                        onClick={() => handleUseFile(file)}
                      >
                        Use
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* RIGHT: MERGE NEW */}
        <div className="card">
          <h2>Merge Two New CSVs</h2>
          <form onSubmit={handleMerge}>
            <div className="row">
              <div>
                <label>CSV A</label>
                <input 
                  type="file" 
                  accept=".csv" 
                  required 
                  onChange={(e) => handleFileSelect(e.target.files[0], 'file_a')}
                />
              </div>
              <div>
                <label>CSV B</label>
                <input 
                  type="file" 
                  accept=".csv" 
                  required 
                  onChange={(e) => handleFileSelect(e.target.files[0], 'file_b')}
                />
              </div>
            </div>

            <div className="row">
              <div>
                <label>De-duplicate</label>
                <select 
                  value={mergeForm.dedupe}
                  onChange={(e) => setMergeForm(prev => ({ ...prev, dedupe: e.target.value }))}
                >
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              </div>
              <div>
                <label>Output name (optional)</label>
                <input 
                  type="text" 
                  placeholder="merged_custom.csv"
                  value={mergeForm.output_name}
                  onChange={(e) => setMergeForm(prev => ({ ...prev, output_name: e.target.value }))}
                />
              </div>
            </div>

            <div className="row">
              <button type="submit" disabled={loading}>
                Merge
              </button>
              <button 
                type="button" 
                className="secondary"
                onClick={() => {
                  setMergeForm({
                    file_a: null,
                    file_b: null,
                    dedupe: 'true',
                    output_name: ''
                  });
                  setMessages(prev => ({ ...prev, merge: '' }));
                }}
              >
                Reset
              </button>
            </div>
          </form>
          {messages.merge && (
            <div className={messages.merge.includes('failed') ? 'err' : 'ok'}>
              {messages.merge}
            </div>
          )}
        </div>
      </div>

      {/* PREVIEW + TRAIN */}
      {selectedFile && (
        <div className="card">
          <h2>Preview & Train</h2>
          <div className="row">
            <div className="pill">{selectedFile.filename}</div>
            <div className="pill">
              {selectedFile.rows ? `${selectedFile.rows} rows` : 'rows: n/a'}
            </div>
            <a 
              className="btnlink" 
              href={selectedFile.url} 
              download={selectedFile.filename}
            >
              Download CSV
            </a>
          </div>

          <div className="row">
            <div style={{ flex: 2, minWidth: '320px' }}>
              <div className="hint">Quick preview (first 10 rows)</div>
              <div style={{
                overflow: 'auto',
                maxHeight: '320px',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,.08)',
                padding: '8px'
              }}>
                {previewData && (
                  <table>
                    <thead>
                      <tr>
                        {previewData.header.map((h, i) => (
                          <th key={i}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {previewData.rows.map((row, i) => (
                        <tr key={i}>
                          {row.map((cell, j) => (
                            <td key={j}>{cell || ''}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>

            <div style={{ flex: 1, minWidth: '280px' }}>
              <form onSubmit={handleTrain}>
                <label>Satellite</label>
                <select 
                  value={trainForm.satellite}
                  onChange={(e) => setTrainForm(prev => ({ ...prev, satellite: e.target.value }))}
                  required
                >
                  <option value="KOI">KOI</option>
                  <option value="K2">K2</option>
                  <option value="TOI">TOI</option>
                </select>

                <label>Model</label>
                <select 
                  value={trainForm.model}
                  onChange={(e) => setTrainForm(prev => ({ ...prev, model: e.target.value }))}
                  required
                >
                  <option value="xgb">xgb</option>
                  <option value="rf">rf</option>
                  <option value="dt">dt</option>
                  <option value="grdb">grdb</option>
                  <option value="logreg">logreg</option>
                  <option value="svm">svm</option>
                </select>

                <button type="submit" disabled={loading}>
                  Start Training
                </button>
              </form>
              {messages.train && (
                <div className={messages.train.includes('failed') ? 'err' : 'hint'}>
                  {messages.train}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  );
};

export default MergeTrain;
