import React, { useState, useEffect } from 'react';
import apiService from '../services/api';

const MergeTrainAdvanced = () => {
  const [uploads, setUploads] = useState([]);
  const [formData, setFormData] = useState({
    fileA: '',
    fileB: '',
    satellite: 'KOI',
    model: 'xgb',
    outputName: '',
    dedupe: 'true'
  });
  const [loading, setLoading] = useState(false);
  const [jobStatus, setJobStatus] = useState({
    jobId: null,
    status: 'PENDING',
    mergedFile: null,
    rows: null,
    mergedUrl: null
  });
  const [statusJson, setStatusJson] = useState({});
  const [logs, setLogs] = useState('—');
  const [polling, setPolling] = useState(false);
  const [pollTimer, setPollTimer] = useState(null);

  useEffect(() => {
    loadUploads();
    
    // Cleanup polling on unmount
    return () => {
      if (pollTimer) {
        clearInterval(pollTimer);
      }
    };
  }, []);

  const loadUploads = async () => {
    try {
      const data = await apiService.getUploads();
      setUploads(data.files || []);
      
      // Auto-select first two files if available
      if (data.files && data.files.length >= 2) {
        setFormData(prev => ({
          ...prev,
          fileA: data.files[0].filename,
          fileB: data.files[1].filename
        }));
      }
    } catch (error) {
      console.error('Failed to load uploads:', error);
    }
  };

  const handleMergeAndTrain = async () => {
    if (!formData.fileA || !formData.fileB) {
      alert('Please select both files');
      return;
    }

    setLoading(true);
    try {
      const payload = {
        file_a: formData.fileA,
        file_b: formData.fileB,
        satellite: formData.satellite,
        model: formData.model,
        dedupe: formData.dedupe === 'true'
      };

      if (formData.outputName.trim()) {
        payload.output_name = formData.outputName.trim();
      }

      const result = await apiService.mergeAndTrain(payload);
      
      setJobStatus({
        jobId: result.job_id,
        status: result.status || 'PENDING',
        mergedFile: result.merged_file,
        rows: result.rows,
        mergedUrl: result.merged_url
      });
      
      setStatusJson(result);
      
      // Start polling if we have a job ID
      if (result.job_id) {
        startPolling();
      }
    } catch (error) {
      console.error('Merge and train failed:', error);
      setStatusJson({ error: error.message });
    } finally {
      setLoading(false);
    }
  };

  const startPolling = () => {
    if (pollTimer) {
      clearInterval(pollTimer);
    }
    
    setPolling(true);
    const timer = setInterval(pollStatus, 2000);
    setPollTimer(timer);
    pollStatus(); // Poll immediately
    refreshLogs();
  };

  const stopPolling = () => {
    if (pollTimer) {
      clearInterval(pollTimer);
      setPollTimer(null);
    }
    setPolling(false);
  };

  const pollStatus = async () => {
    if (!jobStatus.jobId) return;

    try {
      const data = await apiService.getTrainingStatus(jobStatus.jobId);
      setJobStatus(prev => ({
        ...prev,
        status: data.status || prev.status
      }));
      setStatusJson(data);

      // Stop polling if terminal state
      const status = String(data.status || '').toUpperCase();
      if (status === 'SUCCEEDED' || status === 'FAILED') {
        stopPolling();
      }
    } catch (error) {
      console.error('Status poll failed:', error);
    }
  };

  const refreshLogs = async () => {
    if (!jobStatus.jobId) {
      setLogs('—');
      return;
    }

    try {
      const data = await apiService.getTrainingLogs(jobStatus.jobId, 400);
      setLogs(data.logs || '—');
    } catch (error) {
      setLogs('Failed to load logs.');
    }
  };

  const getStatusClass = (status) => {
    if (!status) return 'pending';
    status = String(status).toUpperCase();
    if (status.includes('SUCCEEDED')) return 'succeeded';
    if (status.includes('FAILED')) return 'failed';
    return 'pending';
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    let size = bytes;
    while (size >= 1024 && i < units.length - 1) {
      size /= 1024;
      i++;
    }
    return `${size.toFixed(1)} ${units[i]}`;
  };

  return (
    <div className="wrap">
      <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '24px' }}>
        <div style={{
          width: '40px',
          height: '40px',
          borderRadius: '12px',
          background: 'radial-gradient(circle at 30% 30%, var(--accent), var(--accent2))',
          boxShadow: 'var(--shadow)'
        }}></div>
        <div>
          <h1 style={{ fontSize: '26px', margin: 0 }}>Merge & Train – Exoplanet (KOI/K2/TOI)</h1>
          <div style={{ color: 'var(--muted)', fontSize: '14px' }}>
            Pick two CSVs from <span style={{ background: '#0b1220', border: '1px solid #26314f', borderBottomWidth: '2px', padding: '2px 6px', borderRadius: '6px' }}>/api/uploads</span>, merge, and kick off a training job. Live status appears below.
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        {/* Left: File Selection and Training Options */}
        <section className="card" style={{ padding: '18px' }}>
          <h3 style={{ margin: '0 0 10px 0', fontSize: '16px' }}>1) Select CSV files</h3>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: '#cbd5e1', fontSize: '13px' }}>File A</label>
              <select 
                value={formData.fileA}
                onChange={(e) => setFormData(prev => ({ ...prev, fileA: e.target.value }))}
                style={{ width: '100%', padding: '12px 14px', borderRadius: '12px', border: '1px solid #293349', background: '#0b1220', color: 'var(--text)' }}
              >
                <option value="">Select file A</option>
                {uploads.map((file, index) => (
                  <option key={index} value={file.filename}>{file.filename}</option>
                ))}
              </select>
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: '#cbd5e1', fontSize: '13px' }}>File B</label>
              <select 
                value={formData.fileB}
                onChange={(e) => setFormData(prev => ({ ...prev, fileB: e.target.value }))}
                style={{ width: '100%', padding: '12px 14px', borderRadius: '12px', border: '1px solid #293349', background: '#0b1220', color: 'var(--text)' }}
              >
                <option value="">Select file B</option>
                {uploads.map((file, index) => (
                  <option key={index} value={file.filename}>{file.filename}</option>
                ))}
              </select>
            </div>
          </div>

          {/* File metadata chips */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '10px' }}>
            {formData.fileA && (
              <span style={{ fontSize: '12px', color: '#bfdbfe', background: 'rgba(37,99,235,.12)', padding: '6px 10px', borderRadius: '999px', border: '1px solid #1d3a74' }}>
                A: {formData.fileA}
              </span>
            )}
            {formData.fileB && (
              <span style={{ fontSize: '12px', color: '#bfdbfe', background: 'rgba(37,99,235,.12)', padding: '6px 10px', borderRadius: '999px', border: '1px solid #1d3a74' }}>
                B: {formData.fileB}
              </span>
            )}
          </div>

          {/* Satellite and Model */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginTop: '12px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: '#cbd5e1', fontSize: '13px' }}>Satellite</label>
              <select 
                value={formData.satellite}
                onChange={(e) => setFormData(prev => ({ ...prev, satellite: e.target.value }))}
                style={{ width: '100%', padding: '12px 14px', borderRadius: '12px', border: '1px solid #293349', background: '#0b1220', color: 'var(--text)' }}
              >
                <option value="KOI">KOI</option>
                <option value="K2">K2</option>
                <option value="TOI">TOI</option>
              </select>
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: '#cbd5e1', fontSize: '13px' }}>Model</label>
              <select 
                value={formData.model}
                onChange={(e) => setFormData(prev => ({ ...prev, model: e.target.value }))}
                style={{ width: '100%', padding: '12px 14px', borderRadius: '12px', border: '1px solid #293349', background: '#0b1220', color: 'var(--text)' }}
              >
                <option value="rf">Random Forest (rf)</option>
                <option value="xgb">XGBoost (xgb)</option>
                <option value="dt">Decision Tree (dt)</option>
                <option value="svm">SVM (svm)</option>
                <option value="logreg">LogReg (logreg)</option>
              </select>
            </div>
          </div>

          {/* Output name and options */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginTop: '12px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: '#cbd5e1', fontSize: '13px' }}>Output name (optional)</label>
              <input 
                type="text" 
                placeholder="merged_train_test.csv"
                value={formData.outputName}
                onChange={(e) => setFormData(prev => ({ ...prev, outputName: e.target.value }))}
                style={{ width: '100%', padding: '12px 14px', borderRadius: '12px', border: '1px solid #293349', background: '#0b1220', color: 'var(--text)' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: '#cbd5e1', fontSize: '13px' }}>Options</label>
              <select 
                value={formData.dedupe}
                onChange={(e) => setFormData(prev => ({ ...prev, dedupe: e.target.value }))}
                style={{ width: '100%', padding: '12px 14px', borderRadius: '12px', border: '1px solid #293349', background: '#0b1220', color: 'var(--text)' }}
              >
                <option value="true">Drop duplicate rows</option>
                <option value="false">Keep duplicates</option>
              </select>
            </div>
          </div>

          <div style={{ marginTop: '18px', display: 'flex', gap: '10px', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}>
            <button 
              onClick={handleMergeAndTrain}
              disabled={loading}
              style={{
                background: 'linear-gradient(180deg,#0ea5e9,#2563eb)',
                border: '0',
                color: '#000',
                fontWeight: '700',
                padding: '12px 16px',
                borderRadius: '12px',
                cursor: 'pointer'
              }}
            >
              {loading ? 'Starting…' : '🚀 Merge & Start Training'}
            </button>
            <div style={{ color: 'var(--muted)', fontSize: '12px' }}>
              Tip: refresh files with <span style={{ background: '#0b1220', border: '1px solid #26314f', borderBottomWidth: '2px', padding: '2px 6px', borderRadius: '6px' }}>R</span>
            </div>
          </div>
        </section>

        {/* Right: Uploaded Files List */}
        <section className="card" style={{ padding: '18px' }}>
          <h3 style={{ margin: '0 0 10px 0', fontSize: '16px' }}>2) Uploaded CSVs</h3>
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr',
            gap: '10px',
            maxHeight: '240px',
            overflow: 'auto',
            border: '1px solid #1f2a44',
            borderRadius: '12px',
            padding: '10px',
            background: '#0a1020'
          }}>
            {uploads.map((file, index) => (
              <div key={index} style={{
                display: 'flex',
                justifyContent: 'space-between',
                gap: '10px',
                padding: '8px 10px',
                borderRadius: '10px',
                background: '#0b1224',
                border: '1px solid #182243'
              }}>
                <div>
                  <b style={{ fontSize: '13px' }}>{file.filename}</b>
                  <div style={{ color: 'var(--muted)', fontSize: '12px' }}>
                    {formatFileSize(file.size_bytes)} bytes
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <a 
                    href={file.url} 
                    target="_blank" 
                    rel="noopener"
                    style={{ color: '#7dd3fc', textDecoration: 'none' }}
                  >
                    open
                  </a>
                </div>
              </div>
            ))}
          </div>
          
          <div style={{ marginTop: '18px', display: 'flex', gap: '10px', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}>
            <button 
              onClick={loadUploads}
              style={{
                background: 'linear-gradient(180deg,#0f172a,#121b30)',
                border: '1px solid #2a365a',
                color: 'var(--text)',
                padding: '12px 16px',
                borderRadius: '12px',
                cursor: 'pointer'
              }}
            >
              ↻ Refresh List
            </button>
            <span style={{ color: 'var(--muted)', fontSize: '12px' }}>
              {uploads.length} file(s)
            </span>
          </div>
        </section>
      </div>

      {/* Bottom: Status and Logs */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.3fr .7fr', gap: '16px', marginTop: '16px' }}>
        {/* Job Status */}
        <section className="card" style={{ padding: '18px' }}>
          <h3 style={{ margin: '0 0 10px 0', fontSize: '16px' }}>3) Job Status</h3>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: '8px 0 12px 0' }}>
            <div style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              background: getStatusClass(jobStatus.status) === 'succeeded' ? 'var(--ok)' : 
                         getStatusClass(jobStatus.status) === 'failed' ? 'var(--err)' : 'var(--warn)'
            }}></div>
            <div style={{ color: 'var(--muted)', fontSize: '12px' }}>
              {jobStatus.status || 'Waiting…'}
            </div>
          </div>
          
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '10px' }}>
            <span style={{ padding: '8px 10px', borderRadius: '999px', border: '1px solid #27406b', background: '#09132a', color: '#cde2ff', fontSize: '12px' }}>
              Job ID: <span style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace' }}>{jobStatus.jobId || '—'}</span>
            </span>
            <span style={{ padding: '8px 10px', borderRadius: '999px', border: '1px solid #27406b', background: '#09132a', color: '#cde2ff', fontSize: '12px' }}>
              Merged CSV: <a href={jobStatus.mergedUrl} target="_blank" rel="noopener" style={{ color: '#7dd3fc', textDecoration: 'none' }}>{jobStatus.mergedFile || '—'}</a>
            </span>
            <span style={{ padding: '8px 10px', borderRadius: '999px', border: '1px solid #27406b', background: '#09132a', color: '#cde2ff', fontSize: '12px' }}>
              Rows: <span>{jobStatus.rows || '—'}</span>
            </span>
          </div>
          
          <div style={{ marginTop: '10px' }}>
            <pre style={{
              whiteSpace: 'pre-wrap',
              background: '#0a0f1e',
              border: '1px solid #1c2744',
              padding: '12px',
              borderRadius: '12px',
              maxHeight: '260px',
              overflow: 'auto',
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace'
            }}>
              {JSON.stringify(statusJson, null, 2)}
            </pre>
          </div>
          
          <div style={{ marginTop: '18px', display: 'flex', gap: '10px', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}>
            <button 
              onClick={startPolling}
              disabled={polling}
              style={{
                background: 'linear-gradient(180deg,#0f172a,#121b30)',
                border: '1px solid #2a365a',
                color: 'var(--text)',
                padding: '12px 16px',
                borderRadius: '12px',
                cursor: 'pointer'
              }}
            >
              ⏳ Poll Status
            </button>
            <button 
              onClick={stopPolling}
              style={{
                background: 'linear-gradient(180deg,#0f172a,#121b30)',
                border: '1px solid #2a365a',
                color: 'var(--text)',
                padding: '12px 16px',
                borderRadius: '12px',
                cursor: 'pointer'
              }}
            >
              ✋ Stop Polling
            </button>
          </div>
        </section>

        {/* Logs */}
        <section className="card" style={{ padding: '18px' }}>
          <h3 style={{ margin: '0 0 10px 0', fontSize: '16px' }}>Logs (tail)</h3>
          
          <pre style={{
            whiteSpace: 'pre-wrap',
            background: '#0a0f1e',
            border: '1px solid #1c2744',
            padding: '12px',
            borderRadius: '12px',
            maxHeight: '260px',
            overflow: 'auto',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace'
          }}>
            {logs}
          </pre>
          
          <div style={{ marginTop: '18px', display: 'flex', gap: '10px', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}>
            <button 
              onClick={refreshLogs}
              style={{
                background: 'linear-gradient(180deg,#0f172a,#121b30)',
                border: '1px solid #2a365a',
                color: 'var(--text)',
                padding: '12px 16px',
                borderRadius: '12px',
                cursor: 'pointer'
              }}
            >
              📜 Refresh Logs
            </button>
            <span style={{ color: 'var(--muted)', fontSize: '12px' }}>
              From <span style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace' }}>/api/train/&lt;job_id&gt;/logs?tail=400</span>
            </span>
          </div>
        </section>
      </div>
    </div>
  );
};

export default MergeTrainAdvanced;
