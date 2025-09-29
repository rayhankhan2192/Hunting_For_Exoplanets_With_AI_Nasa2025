import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import apiService from '../services/api';

const Result = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const jobId = searchParams.get('job_id');
  const pollTimerRef = useRef(null);
  const startTimeRef = useRef(Date.now());

  const [status, setStatus] = useState('PENDING');
  const [progress, setProgress] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [hint, setHint] = useState("We're syncing with the training service…");
  const [banner, setBanner] = useState(null);
  const [results, setResults] = useState(null);
  const [media, setMedia] = useState(null);

  useEffect(() => {
    if (!jobId) {
      setBanner({ type: 'err', title: 'Missing job_id', detail: 'Start a run from the training page.' });
      return;
    }

    // Check for merge payload from session storage
    const mergeLast = sessionStorage.getItem("lastMergePayload");
    if (mergeLast) {
      try {
        const j = JSON.parse(mergeLast);
        setBanner({
          type: 'warn',
          title: 'Merge completed; training started',
          detail: JSON.stringify({
            merged_file: j.merged_file,
            rows: j.rows,
            satellite: j.params?.satellite || undefined,
            model: j.params?.model || undefined
          }, null, 2)
        });
      } catch (e) {
        // Ignore parse errors
      }
      sessionStorage.removeItem("lastMergePayload");
    }

    // Start polling
    poll(true);
    pollTimerRef.current = setInterval(() => poll(false), 2000);

    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
      }
    };
  }, [jobId]);

  const poll = async (forceOnce = false) => {
    if (!jobId) return;

    try {
      const data = await apiService.getTrainingStatus(jobId);
      const currentStatus = data.status || "UNKNOWN";
      setStatus(currentStatus);

      // Update elapsed time
      const currentElapsed = data.result?.elapsed_sec || (Date.now() - startTimeRef.current) / 1000;
      setElapsed(currentElapsed);

      // Update progress
      if (typeof data.progress === "number") {
        setProgress(data.progress);
        setHint(`Progress: ${Math.round(data.progress)}%`);
      } else {
        if (currentStatus === "RUNNING" || currentStatus === "PENDING" || currentStatus === "STARTING") {
          setProgress(prev => Math.min(90, prev + 4 + Math.random() * 3));
          setHint("Training in progress…");
        }
      }

      // Handle results
      if (data.result) {
        setResults(data.result);
      }

      // Handle media
      if (data.result) {
        const mediaData = {
          cmUrl: data.result.cm_image_url ? 
            (data.result.cm_image_url.startsWith("http") ? 
              data.result.cm_image_url : 
              `http://203.190.12.138:8080${data.result.cm_image_url}`) : null,
          cmNormUrl: data.result.cm_norm_image_url ? 
            (data.result.cm_norm_image_url.startsWith("http") ? 
              data.result.cm_norm_image_url : 
              `http://203.190.12.138:8080${data.result.cm_norm_image_url}`) : null,
          modelUrl: data.result.model_url ? 
            (data.result.model_url.startsWith("http") ? 
              data.result.model_url : 
              `http://203.190.12.138:8080${data.result.model_url}`) : null
        };
        setMedia(mediaData);
      }

      // Terminal states
      if (currentStatus === "SUCCEEDED") {
        setProgress(100);
        setHint("Training finished.");
        setBanner({ type: 'ok', title: '✅ Model trained successfully', detail: '' });
        if (pollTimerRef.current) {
          clearInterval(pollTimerRef.current);
        }
      } else if (currentStatus === "FAILED") {
        setProgress(100);
        setHint("Training failed.");
        setBanner({ type: 'err', title: '❌ Training failed', detail: data.error || '' });
        if (pollTimerRef.current) {
          clearInterval(pollTimerRef.current);
        }
      }
    } catch (error) {
      setHint("Connection issue… retrying");
    }
  };

  const formatNumber = (n) => {
    if (typeof n === "number" && !Number.isNaN(n)) {
      return n.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
    }
    return "—";
  };

  const getStatusClass = (status) => {
    switch (status) {
      case 'RUNNING': return 'RUNNING';
      case 'PENDING': 
      case 'STARTING': return 'PENDING';
      case 'SUCCEEDED': return 'SUCCEEDED';
      case 'FAILED': return 'FAILED';
      default: return 'PENDING';
    }
  };

  const handleBack = () => {
    navigate('/merge-train');
  };

  const handleRefresh = () => {
    poll(true);
  };

  const handleCopyJobId = async () => {
    try {
      await navigator.clipboard.writeText(jobId);
      // You could add a toast notification here
    } catch (e) {
      // Ignore clipboard errors
    }
  };

  if (!jobId) {
    return (
      <div className="wrap">
        <div className="card">
          <h2>Missing job_id</h2>
          <p>Start a run from the training page.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="wrap">
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', marginBottom: '20px' }}>
          <h1>Training Progress</h1>
          <div className={`status-badge ${getStatusClass(status)}`}>
            <span className="dot"></span>
            <span>{status}</span>
          </div>
        </div>

        {/* RUN INFO */}
        <div className="section-title">Run info</div>
        <div className="grid" role="group" aria-label="Training summary">
          <div className="tile">
            <div className="label">Job ID</div>
            <div className="value mono">{jobId}</div>
            <div className="copy" onClick={handleCopyJobId}>Copy</div>
          </div>
          <div className="tile">
            <div className="label">Satellite</div>
            <div className="value">{results?.params?.satellite || "—"}</div>
          </div>
          <div className="tile">
            <div className="label">Model</div>
            <div className="value">{results?.params?.model || "—"}</div>
          </div>
          <div className="tile">
            <div className="label">Time (s)</div>
            <div className="value">{elapsed.toFixed(2)}</div>
          </div>
        </div>

        {/* PROGRESS */}
        <div className="progress" aria-label="progress">
          <div 
            className={`bar ${progress === 0 ? 'indeterminate' : ''}`}
            style={{ width: `${progress}%` }}
          ></div>
        </div>
        <div className="hint">{hint}</div>

        {/* RESULTS */}
        {results && (
          <div>
            <div className="section-title">Results</div>
            <div className="grid">
              <div className="tile">
                <div className="label">Accuracy</div>
                <div className="value">{formatNumber(results.accuracy)}</div>
              </div>
              <div className="tile">
                <div className="label">AUC</div>
                <div className="value">{formatNumber(results.auc_score)}</div>
              </div>
              <div className="tile">
                <div className="label">CV Mean</div>
                <div className="value">{formatNumber(results.cv_mean)}</div>
              </div>
              <div className="tile">
                <div className="label">CV Std</div>
                <div className="value">{formatNumber(results.cv_std)}</div>
              </div>
            </div>

            {/* MEDIA */}
            {media && (media.cmUrl || media.cmNormUrl || media.modelUrl) && (
              <div className="media">
                {media.cmUrl && (
                  <div>
                    <div className="label">Confusion Matrix (Raw)</div>
                    <a href={media.cmUrl} target="_blank" rel="noopener">
                      <img 
                        src={media.cmUrl} 
                        alt="Confusion Matrix (Raw)"
                        style={{ maxWidth: '360px', width: '100%', height: 'auto', borderRadius: '12px', border: '1px solid rgba(255,255,255,.06)' }}
                      />
                    </a>
                  </div>
                )}

                {media.cmNormUrl && (
                  <div>
                    <div className="label">Confusion Matrix (Normalized)</div>
                    <a href={media.cmNormUrl} target="_blank" rel="noopener">
                      <img 
                        src={media.cmNormUrl} 
                        alt="Confusion Matrix (Normalized)"
                        style={{ maxWidth: '360px', width: '100%', height: 'auto', borderRadius: '12px', border: '1px solid rgba(255,255,255,.06)' }}
                      />
                    </a>
                  </div>
                )}

                <div>
                  <div className="label">Artifacts</div>
                  <div className="kv">
                    {media.modelUrl && (
                      <a className="btn" href={media.modelUrl} download>
                        ⬇ Download Model
                      </a>
                    )}
                    {media.cmUrl && (
                      <a className="btn secondary" href={media.cmUrl} target="_blank" rel="noopener">
                        Open Confusion Matrix
                      </a>
                    )}
                  </div>
                  <div className="hint mono">
                    {results.model_path && `Model: ${results.model_path}`}
                    {results.cm_image_path && `\nCM (raw): ${results.cm_image_path}`}
                    {results.cm_norm_image_path && `\nCM (norm): ${results.cm_norm_image_path}`}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* BANNER */}
        {banner && (
          <div className={`banner ${banner.type}`}>
            <strong>{banner.title}</strong>
            {banner.detail && (
              <div style={{ marginTop: '6px', color: 'var(--muted)' }}>
                {banner.detail}
              </div>
            )}
          </div>
        )}

        {/* CTAs */}
        <div className="cta-row">
          <button className="btn secondary" onClick={handleBack}>
            ⬅ Back
          </button>
          <button className="btn secondary" onClick={handleRefresh}>
            ⟳ Refresh
          </button>
        </div>
      </div>
    </div>
  );
};

export default Result;
