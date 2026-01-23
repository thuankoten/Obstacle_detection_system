import { useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/router';

const API_BASE = 'http://127.0.0.1:8000';

function fmtMs(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  const mm = String(Math.floor(s / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return `${mm}:${ss}`;
}

export default function ResultPage() {
  const router = useRouter();
  const { resultId } = router.query;

  const videoRef = useRef(null);

  const [meta, setMeta] = useState(null);
  const [events, setEvents] = useState([]);
  const [error, setError] = useState(null);

  const videoUrl = useMemo(() => {
    if (!resultId) return null;
    return `${API_BASE}/api/results/${resultId}/video`;
  }, [resultId]);

  useEffect(() => {
    if (!resultId) return;
    let cancelled = false;

    const load = async () => {
      try {
        const [mRes, eRes] = await Promise.all([
          fetch(`${API_BASE}/api/results/${resultId}/meta`),
          fetch(`${API_BASE}/api/results/${resultId}/events`)
        ]);

        const mData = await mRes.json();
        const eData = await eRes.json();

        if (!mRes.ok) throw new Error(mData?.detail || 'Failed to load meta');
        if (!eRes.ok) throw new Error(eData?.detail || 'Failed to load events');

        if (cancelled) return;
        setMeta(mData);
        setEvents(eData?.events || []);
      } catch (e) {
        if (!cancelled) setError(e.message || String(e));
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [resultId]);

  const summary = useMemo(() => {
    const counts = { warning: 0, danger: 0 };
    for (const ev of events) {
      if (ev.risk_level === 'warning') counts.warning += 1;
      if (ev.risk_level === 'danger') counts.danger += 1;
    }
    return counts;
  }, [events]);

  const onSeek = (timestampMs) => {
    const el = videoRef.current;
    if (!el) return;
    el.currentTime = Math.max(0, timestampMs / 1000);
    el.play?.();
  };

  if (!resultId) return null;

  return (
    <div className="container">
      <h1>Results Dashboard</h1>

      {error && (
        <div className="card" style={{ marginTop: 12, borderColor: 'rgba(255, 120, 120, 0.55)' }}>
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Error</div>
          <div className="pre">{error}</div>
        </div>
      )}

      <div className="card" style={{ marginTop: 12 }}>
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div style={{ fontWeight: 700 }}>Annotated Video</div>
          <a className="button" href={videoUrl} download={`${resultId}.mp4`}>Download</a>
        </div>
        <div style={{ marginTop: 10 }}>
          <video ref={videoRef} src={videoUrl} controls style={{ width: '100%', borderRadius: 12 }} />
        </div>
      </div>

      <div className="row" style={{ marginTop: 12 }}>
        <div className="card" style={{ flex: 1, minWidth: 240 }}>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Summary</div>
          <div style={{ opacity: 0.85 }}>Warnings: <b>{summary.warning}</b></div>
          <div style={{ opacity: 0.85 }}>Dangers: <b>{summary.danger}</b></div>
          {meta && (
            <div style={{ marginTop: 10, opacity: 0.85 }}>
              <div>Filename: <b>{meta.filename}</b></div>
              <div>Mode: <b>{meta.detection_mode}</b></div>
              <div>FPS: <b>{meta.fps ?? 'n/a'}</b></div>
              <div>Frames: <b>{meta.frame_count ?? 'n/a'}</b></div>
              <div>Processing time: <b>{meta.processing_time_s}s</b></div>
            </div>
          )}
        </div>

        <div className="card" style={{ flex: 2, minWidth: 320 }}>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Timeline</div>
          <div className="timeline">
            {events.length === 0 && <div style={{ opacity: 0.75 }}>No warning/danger events.</div>}
            {events.map((ev, idx) => {
              const cls = ev.risk_level === 'danger' ? 'marker danger' : 'marker warning';
              return (
                <button
                  key={`${ev.frame_index}_${idx}`}
                  className={cls}
                  title={`${fmtMs(ev.timestamp_ms)} | ${ev.risk_level} | ${ev.class_name}`}
                  onClick={() => onSeek(ev.timestamp_ms)}
                  type="button"
                />
              );
            })}
          </div>
          <div style={{ marginTop: 8, opacity: 0.75 }}>
            Click a marker to seek the video.
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Events</div>
        {events.length === 0 ? (
          <div style={{ opacity: 0.8 }}>No events.</div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Risk</th>
                  <th>Class</th>
                  <th>Conf</th>
                  <th>Snapshot</th>
                </tr>
              </thead>
              <tbody>
                {events.map((ev, idx) => {
                  const snapUrl = ev.snapshot
                    ? `${API_BASE}/api/results/${resultId}/snapshots/${ev.snapshot}`
                    : null;
                  return (
                    <tr key={`${ev.frame_index}_${idx}`}>
                      <td>
                        <button className="linkButton" type="button" onClick={() => onSeek(ev.timestamp_ms)}>
                          {fmtMs(ev.timestamp_ms)}
                        </button>
                      </td>
                      <td>
                        <span className={ev.risk_level === 'danger' ? 'pill danger' : 'pill warning'}>
                          {ev.risk_level}
                        </span>
                      </td>
                      <td>{ev.class_name}</td>
                      <td>{Math.round((ev.confidence || 0) * 100)}%</td>
                      <td>
                        {snapUrl ? (
                          <a href={snapUrl} target="_blank" rel="noreferrer">view</a>
                        ) : (
                          <span style={{ opacity: 0.7 }}>-</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <button className="button" type="button" onClick={() => router.push('/')}>Analyze another video</button>
          <div style={{ opacity: 0.75 }}>Result ID: <code>{resultId}</code></div>
        </div>
      </div>
    </div>
  );
}
