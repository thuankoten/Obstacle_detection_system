import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/router';

const API_BASE = 'http://127.0.0.1:8000';

export default function JobPage() {
  const router = useRouter();
  const { jobId } = router.query;

  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);

  const progressPct = useMemo(() => {
    if (!job) return 0;
    return Math.round((job.progress || 0) * 100);
  }, [job]);

  useEffect(() => {
    if (!jobId) return;

    let cancelled = false;
    const tick = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data?.detail || 'Failed to fetch job');
        if (cancelled) return;
        setJob(data);

        if (data.status === 'done' && data.result_id) {
          router.replace(`/results/${data.result_id}`);
        }
      } catch (e) {
        if (!cancelled) setError(e.message || String(e));
      }
    };

    tick();
    const id = setInterval(tick, 800);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [jobId, router]);

  return (
    <div className="container">
      <h1>Processing Job</h1>

      <div className="card" style={{ marginTop: 12 }}>
        <div style={{ opacity: 0.85 }}>Job ID: <code>{jobId}</code></div>
        {error && (
          <div style={{ marginTop: 10 }} className="pre">{error}</div>
        )}

        {!job && !error && <div style={{ marginTop: 10 }}>Loading...</div>}

        {job && (
          <>
            <div style={{ marginTop: 10 }}>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>Status: {job.status}</div>
              <div style={{ opacity: 0.8 }}>{job.message || ''}</div>
            </div>

            <div style={{ marginTop: 12 }}>
              <div className="progressBar">
                <div className="progressFill" style={{ width: `${progressPct}%` }} />
              </div>
              <div style={{ marginTop: 8, opacity: 0.85 }}>
                {progressPct}%
                {job.total_frames ? ` (${job.processed_frames}/${job.total_frames} frames)` : ''}
              </div>
            </div>

            {job.status === 'error' && job.error && (
              <div style={{ marginTop: 12 }}>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>Error</div>
                <div className="pre">{job.error}</div>
              </div>
            )}
          </>
        )}
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <div style={{ opacity: 0.85 }}>
          Nếu lần đầu chạy YOLO, backend có thể tải model <code>yolov8n.pt</code> nên sẽ chậm hơn.
        </div>
      </div>
    </div>
  );
}
