import { useMemo, useState } from 'react';

export default function Home() {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [annotatedUrl, setAnnotatedUrl] = useState(null);
  const [error, setError] = useState(null);

  const videoUrl = useMemo(() => {
    if (!file) return null;
    return URL.createObjectURL(file);
  }, [file]);

  async function onSubmit(e) {
    e.preventDefault();
    setError(null);
    setAnnotatedUrl(null);

    if (!file) {
      setError('Bạn chưa chọn video');
      return;
    }

    setBusy(true);
    try {
      const form = new FormData();
      form.append('file', file);

      const res = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: form
      });

      if (!res.ok) {
        let detail = 'Upload failed';
        try {
          const data = await res.json();
          detail = data?.detail || detail;
        } catch {
          // ignore
        }
        throw new Error(detail);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setAnnotatedUrl(url);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="container">
      <h1>Car Obstacle Detection</h1>

      <div className="card" style={{ marginTop: 12 }}>
        <form onSubmit={onSubmit}>
          <div className="row">
            <input
              className="input"
              type="file"
              accept="video/mp4,video/x-m4v,video/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <button className="button" type="submit" disabled={busy}>
              {busy ? 'Đang xử lý...' : 'Upload & Analyze'}
            </button>
          </div>
          <div style={{ opacity: 0.8, marginTop: 10 }}>
            Lưu ý: video càng dài thì xử lý càng lâu. MVP hiện tại sẽ lấy mẫu mỗi vài frame và tìm chuyển động, sau đó xuất video.
          </div>
        </form>
      </div>

      {videoUrl && (
        <div className="card" style={{ marginTop: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 10 }}>Preview</div>
          <video src={videoUrl} controls style={{ width: '100%', borderRadius: 12 }} />
        </div>
      )}

      {error && (
        <div className="card" style={{ marginTop: 12, borderColor: 'rgba(255, 120, 120, 0.55)' }}>
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Error</div>
          <div className="pre">{error}</div>
        </div>
      )}

      {annotatedUrl && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <div style={{ fontWeight: 700 }}>Annotated Video</div>
            <a className="button" href={annotatedUrl} download={`annotated_${file?.name || 'video'}.mp4`}>
              Download
            </a>
          </div>
          <div style={{ marginTop: 10 }}>
            <video src={annotatedUrl} controls style={{ width: '100%', borderRadius: 12 }} />
          </div>
        </div>
      )}
    </div>
  );
}
