import { useMemo, useState } from 'react';
import { useRouter } from 'next/router';

const API_BASE = 'http://127.0.0.1:8000';

export default function DetectionVideoPage() {
  const router = useRouter();
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [showNotes, setShowNotes] = useState(false);
  const [sampleEveryN, setSampleEveryN] = useState(1);
  const [conf, setConf] = useState(0.5);
  const [roiWarn, setRoiWarn] = useState(0.65);
  const [roiDanger, setRoiDanger] = useState(0.8);
  const [laneEnabled, setLaneEnabled] = useState(true);
  const [laneCenterX, setLaneCenterX] = useState(0.5);
  const [laneTopY, setLaneTopY] = useState(0.55);
  const [laneBottomY, setLaneBottomY] = useState(0.98);
  const [laneTopW, setLaneTopW] = useState(0.25);
  const [laneBottomW, setLaneBottomW] = useState(0.9);

  const videoUrl = useMemo(() => {
    if (!file) return null;
    return URL.createObjectURL(file);
  }, [file]);

  async function onSubmit(e) {
    e.preventDefault();
    setError(null);

    if (!file) {
      setError('Bạn chưa chọn video');
      return;
    }

    setBusy(true);
    try {
      const form = new FormData();
      form.append('file', file);

      form.append('sampled_every_n_frames', String(sampleEveryN));
      form.append('confidence_threshold', String(conf));
      form.append('roi_warning_y_ratio', String(roiWarn));
      form.append('roi_danger_y_ratio', String(roiDanger));
      form.append('lane_roi_enabled', String(laneEnabled));
      form.append('lane_roi_center_x_ratio', String(laneCenterX));
      form.append('lane_roi_top_y_ratio', String(laneTopY));
      form.append('lane_roi_bottom_y_ratio', String(laneBottomY));
      form.append('lane_roi_top_width_ratio', String(laneTopW));
      form.append('lane_roi_bottom_width_ratio', String(laneBottomW));

      const res = await fetch(`${API_BASE}/api/jobs`, {
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

      const data = await res.json();
      if (!data?.job_id) throw new Error('Missing job_id');
      router.push(`/jobs/${data.job_id}`);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="container">
      <h1>Detection Video</h1>

      <div className="card" style={{ marginTop: 12 }}>
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div style={{ fontWeight: 700 }}>Chú thích tham số</div>
          <button className="button" type="button" onClick={() => setShowNotes((v) => !v)}>
            {showNotes ? 'Thu gọn' : 'Xem thêm'}
          </button>
        </div>

        {showNotes && (
          <div style={{ opacity: 0.9 }}>
            <div style={{ marginTop: 10 }}>
              <b>Sample every N frames</b>: xử lý 1 frame mỗi N frame.
              N càng lớn thì chạy càng nhanh nhưng có thể bỏ sót vật cản xuất hiện nhanh.
            </div>
            <div style={{ marginTop: 6 }}>
              <b>Confidence</b>: ngưỡng độ tin cậy của YOLO.
              Tăng lên để giảm báo nhầm, giảm xuống để bắt được nhiều đối tượng hơn.
            </div>
            <div style={{ marginTop: 6 }}>
              <b>ROI warn y</b>: ngưỡng bắt đầu cảnh báo (warning).
              Nếu đáy bbox vượt qua tỉ lệ này (tính từ trên xuống), hệ thống đánh dấu warning.
            </div>
            <div style={{ marginTop: 6 }}>
              <b>ROI danger y</b>: ngưỡng nguy hiểm (danger).
              Nếu đáy bbox vượt qua tỉ lệ này, hệ thống đánh dấu danger.
            </div>
            <div style={{ marginTop: 6 }}>
              <b>Lane ROI (hình thang)</b>: chỉ giữ bbox/event nằm trong vùng hình thang phía trước xe.
              Vật thể ở làn bên cạnh sẽ bị loại.
            </div>
          </div>
        )}
      </div>

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
          <div className="row" style={{ marginTop: 12 }}>
            <label style={{ opacity: 0.85 }}>
              Sample every N frames
              <input
                className="input"
                type="number"
                min="1"
                value={sampleEveryN}
                onChange={(e) => setSampleEveryN(Number(e.target.value || 1))}
                style={{ marginLeft: 8, width: 120 }}
              />
            </label>
            <label style={{ opacity: 0.85 }}>
              Confidence
              <input
                className="input"
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={conf}
                onChange={(e) => setConf(Number(e.target.value || 0.5))}
                style={{ marginLeft: 8, width: 120 }}
              />
            </label>
            <label style={{ opacity: 0.85 }}>
              ROI warn y
              <input
                className="input"
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={roiWarn}
                onChange={(e) => setRoiWarn(Number(e.target.value || 0.65))}
                style={{ marginLeft: 8, width: 120 }}
              />
            </label>
            <label style={{ opacity: 0.85 }}>
              ROI danger y
              <input
                className="input"
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={roiDanger}
                onChange={(e) => setRoiDanger(Number(e.target.value || 0.8))}
                style={{ marginLeft: 8, width: 120 }}
              />
            </label>
          </div>

          <div className="row" style={{ marginTop: 12 }}>
            <label style={{ opacity: 0.85 }}>
              Lane ROI enabled
              <select
                className="input"
                value={laneEnabled ? 'true' : 'false'}
                onChange={(e) => setLaneEnabled(e.target.value === 'true')}
                style={{ marginLeft: 8, width: 140 }}
              >
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </label>
            <label style={{ opacity: 0.85 }}>
              Lane center x
              <input
                className="input"
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={laneCenterX}
                onChange={(e) => setLaneCenterX(Number(e.target.value || 0.5))}
                style={{ marginLeft: 8, width: 120 }}
              />
            </label>
            <label style={{ opacity: 0.85 }}>
              Lane top y
              <input
                className="input"
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={laneTopY}
                onChange={(e) => setLaneTopY(Number(e.target.value || 0.55))}
                style={{ marginLeft: 8, width: 120 }}
              />
            </label>
            <label style={{ opacity: 0.85 }}>
              Lane bottom y
              <input
                className="input"
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={laneBottomY}
                onChange={(e) => setLaneBottomY(Number(e.target.value || 0.98))}
                style={{ marginLeft: 8, width: 120 }}
              />
            </label>
            <label style={{ opacity: 0.85 }}>
              Lane top width
              <input
                className="input"
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={laneTopW}
                onChange={(e) => setLaneTopW(Number(e.target.value || 0.25))}
                style={{ marginLeft: 8, width: 120 }}
              />
            </label>
            <label style={{ opacity: 0.85 }}>
              Lane bottom width
              <input
                className="input"
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={laneBottomW}
                onChange={(e) => setLaneBottomW(Number(e.target.value || 0.9))}
                style={{ marginLeft: 8, width: 120 }}
              />
            </label>
          </div>
          <div style={{ opacity: 0.8, marginTop: 10 }}>
            Lưu ý: video càng dài thì xử lý càng lâu. Nếu lần đầu chạy YOLO, backend có thể tải model nên sẽ chậm hơn.
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

    </div>
  );
}
