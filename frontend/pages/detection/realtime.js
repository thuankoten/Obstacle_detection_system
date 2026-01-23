import { useEffect, useMemo, useRef, useState } from 'react';

const API_BASE = 'http://127.0.0.1:8000';

function colorForRisk(riskLevel) {
  if (riskLevel === 'danger') return 'rgba(255, 60, 60, 0.95)';
  if (riskLevel === 'warning') return 'rgba(255, 215, 0, 0.95)';
  return 'rgba(0, 255, 255, 0.85)';
}

export default function DetectionRealtimePage() {
  const imgRef = useRef(null);
  const canvasRef = useRef(null);

  const [showNotes, setShowNotes] = useState(false);

  const [cameraIndex, setCameraIndex] = useState(0);
  const [sampleEveryN, setSampleEveryN] = useState(1);
  const [conf, setConf] = useState(0.5);
  const [roiWarn, setRoiWarn] = useState(0.65);
  const [roiDanger, setRoiDanger] = useState(0.8);

  const [connStatus, setConnStatus] = useState('disconnected');
  const [lastMsg, setLastMsg] = useState(null);
  const [error, setError] = useState(null);

  const streamUrl = useMemo(() => {
    const u = new URL(`${API_BASE}/api/realtime/stream`);
    u.searchParams.set('src', String(cameraIndex));
    u.searchParams.set('sampled_every_n_frames', String(sampleEveryN));
    u.searchParams.set('confidence_threshold', String(conf));
    u.searchParams.set('roi_warning_y_ratio', String(roiWarn));
    u.searchParams.set('roi_danger_y_ratio', String(roiDanger));
    u.searchParams.set('t', String(Date.now()));
    return u.toString();
  }, [cameraIndex, conf, roiDanger, roiWarn, sampleEveryN]);

  useEffect(() => {
    let ws;
    let cancelled = false;

    const connect = () => {
      setError(null);
      setConnStatus('connecting');

      const wsUrl = `${API_BASE.replace('http', 'ws')}/ws/realtime?src=${encodeURIComponent(String(cameraIndex))}&sampled_every_n_frames=${encodeURIComponent(String(sampleEveryN))}&confidence_threshold=${encodeURIComponent(String(conf))}&roi_warning_y_ratio=${encodeURIComponent(String(roiWarn))}&roi_danger_y_ratio=${encodeURIComponent(String(roiDanger))}`;
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        if (cancelled) return;
        setConnStatus('connected');
      };

      ws.onmessage = (ev) => {
        if (cancelled) return;
        try {
          const data = JSON.parse(ev.data);
          setLastMsg(data);
        } catch {
          // ignore
        }
      };

      ws.onerror = () => {
        if (cancelled) return;
        setError('WebSocket error');
      };

      ws.onclose = () => {
        if (cancelled) return;
        setConnStatus('disconnected');
      };
    };

    connect();
    return () => {
      cancelled = true;
      try {
        ws?.close();
      } catch {
        // ignore
      }
    };
  }, [cameraIndex, conf, roiDanger, roiWarn, sampleEveryN]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const msg = lastMsg;
      const rect = img.getBoundingClientRect();
      const w = Math.max(1, Math.floor(rect.width));
      const h = Math.max(1, Math.floor(rect.height));

      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!msg || !msg.frame_width || !msg.frame_height) return;

      const sx = canvas.width / msg.frame_width;
      const sy = canvas.height / msg.frame_height;

      const dets = msg.detections || [];
      for (const d of dets) {
        const x = (d.bbox?.x ?? d.x ?? 0) * sx;
        const y = (d.bbox?.y ?? d.y ?? 0) * sy;
        const bw = (d.bbox?.w ?? d.w ?? 0) * sx;
        const bh = (d.bbox?.h ?? d.h ?? 0) * sy;

        const risk = d.risk_level || 'info';
        const color = colorForRisk(risk);

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, bw, bh);

        const label = `${risk.toUpperCase()} | ${d.class_name || 'obj'} ${(Math.round((d.confidence || 0) * 100))}%`;
        ctx.font = '14px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial';
        const tw = ctx.measureText(label).width;
        const th = 18;

        ctx.fillStyle = color;
        ctx.fillRect(x, Math.max(0, y - th), tw + 8, th);
        ctx.fillStyle = '#0b1020';
        ctx.fillText(label, x + 4, Math.max(14, y - 5));
      }
    };

    draw();
  }, [lastMsg]);

  return (
    <div className="container">
      <h1>Detection Real Time</h1>

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
              <b>Camera index</b>: chỉ số camera mà OpenCV sẽ mở.
              Thường webcam mặc định là 0, nếu không lên hình hãy thử 1, 2, 3...
            </div>
            <div style={{ marginTop: 6 }}>
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
          </div>
        )}
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <div className="row">
          <label style={{ opacity: 0.85 }}>
            Camera index
            <input
              className="input"
              type="number"
              min="0"
              value={cameraIndex}
              onChange={(e) => setCameraIndex(Number(e.target.value || 0))}
              style={{ marginLeft: 8, width: 120 }}
            />
          </label>
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

        {error && <div className="pre" style={{ marginTop: 10 }}>{error}</div>}

        <div style={{ marginTop: 10, opacity: 0.85 }}>
          Status: <b>{connStatus}</b>
          {lastMsg?.detection_mode ? <> | Mode: <b>{lastMsg.detection_mode}</b></> : null}
          {typeof lastMsg?.fps === 'number' ? <> | Inference FPS: <b>{lastMsg.fps.toFixed(1)}</b></> : null}
        </div>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <div className="realtimeWrap">
          <img ref={imgRef} className="realtimeImg" src={streamUrl} alt="realtime" />
          <canvas ref={canvasRef} className="realtimeCanvas" />
        </div>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Detections</div>
        <div className="pre" style={{ maxHeight: 220, overflow: 'auto' }}>
          {JSON.stringify(lastMsg?.detections || [], null, 2)}
        </div>
      </div>
    </div>
  );
}
