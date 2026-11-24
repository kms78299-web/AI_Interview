const byId = id => document.getElementById(id);
const ui = Object.fromEntries([
  "question", "connection", "player", "timer", "record", "cancel-camera", "stop", "upload", "file", "error",
  "analyze", "empty", "empty-title", "empty-caption", "example", "result", "result-type", "elapsed",
  "warning", "scores", "score-note", "transcript", "download", "clear-example", "status", "output",
  "video-placeholder",
].map(id => [id, byId(id)]));

const metrics = [
  ["interview_score", "면접 평가"], ["answer_score", "답변 평가"],
  ["speaking_skills", "말하기 능력"], ["confidence_score", "자신감"],
  ["facial_expression", "표정"], ["overall_performance", "종합 예측"],
];
const videoLimit = 100 * 1024 * 1024;
let clip, previewUrl, stream, recorder, timer, result;
let pending = false;
let requestingCamera = false;
let examplePending = false;
let closed = false;

function setError(message = "") {
  ui.error.textContent = message;
  ui.error.hidden = !message;
}

function syncControls() {
  const cameraActive = Boolean(stream) || requestingCamera;
  ui.record.hidden = Boolean(recorder);
  ui.record.disabled = pending || requestingCamera;
  ui.record.textContent = requestingCamera ? "카메라 연결 중…" : stream ? "녹화 시작" : "카메라 켜기";
  ui.stop.hidden = !recorder;
  ui.stop.disabled = recorder?.state !== "recording";
  ui["cancel-camera"].hidden = !stream || Boolean(recorder);
  ui.upload.disabled = pending || cameraActive;
  ui.analyze.disabled = !clip || pending || cameraActive || examplePending;
  ui.example.disabled = pending || examplePending;
  ui.analyze.firstElementChild.textContent = pending ? "모델 분석 중…" : "모델 분석 실행";
}

function showPreview() {
  ui.player.pause();
  ui.player.srcObject = null;
  ui.player.removeAttribute("src");
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  ui.player.hidden = !stream && !clip;
  ui["video-placeholder"].hidden = !ui.player.hidden;
  ui.player.muted = Boolean(stream);
  ui.player.controls = !stream && Boolean(clip);
  if (stream) {
    ui.player.srcObject = stream;
    ui.player.play().catch(() => {});
  } else if (clip) {
    previewUrl = URL.createObjectURL(clip);
    ui.player.src = previewUrl;
  } else {
    ui.player.load();
  }
}

function setClip(next) {
  if (!next.size || next.size > videoLimit) {
    setError("0바이트보다 크고 100MB 이하인 영상을 올려 주세요.");
    showPreview();
    return;
  }
  clip = next;
  setError();
  showPreview();
  showResult(null);
  ui.status.textContent = "답변 영상이 준비됐습니다.";
  syncControls();
}

function releaseCamera() {
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  recorder = null;
  clearInterval(timer);
  ui.timer.hidden = true;
  syncControls();
}

ui.record.addEventListener("click", async () => {
  setError();
  if (!stream) {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setError("이 브라우저에서 카메라 녹화를 사용할 수 없어요. 영상 업로드를 이용해 주세요.");
      return;
    }
    requestingCamera = true;
    syncControls();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 720 } }, audio: true });
      if (closed) releaseCamera();
      showPreview();
    } catch {
      setError("카메라와 마이크를 열 수 없어요. 권한을 확인하거나 영상을 올려 주세요.");
    } finally {
      requestingCamera = false;
      syncControls();
    }
    return;
  }

  const supported = ["video/webm;codecs=vp8,opus", "video/webm", "video/mp4"].find(type => MediaRecorder.isTypeSupported(type));
  try {
    recorder = new MediaRecorder(stream, supported ? { mimeType: supported } : undefined);
    const current = recorder;
    const chunks = [];
    let failed = false;
    current.addEventListener("dataavailable", event => chunks.push(event.data));
    current.addEventListener("error", () => { failed = true; });
    current.addEventListener("stop", () => {
      releaseCamera();
      if (closed) return;
      if (failed) {
        showPreview();
        setError("영상 녹화에 실패했어요. 다시 녹화하거나 영상을 올려 주세요.");
        return;
      }
      const type = (current.mimeType || chunks.find(chunk => chunk.type)?.type || "").split(";")[0];
      if (!["video/mp4", "video/webm"].includes(type)) {
        showPreview();
        setError("이 브라우저의 녹화 형식을 지원하지 않아요. 영상 업로드를 이용해 주세요.");
        return;
      }
      setClip(new File(chunks, type === "video/mp4" ? "answer.mp4" : "answer.webm", { type }));
    });
    current.start();
    const started = performance.now();
    ui.timer.textContent = "● 00:00";
    ui.timer.hidden = false;
    timer = setInterval(() => {
      const seconds = Math.min(60, Math.floor((performance.now() - started) / 1000));
      ui.timer.textContent = seconds === 60 ? "● 01:00" : `● 00:${String(seconds).padStart(2, "0")}`;
      if (seconds === 60 && current.state === "recording") current.stop();
    }, 200);
    syncControls();
  } catch {
    releaseCamera();
    showPreview();
    setError("녹화를 시작할 수 없어요. 영상 업로드를 이용해 주세요.");
  }
});

ui.stop.addEventListener("click", () => {
  if (recorder?.state === "recording") {
    recorder.stop();
    syncControls();
  }
});

ui["cancel-camera"].addEventListener("click", () => {
  releaseCamera();
  showPreview();
});

ui.upload.addEventListener("click", () => ui.file.click());
ui.file.addEventListener("change", () => {
  const [file] = ui.file.files;
  ui.file.value = "";
  if (!file) return;
  if (!/\.(mp4|webm|mov)$/i.test(file.name)) {
    setError("WebM, MP4 또는 MOV 영상을 올려 주세요.");
    return;
  }
  setClip(file);
});

async function request(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => null);
  if (!response.ok || !body) {
    throw new Error(typeof body?.detail === "string" ? body.detail : "서버 응답을 받지 못했어요. 다시 시도해 주세요.");
  }
  return body;
}

function makeMetric(label, score) {
  const item = document.createElement("article");
  item.className = "metric";
  const title = document.createElement("h3");
  title.textContent = label;
  const value = document.createElement("strong");
  const percentile = score.percentile;
  value.textContent = percentile == null ? "—" : `${percentile} 백분위`;
  const detail = document.createElement("small");
  const interval = score.percentile_low != null && score.percentile_high != null;
  detail.textContent = [score.band, interval ? `80% 구간 ${score.percentile_low}–${score.percentile_high}` : null, `Raw ${score.raw}`].filter(Boolean).join(" · ");
  item.append(title, value, detail);
  if (percentile != null) {
    const meter = document.createElement("div");
    meter.className = "meter";
    meter.setAttribute("aria-hidden", "true");
    const bar = document.createElement("span");
    bar.style.width = `${percentile}%`;
    meter.append(bar);
    item.append(meter);
  }
  return item;
}

function showResult(data) {
  result = data;
  ui.empty.hidden = Boolean(data);
  ui.result.hidden = !data;
  if (!data) return;
  ui["result-type"].textContent = data.is_example ? "예시 · 가상 데이터" : "실제 모델 응답";
  ui["result-type"].classList.toggle("example", data.is_example);
  ui.elapsed.textContent = data.elapsed_ms == null ? "" : `모델 요청 ${(data.elapsed_ms / 1000).toFixed(2)}초`;
  ui.warning.textContent = data.warning || "";
  ui.warning.hidden = !data.warning;
  ui.scores.replaceChildren(...(data.scores ? metrics.map(([key, label]) => makeMetric(label, data.scores[key])) : []));
  ui["score-note"].hidden = !data.scores;
  ui.transcript.textContent = data.transcript.trim() || "전사된 답변이 없습니다.";
  ui["clear-example"].hidden = !data.is_example;
  ui.status.textContent = data.is_example ? "예시 결과를 표시했습니다." : "모델 분석이 완료됐습니다.";
}

ui.analyze.addEventListener("click", async () => {
  if (!clip || pending) return;
  pending = true;
  syncControls();
  setError();
  showResult(null);
  ui["empty-title"].textContent = "모델 분석 중…";
  ui["empty-caption"].textContent = "Colab에서 음성을 전사하고 평가 결과를 계산하고 있어요.";
  ui.example.hidden = true;
  ui.status.textContent = "모델 분석을 시작했습니다.";
  const form = new FormData();
  form.append("video", clip);
  try {
    showResult(await request("/api/demo/analyze", { method: "POST", body: form }));
    ui.output.focus({ preventScroll: true });
    if (matchMedia("(max-width: 720px)").matches) ui.output.scrollIntoView({ block: "start" });
  } catch (error) {
    setError(error.message);
  } finally {
    pending = false;
    ui["empty-title"].textContent = "분석 결과가 여기에 표시됩니다";
    ui["empty-caption"].textContent = "답변 영상을 추가한 뒤 분석을 실행해 주세요.";
    ui.example.hidden = false;
    syncControls();
  }
});

ui.example.addEventListener("click", async () => {
  examplePending = true;
  setError();
  syncControls();
  try {
    showResult(await request("/api/demo/example"));
    ui.output.focus({ preventScroll: true });
  } catch (error) {
    setError(error.message);
  } finally {
    examplePending = false;
    syncControls();
  }
});

ui["clear-example"].addEventListener("click", () => {
  showResult(null);
  ui.example.focus();
});

ui.download.addEventListener("click", () => {
  if (!result) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify({ question: ui.question.textContent, ...result }, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = result.is_example ? "recruitview-example.json" : "recruitview-introduction.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

window.addEventListener("pagehide", () => { closed = true; releaseCamera(); });
window.addEventListener("pageshow", event => {
  if (!event.persisted) return;
  closed = false;
  showPreview();
});

request("/api/demo").then(config => {
  ui.question.textContent = config.question;
  ui.connection.textContent = config.configured ? "Colab 모델 · 주소 설정됨" : "Colab 모델 · 주소 설정 필요";
}).catch(() => { ui.connection.textContent = "데모 서버에 연결할 수 없음"; });
