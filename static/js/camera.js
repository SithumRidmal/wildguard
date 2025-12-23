const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultBox = document.getElementById("results");

let history = [];
let captureTimer = null;

const SMOOTHING_WINDOW = 5;
const CAPTURE_INTERVAL = 800;

// ---------- START CAMERA ----------
async function startCamera(useBackCamera = true) {
  // Stop previous stream if exists
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
    video.srcObject = null;
  }

  try {
    // Try to open requested camera
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: useBackCamera ? { exact: "environment" } : "user"
      },
      audio: false
    });
    video.srcObject = stream;

  } catch (error) {
    console.warn("Requested camera not available, using default camera", error);

    // Fallback to any available camera
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });
    video.srcObject = stream;
  }

  // Clear previous interval if exists
  if (captureTimer) clearInterval(captureTimer);
  captureTimer = setInterval(captureFrame, CAPTURE_INTERVAL);
}


// ---------- CAPTURE FRAME ----------
async function captureFrame() {
  if (!video.srcObject) return;

  ctx.drawImage(video, 0, 0, 224, 224);

  const blob = await new Promise(res =>
    canvas.toBlob(res, "image/jpeg", 0.8)
  );

  const fd = new FormData();
  fd.append("image", blob);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: fd
    });

    if (!res.ok) return;

    const data = await res.json();
    smoothPredictions(data.predictions);

  } catch (err) {
    console.error("Prediction error:", err);
  }
}

// ---------- PREDICTION SMOOTHING ----------
function smoothPredictions(preds) {
  history.push(preds);
  if (history.length > SMOOTHING_WINDOW) history.shift();

  const aggregated = {};

  history.flat().forEach(p => {
    aggregated[p.label] =
      (aggregated[p.label] || 0) + p.confidence;
  });

  const finalPreds = Object.entries(aggregated)
    .map(([label, confidence]) => ({
      label,
      confidence: (confidence / history.length).toFixed(2)
    }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3);

  renderResults(finalPreds);
}

// ---------- RENDER UI ----------
function renderResults(preds) {
  resultBox.innerHTML = preds.map(p => `
    <div class="border border-gray-300 p-3 rounded-xl">
      <div class="flex justify-between text-sm font-medium">
        <span>${p.label}</span>
        <span>${p.confidence}%</span>
      </div>
      <div class="w-full bg-gray-200 h-2 rounded-full mt-2">
        <div class="bg-black h-2 rounded-full"
             style="width:${p.confidence}%"></div>
      </div>
    </div>
  `).join("");
}
