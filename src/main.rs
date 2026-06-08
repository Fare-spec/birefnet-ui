mod memory;
mod processing;

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, anyhow};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use memory::WipeOnDrop;
use processing::{
    Background, InputImage, ModelOption, ModelRegistry, decode_background_image, process_images,
    require_images, sanitize_filename, zip_images,
};
use serde::Serialize;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;
use zeroize::Zeroize;

const MAX_UPLOAD_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone)]
struct AppState {
    models: Arc<ModelRegistry>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let models = ModelRegistry::load_from_env()?;
    let state = AppState {
        models: Arc::new(models),
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/ui", get(index))
        .route("/ui/process", post(process_form))
        .route("/birefnet/remove-background", post(process_form))
        .route("/models", get(list_models))
        .route("/health", get(health))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = std::env::var("BIND_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3000".to_string())
        .parse()
        .context("BIND_ADDR invalide")?;

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("serveur disponible sur http://{addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn health() -> impl IntoResponse {
    "ok"
}

#[derive(Serialize)]
struct ModelsResponse {
    default_model: String,
    models: Vec<ModelOptionResponse>,
}

#[derive(Serialize)]
struct ModelOptionResponse {
    id: String,
    label: String,
    is_birefnet: bool,
}

impl From<ModelOption> for ModelOptionResponse {
    fn from(value: ModelOption) -> Self {
        Self {
            id: value.id,
            label: value.label,
            is_birefnet: value.is_birefnet,
        }
    }
}

async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        default_model: state.models.default_model_id().to_string(),
        models: state
            .models
            .options()
            .into_iter()
            .map(ModelOptionResponse::from)
            .collect(),
    })
}

async fn process_form(State(state): State<AppState>, multipart: Multipart) -> Response {
    match handle_multipart(&state, multipart).await {
        Ok(response) => response,
        Err(error) => (StatusCode::BAD_REQUEST, error.to_string()).into_response(),
    }
}

async fn handle_multipart(state: &AppState, mut multipart: Multipart) -> anyhow::Result<Response> {
    let mut inputs = Vec::new();
    let mut background_mode = "transparent".to_string();
    let mut background_image: Option<WipeOnDrop<Vec<u8>>> = None;
    let mut model_name = "edge-color".to_string();

    while let Some(field) = multipart.next_field().await? {
        let name = field.name().unwrap_or_default().to_string();
        let filename = field
            .file_name()
            .map(sanitize_filename)
            .unwrap_or_else(|| "image.png".to_string());
        let bytes = read_field_bytes(field).await?;

        match name.as_str() {
            "images" | "files" | "file" => {
                if !bytes.is_empty() {
                    inputs.push(InputImage {
                        filename,
                        bytes: bytes.to_vec(),
                    });
                }
            }
            "background_image" | "background" => {
                if !bytes.is_empty() {
                    background_image = Some(WipeOnDrop::new(bytes.to_vec()));
                }
            }
            "bg_mode" => {
                background_mode = String::from_utf8(bytes.to_vec())
                    .unwrap_or_else(|_| "transparent".to_string())
                    .trim()
                    .to_lowercase();
            }
            "model" => {
                model_name = String::from_utf8(bytes.to_vec())
                    .unwrap_or_else(|_| "edge-color".to_string())
                    .trim()
                    .to_lowercase();
            }
            _ => {}
        }
    }

    let background = match background_mode.as_str() {
        "transparent" => Background::Transparent,
        "white" => Background::White,
        "black" => Background::Black,
        "image" => {
            let bytes = background_image
                .as_ref()
                .ok_or_else(|| anyhow!("bg_mode=image requiert background_image"))?;
            Background::Image(decode_background_image(bytes)?)
        }
        other => return Err(anyhow!("bg_mode invalide: {other}")),
    };

    let inputs = require_images(inputs)?;
    let mut processed = process_images(state.models.as_ref(), &model_name, inputs, &background)?;

    let body = if processed.len() == 1 {
        let png = processed.pop().expect("image unique").png;
        single_png_response(png)
    } else {
        let zip = zip_images(&processed)?;
        zip_response(zip)
    };

    for image in &mut processed {
        image.png.zeroize();
    }

    Ok(body)
}

async fn read_field_bytes(field: axum::extract::multipart::Field<'_>) -> anyhow::Result<Bytes> {
    field
        .bytes()
        .await
        .context("lecture multipart impossible")
}

fn single_png_response(png: Vec<u8>) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("attachment; filename=\"birefnet-result.png\""),
    );
    (headers, png).into_response()
}

fn zip_response(zip: Vec<u8>) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/zip"));
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("attachment; filename=\"birefnet-results.zip\""),
    );
    (headers, zip).into_response()
}

const INDEX_HTML: &str = r##"<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BiRefNet Rust</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #090d12;
      --surface: #111821;
      --surface-2: #151f2a;
      --surface-3: #0d131a;
      --ink: #f3f7fb;
      --muted: #93a3b5;
      --line: #263341;
      --line-strong: #344557;
      --accent: #2dd4bf;
      --accent-2: #60a5fa;
      --accent-ink: #ffffff;
      --danger: #fb7185;
      --ok: #34d399;
      --shadow: 0 20px 70px rgba(0, 0, 0, .42);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(45, 212, 191, .12), transparent 34rem),
        linear-gradient(180deg, #0b1118 0%, var(--bg) 42%);
      background-attachment: fixed;
    }

    main {
      width: min(1280px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 44px;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 22px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .mark {
      display: grid;
      width: 38px;
      height: 38px;
      place-items: center;
      border: 1px solid rgba(45, 212, 191, .35);
      border-radius: 8px;
      background: linear-gradient(135deg, rgba(45, 212, 191, .22), rgba(96, 165, 250, .14));
      color: var(--accent);
      font-weight: 900;
    }

    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.1;
      letter-spacing: 0;
    }

    .subtitle {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 14px;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(17, 24, 33, .78);
      color: var(--muted);
      font-size: 13px;
    }

    .status-stack {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }

    .status-pill.secure {
      color: #b7f7df;
      border-color: rgba(52, 211, 153, .32);
    }

    .status-pill.local {
      color: #bfdbfe;
      border-color: rgba(96, 165, 250, .34);
    }

    .status-pill.insecure {
      color: #fecdd3;
      border-color: rgba(251, 113, 133, .42);
    }

    .status-pill.secure .status-dot,
    .status-pill.local .status-dot {
      background: var(--ok);
      box-shadow: 0 0 18px rgba(52, 211, 153, .75);
    }

    .status-pill.insecure .status-dot {
      background: var(--danger);
      box-shadow: 0 0 18px rgba(251, 113, 133, .65);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--ok);
      box-shadow: 0 0 18px rgba(52, 211, 153, .75);
    }

    .workspace {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }

    form {
      display: grid;
      gap: 16px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(17, 24, 33, .9);
      box-shadow: var(--shadow);
      position: sticky;
      top: 20px;
    }

    label {
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }

    input,
    select,
    button {
      width: 100%;
      min-height: 44px;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: var(--surface-3);
      color: var(--ink);
      font: inherit;
      padding: 10px 12px;
    }

    input[type="file"] {
      min-height: auto;
      padding: 12px;
      border-style: dashed;
      background: rgba(13, 19, 26, .86);
    }

    input:focus,
    select:focus,
    button:focus-visible,
    .download:focus-visible {
      outline: 2px solid rgba(45, 212, 191, .55);
      outline-offset: 2px;
    }

    button {
      border: 0;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: var(--accent-ink);
      font-weight: 750;
      cursor: pointer;
    }

    button:disabled {
      cursor: wait;
      opacity: .68;
    }

    .switch-row {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--ink);
      font-size: 14px;
      font-weight: 700;
    }

    .switch-row input {
      width: 18px;
      min-height: 18px;
      accent-color: var(--accent);
    }

    .grid {
      display: grid;
      gap: 14px;
    }

    .actions {
      display: grid;
      gap: 10px;
    }

    .secondary-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .download {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 38px;
      place-items: center;
      padding: 8px 12px;
      border-radius: 6px;
      border: 1px solid var(--line-strong);
      background: var(--surface-2);
      color: var(--ink);
      text-decoration: none;
      font-size: 14px;
      font-weight: 750;
    }

    .ghost-button,
    .danger-button {
      min-height: 38px;
      border: 1px solid var(--line-strong);
      background: var(--surface-2);
      color: var(--ink);
      font-size: 14px;
    }

    .danger-button {
      border-color: rgba(251, 113, 133, .35);
      color: #fecdd3;
      background: rgba(251, 113, 133, .1);
    }

    .progress-wrap {
      display: grid;
      gap: 8px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface-3);
    }

    .progress-meta {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }

    .progress-track {
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: #070b10;
    }

    .progress-bar {
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      transition: width .2s ease;
    }

    .progress-bar.busy {
      width: 38%;
      animation: progress-busy 1.1s ease-in-out infinite;
    }

    @keyframes progress-busy {
      0% { transform: translateX(-110%); }
      100% { transform: translateX(280%); }
    }

    .download[hidden],
    .progress-wrap[hidden],
    .secondary-actions[hidden],
    .empty[hidden],
    .error[hidden] {
      display: none;
    }

    .results {
      display: grid;
      gap: 16px;
    }

    .pair {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(17, 24, 33, .88);
      box-shadow: 0 14px 40px rgba(0, 0, 0, .24);
    }

    .result-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      grid-column: 1 / -1;
      min-width: 0;
    }

    .file-name {
      overflow: hidden;
      color: var(--ink);
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 750;
    }

    .result-head-actions {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }

    .badge {
      flex: 0 0 auto;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      color: var(--muted);
      background: var(--surface-3);
      font-size: 12px;
      font-weight: 750;
    }

    .badge.done {
      color: #b7f7df;
      border-color: rgba(52, 211, 153, .35);
      background: rgba(52, 211, 153, .1);
    }

    .badge.error {
      color: #fecdd3;
      border-color: rgba(251, 113, 133, .42);
      background: rgba(251, 113, 133, .1);
    }

    .icon-button {
      flex: 0 0 auto;
      width: 32px;
      min-height: 32px;
      padding: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface-3);
      color: var(--muted);
      font-size: 18px;
      line-height: 1;
    }

    .frame {
      display: grid;
      gap: 8px;
      min-width: 0;
    }

    .frame strong {
      color: var(--muted);
      font-size: 14px;
      font-weight: 750;
    }

    .preview {
      display: grid;
      place-items: center;
      min-height: 180px;
      max-height: 520px;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background-color: #0d141c;
      background-image:
        linear-gradient(45deg, #25303b 25%, transparent 25%),
        linear-gradient(-45deg, #25303b 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #25303b 75%),
        linear-gradient(-45deg, transparent 75%, #25303b 75%);
      background-size: 24px 24px;
      background-position: 0 0, 0 12px, 12px -12px, -12px 0;
    }

    .preview.no-checker {
      background: #0c1219;
    }

    img {
      display: block;
      width: auto;
      height: auto;
      max-width: 100%;
      max-height: 520px;
      object-fit: contain;
    }

    .empty,
    .error {
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(17, 24, 33, .74);
      color: var(--muted);
    }

    .error {
      background: #2a1118;
      color: var(--danger);
      border: 1px solid #7f1d1d;
      margin-bottom: 16px;
    }

    .empty {
      min-height: 260px;
      display: grid;
      place-items: center;
      text-align: center;
    }

    .empty strong {
      display: block;
      margin-bottom: 6px;
      color: var(--ink);
      font-size: 18px;
    }

    @media (max-width: 900px) {
      .workspace { grid-template-columns: 1fr; }
      form { position: static; }
    }

    @media (max-width: 720px) {
      .topbar {
        align-items: flex-start;
        flex-direction: column;
      }

      .pair { grid-template-columns: 1fr; }
      .secondary-actions { grid-template-columns: 1fr; }
      main { width: min(100vw - 20px, 960px); padding: 20px 0; }
    }
  </style>
</head>
<body>
  <main>
    <header class="topbar">
      <div class="brand">
        <div class="mark">B</div>
        <div>
          <h1>BiRefNet Studio</h1>
          <p class="subtitle">Batch background removal with local TorchScript models.</p>
        </div>
      </div>
      <div class="status-stack">
        <div class="status-pill">
          <span class="status-dot"></span>
          <span id="model-count">Chargement des modeles</span>
        </div>
        <div id="security-status" class="status-pill">
          <span class="status-dot"></span>
          <span id="security-label">Verification connexion</span>
        </div>
      </div>
    </header>

    <div class="workspace">
      <form id="process-form">
        <label>
          Modele
          <select id="model-select" name="model">
            <option value="">Chargement...</option>
          </select>
        </label>

        <label>
          Images source
          <input id="images" type="file" name="images" accept="image/*" multiple required>
        </label>

        <div class="grid">
          <label>
            Arriere-plan
          <select name="bg_mode">
            <option value="transparent">Transparent</option>
            <option value="white">Blanc</option>
            <option value="black">Noir</option>
            <option value="image">Image de fond</option>
          </select>
          </label>

          <label>
            Image de fond
            <input type="file" name="background_image" accept="image/*">
          </label>
        </div>

        <div id="progress-wrap" class="progress-wrap" hidden>
          <div class="progress-meta">
            <span id="progress-label">Pret</span>
            <span id="progress-count">0 / 0</span>
          </div>
          <div class="progress-track">
            <div id="progress-bar" class="progress-bar"></div>
          </div>
        </div>

        <div class="actions">
          <button id="submit" type="submit">Traiter les images</button>
          <div id="secondary-actions" class="secondary-actions" hidden>
            <a id="zip-download" class="download" hidden download="birefnet-results.zip">Telecharger toutes les images</a>
            <button id="clear-all" class="danger-button" type="button">Tout supprimer</button>
          </div>
        </div>
      </form>

      <section>
        <div id="error" class="error" hidden></div>
        <div id="empty" class="empty">
          <div>
            <strong>Aucune image selectionnee</strong>
            Choisissez une ou plusieurs images pour afficher une comparaison avant/apres.
          </div>
        </div>
        <div id="results" class="results" aria-live="polite"></div>
      </section>
    </div>
  </main>

  <script>
    const form = document.querySelector("#process-form");
    const input = document.querySelector("#images");
    const button = document.querySelector("#submit");
    const modelSelect = document.querySelector("#model-select");
    const modelCount = document.querySelector("#model-count");
    const securityStatus = document.querySelector("#security-status");
    const securityLabel = document.querySelector("#security-label");
    const zipDownload = document.querySelector("#zip-download");
    const clearAll = document.querySelector("#clear-all");
    const secondaryActions = document.querySelector("#secondary-actions");
    const progressWrap = document.querySelector("#progress-wrap");
    const progressLabel = document.querySelector("#progress-label");
    const progressCount = document.querySelector("#progress-count");
    const progressBar = document.querySelector("#progress-bar");
    const results = document.querySelector("#results");
    const empty = document.querySelector("#empty");
    const error = document.querySelector("#error");
    let activeUrls = [];
    let selectedFiles = [];
    let processedFiles = [];
    let isProcessing = false;

    async function loadModels() {
      try {
        const response = await fetch("/models");
        if (!response.ok) throw new Error(await response.text());
        const payload = await response.json();
        modelSelect.innerHTML = "";
        for (const model of payload.models) {
          const option = document.createElement("option");
          option.value = model.id;
          option.textContent = model.label;
          if (model.id === payload.default_model) option.selected = true;
          modelSelect.appendChild(option);
        }
        modelCount.textContent = `${payload.models.length} modele${payload.models.length > 1 ? "s" : ""} disponible${payload.models.length > 1 ? "s" : ""}`;
      } catch (err) {
        modelSelect.innerHTML = "";
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "Aucun modele charge";
        modelSelect.appendChild(option);
        modelCount.textContent = "Aucun modele disponible";
      }
    }

    function clearUrls() {
      for (const url of activeUrls) URL.revokeObjectURL(url);
      activeUrls = [];
    }

    function objectUrl(blob) {
      const url = URL.createObjectURL(blob);
      activeUrls.push(url);
      return url;
    }

    function resetProcessedOutput() {
      clearUrls();
      results.innerHTML = "";
      processedFiles = [];
      zipDownload.hidden = true;
      zipDownload.removeAttribute("href");
      secondaryActions.hidden = selectedFiles.length === 0;
      error.hidden = true;
      error.textContent = "";
      empty.hidden = selectedFiles.length !== 0;
      updateProgress(0, selectedFiles.length, "Pret");
    }

    function renderList() {
      clearUrls();
      results.innerHTML = "";
      secondaryActions.hidden = selectedFiles.length === 0;
      empty.hidden = selectedFiles.length !== 0;

      for (const [index, file] of selectedFiles.entries()) {
        const processed = processedFiles[index];
        const pair = document.createElement("article");
        pair.className = "pair";
        pair.dataset.index = String(index);
        pair.innerHTML = `
          <div class="result-head">
            <div class="file-name"></div>
            <div class="result-head-actions">
              <span class="badge${processed ? " done" : ""}">${processed ? "Termine" : "En attente"}</span>
              <button class="icon-button remove-image" type="button" title="Supprimer cette image" aria-label="Supprimer cette image" ${isProcessing ? "disabled" : ""}>×</button>
            </div>
          </div>
          <div class="frame">
            <strong>Avant</strong>
            <div class="preview"><img alt=""></div>
          </div>
          <div class="frame">
            <strong>Apres</strong>
            <div class="preview meta">En attente</div>
            <a class="download" hidden download>Telecharger</a>
          </div>
        `;
        pair.querySelector(".file-name").textContent = file.name;
        pair.querySelector("img").src = objectUrl(file);
        pair.querySelector("img").alt = file.name;
        pair.querySelector(".remove-image").addEventListener("click", () => removeImage(index));
        if (processed) {
          renderProcessedPair(pair, file, processed);
        }
        results.appendChild(pair);
      }
    }

    function renderProcessedPair(pair, file, processed) {
      const bgMode = form.querySelector("[name='bg_mode']").value;
      const url = objectUrl(processed.blob);
      const badge = pair.querySelector(".badge");
      badge.textContent = "Termine";
      badge.classList.add("done");

      const frame = pair.querySelector(".frame:last-child .preview");
      frame.innerHTML = "";
      frame.classList.toggle("no-checker", bgMode !== "transparent");
      const img = document.createElement("img");
      img.src = url;
      img.alt = `Resultat ${file.name}`;
      frame.appendChild(img);

      const link = pair.querySelector(".download");
      link.href = url;
      link.download = processed.filename;
      link.hidden = false;
    }

    function updateProgress(done, total, label) {
      progressWrap.hidden = total === 0;
      progressLabel.textContent = label;
      progressCount.textContent = `${done} / ${total}`;
      progressBar.classList.remove("busy");
      progressBar.style.width = total === 0 ? "0%" : `${Math.round((done / total) * 100)}%`;
    }

    function updateProgressPercent(percent, label) {
      progressWrap.hidden = false;
      progressLabel.textContent = label;
      progressCount.textContent = `${Math.round(percent)}%`;
      progressBar.classList.remove("busy");
      progressBar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    }

    function setProgressBusy(label) {
      progressWrap.hidden = false;
      progressLabel.textContent = label;
      progressCount.textContent = "";
      progressBar.style.width = "38%";
      progressBar.classList.add("busy");
    }

    async function removeImage(index) {
      if (isProcessing) return;
      selectedFiles.splice(index, 1);
      processedFiles.splice(index, 1);
      input.value = "";
      renderList();
      await refreshZipDownload();
      updateProgress(processedFiles.filter(Boolean).length, selectedFiles.length, "Pret");
    }

    function clearAllImages() {
      selectedFiles = [];
      processedFiles = [];
      input.value = "";
      resetProcessedOutput();
    }

    input.addEventListener("change", () => {
      selectedFiles = [...input.files];
      resetProcessedOutput();
      renderList();
    });

    clearAll.addEventListener("click", clearAllImages);

    function isLocalHost() {
      return ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
    }

    function isTransportProtected() {
      return window.location.protocol === "https:" || isLocalHost();
    }

    function updateSecurityStatus() {
      securityStatus.classList.remove("secure", "local", "insecure");
      if (window.location.protocol === "https:") {
        securityStatus.classList.add("secure");
        securityLabel.textContent = "HTTPS: upload chiffre";
      } else if (isLocalHost()) {
        securityStatus.classList.add("local");
        securityLabel.textContent = "Localhost: aucun envoi distant";
      } else {
        securityStatus.classList.add("insecure");
        securityLabel.textContent = "HTTP: upload non chiffre";
      }
    }

    updateSecurityStatus();

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!selectedFiles.length) return;
      if (!isTransportProtected()) {
        error.textContent = "Connexion non chiffree: utilisez HTTPS avant d'envoyer des images, ou lancez l'interface en localhost.";
        error.hidden = false;
        return;
      }

      isProcessing = true;
      resetProcessedOutput();
      renderList();
      button.disabled = true;
      button.textContent = "Traitement en cours...";
      let completed = 0;
      const uploadProgress = new Array(selectedFiles.length).fill(0);
      const uploadTotal = selectedFiles.reduce((sum, file) => sum + file.size, 0);
      updateProgressPercent(0, "Upload");

      try {
        const model = form.querySelector("[name='model']").value;
        const bgMode = form.querySelector("[name='bg_mode']").value;
        const background = form.querySelector("[name='background_image']").files[0];
        const pairs = [...results.querySelectorAll(".pair")];

        await runWithConcurrency(selectedFiles.map((file, index) => async () => {
          const pair = pairs[index];
          pair.querySelector(".badge").textContent = "Upload";

          const response = await uploadImage(file, {
            model,
            bgMode,
            background,
            onUploadProgress: (loaded) => {
              uploadProgress[index] = Math.min(file.size, loaded);
              const totalLoaded = uploadProgress.reduce((sum, value) => sum + value, 0);
              updateProgressPercent(uploadTotal === 0 ? 100 : (totalLoaded / uploadTotal) * 100, "Upload");
              if (loaded >= file.size) {
                pair.querySelector(".badge").textContent = "Modele";
              }
              if (uploadTotal > 0 && totalLoaded >= uploadTotal) {
                setProgressBusy("Traitement modele");
              }
            },
          });

          const filename = `${file.name.replace(/\.[^.]+$/, "") || "image"}-birefnet.png`;
          processedFiles[index] = {
            filename,
            blob: response.blob,
          };
          renderProcessedPair(pair, file, processedFiles[index]);
          completed += 1;
          updateProgress(completed, selectedFiles.length, "Traitement modele");
        }), parallelRequestCount());

        if (processedFiles.filter(Boolean).length) {
          updateProgress(completed, selectedFiles.length, "Creation du ZIP");
          await refreshZipDownload();
        }
        updateProgress(completed, selectedFiles.length, "Termine");
      } catch (err) {
        for (const badge of results.querySelectorAll(".badge:not(.done)")) {
          badge.textContent = "Erreur";
          badge.classList.add("error");
        }
        error.textContent = err.message || String(err);
        error.hidden = false;
      } finally {
        isProcessing = false;
        for (const control of results.querySelectorAll(".remove-image")) {
          control.disabled = false;
        }
        button.disabled = false;
        button.textContent = "Traiter les images";
      }
    });

    function crc32(bytes) {
      let crc = -1;
      for (let i = 0; i < bytes.length; i++) {
        crc ^= bytes[i];
        for (let j = 0; j < 8; j++) {
          crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
        }
      }
      return (crc ^ -1) >>> 0;
    }

    function u16(value) {
      return [value & 255, (value >>> 8) & 255];
    }

    function u32(value) {
      return [value & 255, (value >>> 8) & 255, (value >>> 16) & 255, (value >>> 24) & 255];
    }

    async function refreshZipDownload() {
      const files = processedFiles.filter(Boolean);
      if (!files.length) {
        zipDownload.hidden = true;
        zipDownload.removeAttribute("href");
        return;
      }

      zipDownload.href = objectUrl(await createZip(files));
      zipDownload.hidden = false;
    }

    async function createZip(files) {
      const zipTime = 0;
      const zipDate = 33;
      const encoder = new TextEncoder();
      const chunks = [];
      const central = [];
      let offset = 0;

      for (const file of files) {
        const name = encoder.encode(file.filename);
        const data = new Uint8Array(await file.blob.arrayBuffer());
        const checksum = crc32(data);
        const local = new Uint8Array([
          ...u32(0x04034b50), ...u16(20), ...u16(0x0800), ...u16(0), ...u16(zipTime), ...u16(zipDate),
          ...u32(checksum), ...u32(data.length), ...u32(data.length), ...u16(name.length), ...u16(0),
        ]);
        chunks.push(local, name, data);

        central.push(new Uint8Array([
          ...u32(0x02014b50), ...u16(20), ...u16(20), ...u16(0x0800), ...u16(0), ...u16(zipTime), ...u16(zipDate),
          ...u32(checksum), ...u32(data.length), ...u32(data.length), ...u16(name.length), ...u16(0), ...u16(0),
          ...u16(0), ...u16(0), ...u32(0), ...u32(offset),
        ]), name);
        offset += local.length + name.length + data.length;
      }

      const centralSize = central.reduce((sum, part) => sum + part.length, 0);
      const end = new Uint8Array([
        ...u32(0x06054b50), ...u16(0), ...u16(0), ...u16(files.length), ...u16(files.length),
        ...u32(centralSize), ...u32(offset), ...u16(0),
      ]);

      return new Blob([...chunks, ...central, end], { type: "application/zip" });
    }

    function parallelRequestCount() {
      const cores = navigator.hardwareConcurrency || 4;
      return Math.max(1, Math.min(selectedFiles.length, cores, 4));
    }

    async function runWithConcurrency(tasks, limit) {
      let next = 0;
      const workers = Array.from({ length: limit }, async () => {
        while (next < tasks.length) {
          const task = tasks[next++];
          await task();
        }
      });
      await Promise.all(workers);
    }

    function uploadImage(file, options) {
      return new Promise((resolve, reject) => {
        const body = new FormData();
        body.append("images", file, file.name);
        body.append("model", options.model);
        body.append("bg_mode", options.bgMode);
        if (options.background) {
          body.append("background_image", options.background, options.background.name);
        }

        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/ui/process");
        xhr.responseType = "blob";

        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            options.onUploadProgress(Math.min(file.size, event.loaded));
          }
        };

        xhr.onload = async () => {
          if (xhr.status < 200 || xhr.status >= 300) {
            const text = await xhr.response.text();
            reject(new Error(text || `Erreur HTTP ${xhr.status}`));
            return;
          }

          resolve({
            blob: xhr.response,
            contentType: xhr.getResponseHeader("content-type") || "",
          });
        };

        xhr.onerror = () => reject(new Error("Erreur reseau pendant l'upload"));
        xhr.upload.onload = () => options.onUploadProgress(file.size);
        xhr.send(body);
      });
    }

    async function readStoredZip(blob) {
      const bytes = new Uint8Array(await blob.arrayBuffer());
      const decoder = new TextDecoder();
      const files = [];
      let offset = 0;

      while (offset + 30 <= bytes.length) {
        const signature = readU32(bytes, offset);
        if (signature !== 0x04034b50) break;

        const flags = readU16(bytes, offset + 6);
        const method = readU16(bytes, offset + 8);
        const compressedSize = readU32(bytes, offset + 18);
        const uncompressedSize = readU32(bytes, offset + 22);
        const nameLength = readU16(bytes, offset + 26);
        const extraLength = readU16(bytes, offset + 28);
        const nameStart = offset + 30;
        const dataStart = nameStart + nameLength + extraLength;
        const dataEnd = dataStart + compressedSize;

        if (method !== 0) throw new Error("ZIP non supporte: compression detectee");
        if (dataEnd > bytes.length) throw new Error("ZIP invalide");

        const filename = decoder.decode(bytes.slice(nameStart, nameStart + nameLength));
        const data = bytes.slice(dataStart, dataEnd);
        if (data.length !== uncompressedSize) throw new Error("ZIP invalide");

        files.push({
          filename,
          blob: new Blob([data], { type: "image/png" }),
        });

        offset = dataEnd;
        if (flags & 0x0008) throw new Error("ZIP non supporte: data descriptor detecte");
      }

      return files;
    }

    function readU16(bytes, offset) {
      return bytes[offset] | (bytes[offset + 1] << 8);
    }

    function readU32(bytes, offset) {
      return (
        bytes[offset]
        | (bytes[offset + 1] << 8)
        | (bytes[offset + 2] << 16)
        | (bytes[offset + 3] << 24)
      ) >>> 0;
    }

    loadModels();
  </script>
</body>
</html>"##;
