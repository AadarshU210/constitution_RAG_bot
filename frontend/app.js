const API = "/api/v1/chat";

const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send");
const thread = document.getElementById("thread");
const welcome = document.getElementById("welcome");
const stage = document.getElementById("stage");
const statusEl = document.getElementById("status");
const sourcesPanel = document.getElementById("sources-panel");
const sourcesList = document.getElementById("sources-list");
const closeSources = document.getElementById("close-sources");
const suggestions = document.getElementById("suggestions");

const sourceCache = new Map();
let busy = false;

function autoGrow() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 128)}px`;
}

input.addEventListener("input", autoGrow);

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

suggestions?.addEventListener("click", (event) => {
  const btn = event.target.closest("button[data-q]");
  if (!btn || busy) return;
  input.value = btn.dataset.q;
  autoGrow();
  form.requestSubmit();
});

closeSources?.addEventListener("click", () => {
  sourcesPanel.hidden = true;
});

function showThread() {
  welcome.hidden = true;
  thread.hidden = false;
}

function appendBubble(role, text, extra = {}) {
  showThread();
  const wrap = document.createElement("article");
  wrap.className = `bubble ${role}${extra.error ? " error" : ""}${extra.typing ? " typing" : ""}`;

  const meta = document.createElement("p");
  meta.className = "meta";
  meta.textContent = role === "user" ? "You" : "Samvidhaan";

  const body = document.createElement("div");
  body.className = "body";

  if (extra.typing) {
    body.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  } else {
    body.textContent = text;
  }

  wrap.append(meta, body);

  if (extra.messageId && extra.sources?.length) {
    sourceCache.set(extra.messageId, extra.sources);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sources-btn";
    btn.textContent = `View ${extra.sources.length} source${extra.sources.length === 1 ? "" : "s"}`;
    btn.addEventListener("click", () => openSources(extra.messageId));
    wrap.append(btn);
  }

  thread.append(wrap);
  stage.scrollTop = stage.scrollHeight;
  return wrap;
}

function openSources(messageId) {
  const sources = sourceCache.get(messageId) || [];
  sourcesList.innerHTML = "";
  for (const src of sources) {
    const item = document.createElement("article");
    item.className = "source-item";
    const title = src.article
      ? `Article ${src.article}${src.article_title ? ` — ${src.article_title}` : ""}`
      : src.article_title || src.chunk_id || "Excerpt";
    const pages =
      src.page_start != null
        ? `pp. ${src.page_start}${src.page_end && src.page_end !== src.page_start ? `–${src.page_end}` : ""}`
        : "";
    const part = [src.part, src.part_title].filter(Boolean).join(" · ");
    item.innerHTML = `
      <h4>${escapeHtml(title)}</h4>
      <p class="sub">${escapeHtml([part, pages].filter(Boolean).join(" · "))}</p>
      <p>${escapeHtml(src.excerpt || "")}</p>
    `;
    sourcesList.append(item);
  }
  sourcesPanel.hidden = false;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function setBusy(next) {
  busy = next;
  sendBtn.disabled = next;
  input.disabled = next;
  statusEl.textContent = next ? "Retrieving…" : "";
}

async function ask(question) {
  const typing = appendBubble("assistant", "", { typing: true });
  setBusy(true);
  try {
    const res = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json().catch(() => ({}));
    typing.remove();

    if (!res.ok) {
      const detail =
        typeof data.detail === "string"
          ? data.detail
          : Array.isArray(data.detail)
            ? data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
            : `Request failed (${res.status})`;
      appendBubble("assistant", detail, { error: true });
      return;
    }

    const messageId = `m-${Date.now()}`;
    appendBubble("assistant", data.answer || "(empty answer)", {
      messageId,
      sources: data.sources || [],
    });
  } catch (err) {
    typing.remove();
    appendBubble(
      "assistant",
      "Could not reach the API. Is the server running on this host?",
      { error: true },
    );
    console.error(err);
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const question = input.value.trim();
  if (!question) return;
  appendBubble("user", question);
  input.value = "";
  autoGrow();
  await ask(question);
});
