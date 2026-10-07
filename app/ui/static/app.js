// The admin UI's only script: confirmations, job progress, the filters, copying a
// command, and the start.sh line that follows a form. Loaded with the page's CSP nonce; it
// writes text only (textContent), never HTML.
"use strict";

const FINAL = ["done", "failed", "cancelled"];
const API = "./start.sh --admin";

function followJob(section) {
  const id = section.dataset.jobId;
  const status = document.getElementById("job-status");
  const progress = document.getElementById("job-progress");
  const message = document.getElementById("job-message");
  const tick = async () => {
    try {
      const response = await fetch(`/ui/jobs/${encodeURIComponent(id)}/status`, {
        credentials: "same-origin",
        cache: "no-store",
      });
      if (response.status === 401) {
        status.textContent = "signed out (the application restarted?): sign in again";
        return;
      }
      if (response.ok) {
        const job = await response.json();
        status.textContent = job.status;
        message.textContent = job.message || job.error || "";
        if (typeof job.progress === "number") {
          progress.value = job.progress;
        }
        if (FINAL.includes(job.status)) {
          window.location.reload();
          return;
        }
      } else {
        status.textContent = `waiting (HTTP ${response.status})`;
      }
    } catch (error) {
      // The application may be restarting (a host job): keep asking.
      status.textContent = "waiting for the application...";
    }
    window.setTimeout(tick, 1000);
  };
  window.setTimeout(tick, 1000);
}

// The quoting of Python's shlex.quote, which the server uses for the first line.
function quote(value) {
  if (value === "") {
    return "''";
  }
  if (/^[A-Za-z0-9@%+=:,./_-]+$/.test(value)) {
    return value;
  }
  return "'" + value.replace(/'/g, "'\"'\"'") + "'";
}

// The form's values as `./start.sh --admin COMMAND --flag value ...`: what the server writes in
// app/ui/parity.py::cli_line, kept in step while the administrator types.
function cliLine(form) {
  const parts = [API, form.dataset.cli];
  for (const input of form.querySelectorAll("[data-flag]")) {
    let value = input.value;
    if (!value || !value.trim()) {
      continue;
    }
    if (input.dataset.secret === "yes") {
      parts.push(input.dataset.flag, "-");
      continue;
    }
    if (input.dataset.lines === "yes" && !value.trim().startsWith("[")) {
      const lines = value.split("\n").map((l) => l.trim()).filter((l) => l);
      value = JSON.stringify(lines);
    }
    parts.push(input.dataset.flag, quote(value));
  }
  return parts.join(" ");
}

function filterRows(input) {
  const rows = document.querySelectorAll(input.dataset.filter);
  const words = input.value.toLowerCase().trim().split(/\s+/).filter((w) => w);
  for (const row of rows) {
    const text = row.textContent.toLowerCase();
    row.hidden = !words.every((w) => text.includes(w));
  }
  for (const group of document.querySelectorAll("[data-group]")) {
    const visible = group.querySelectorAll("tbody tr:not([hidden])").length;
    group.hidden = visible === 0;
  }
}

async function copyText(button, text) {
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "Copied";
  } catch (error) {
    button.textContent = "Select and copy";
  }
  window.setTimeout(() => {
    button.textContent = "Copy";
  }, 1500);
}

document.addEventListener("DOMContentLoaded", () => {
  for (const form of document.querySelectorAll("form[data-confirm]")) {
    form.addEventListener("submit", (event) => {
      if (!window.confirm(form.dataset.confirm)) {
        event.preventDefault();
      }
    });
  }
  const job = document.getElementById("job");
  if (job && job.dataset.final !== "yes") {
    followJob(job);
  }
  const line = document.getElementById("cli-line");
  const form = document.querySelector("form[data-cli]");
  if (line && form) {
    const update = () => {
      line.textContent = cliLine(form);
    };
    form.addEventListener("input", update);
    form.addEventListener("change", update);
  }
  for (const input of document.querySelectorAll("input[data-filter]")) {
    input.addEventListener("input", () => filterRows(input));
  }
  for (const button of document.querySelectorAll("[data-copy-term]")) {
    button.addEventListener("click", () => {
      const lines = button.closest(".term").querySelectorAll(".line");
      copyText(button, Array.from(lines, (l) => l.textContent).join("\n"));
    });
  }
});
