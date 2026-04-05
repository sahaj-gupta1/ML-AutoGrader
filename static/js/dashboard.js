// ── Teacher Dashboard JS ─────────────────────────────────

document.addEventListener("DOMContentLoaded", function () {

    // Search filter
    document.querySelectorAll(".search-input").forEach(input => {
        input.addEventListener("keyup", function () {
            const filter  = this.value.toLowerCase();
            const tableId = this.dataset.table;
            const table   = document.getElementById(tableId);
            if (!table) return;
            table.querySelectorAll("tbody tr").forEach(row => {
                const enroll = row.querySelector(".enroll")?.textContent.toLowerCase() || "";
                const name   = row.querySelector(".sname")?.textContent.toLowerCase()  || "";
                row.style.display =
                    (enroll.includes(filter) || name.includes(filter)) ? "" : "none";
            });
        });
    });

    // Grade All buttons
    document.querySelectorAll(".grade-all-btn").forEach(btn => {
        btn.addEventListener("click", async function () {
            const assignId = this.dataset.assignId;
            if (!confirm("Start grading all new submissions?\n\nResults will appear as each submission is graded.")) return;

            this.disabled    = true;
            this.textContent = "⏳ Starting...";

            try {
                const resp = await fetch(`/teacher/grade_all/${assignId}`, { method: "POST" });
                const data = await resp.json();

                if (data.success) {
                    showToast(data.message, "bg-primary");
                    this.textContent = "⏳ Grading...";

                    // Show progress bar
                    const pb = document.getElementById(`progress-bar-${assignId}`);
                    if (pb) pb.classList.remove("d-none");

                    // Update status badge
                    const badge = document.getElementById(`status-badge-${assignId}`);
                    if (badge) {
                        badge.className = "badge bg-warning text-dark";
                        badge.textContent = "Grading...";
                    }

                    startPolling(assignId);
                } else {
                    showToast(data.message, "bg-danger");
                    this.disabled    = false;
                    this.textContent = "🤖 Grade All";
                }
            } catch (err) {
                showToast("Network error. Try again.", "bg-danger");
                this.disabled    = false;
                this.textContent = "🤖 Grade All";
            }
        });
    });

});


// ── Live polling ──────────────────────────────────────────
function startPolling(assignId) {
    const interval = setInterval(async () => {
        try {
            const resp = await fetch(`/teacher/grading_status/${assignId}`);
            const data = await resp.json();

            updateProgressUI(assignId, data);

            if (data.assignment_status === "completed") {
                clearInterval(interval);
                onGradingComplete(assignId, data);
            }
        } catch (err) {
            clearInterval(interval);
        }
    }, 3000); // poll every 3 seconds
}


function updateProgressUI(assignId, data) {
    const total   = data.total   || 0;
    const graded  = data.graded  || 0;
    const pct     = total > 0 ? Math.round((graded / total) * 100) : 0;

    // Progress bar fill
    const fill = document.getElementById(`progress-fill-${assignId}`);
    if (fill) fill.style.width = pct + "%";

    // Progress label
    const label = document.getElementById(`progress-label-${assignId}`);
    if (label) label.textContent = `Grading in progress...`;

    const count = document.getElementById(`progress-count-${assignId}`);
    if (count) count.textContent = `${graded} / ${total} graded`;

    // Per-student breakdown list
    const list = document.getElementById(`progress-list-${assignId}`);
    if (list && data.breakdown) {
        list.innerHTML = data.breakdown.map(s => {
            let icon  = "⏳";
            let color = "text-muted";
            if (s.status === "graded") {
                icon  = "✅";
                color = "text-success";
            } else if (s.status === "grading") {
                icon  = "🔄";
                color = "text-warning";
            }
            const score = s.status === "graded" && s.score !== null
                ? ` — <strong>${s.score}/${s.max_score}</strong>`
                : "";
            return `<span class="${color} me-3">${icon} ${s.name} (${s.enrollment})${score}</span>`;
        }).join("");
    }

    // Update graded count in stats bar
    const gradedCount = document.getElementById(`graded-count-${assignId}`);
    if (gradedCount) gradedCount.textContent = graded;
}


function onGradingComplete(assignId, data) {
    // Update progress bar to full green
    const fill = document.getElementById(`progress-fill-${assignId}`);
    if (fill) {
        fill.style.width = "100%";
        fill.className   = fill.className
            .replace("bg-warning", "bg-success")
            .replace("progress-bar-animated", "")
            .replace("progress-bar-striped", "");
    }

    const label = document.getElementById(`progress-label-${assignId}`);
    if (label) label.textContent = "✅ Grading complete!";

    // Update status badge
    const badge = document.getElementById(`status-badge-${assignId}`);
    if (badge) {
        badge.className   = "badge bg-success";
        badge.textContent = "Graded";
    }

    // Re-enable grade button
    const btn = document.getElementById(`grade-btn-${assignId}`);
    if (btn) {
        btn.disabled    = true;
        btn.textContent = "✅ Graded";
        btn.className   = btn.className.replace("btn-primary", "btn-secondary");
    }

    showToast("✅ Grading complete! Reloading...", "bg-success");
    setTimeout(() => window.location.reload(), 2000);
}


// ── Toast ─────────────────────────────────────────────────
function showToast(message, bgClass = "bg-primary") {
    const toastEl  = document.getElementById("grading-toast");
    const toastMsg = document.getElementById("grading-toast-msg");
    if (!toastEl || !toastMsg) return;

    toastEl.className = toastEl.className.replace(/bg-\S+/, bgClass);
    toastMsg.textContent = message;

    const toast = new bootstrap.Toast(toastEl, { delay: 5000 });
    toast.show();
}
