let rowCount = 0;

function addRubricRow(name = "", points = 1) {
    rowCount++;
    const container = document.getElementById("rubric-container");
    const div = document.createElement("div");
    div.className = "rubric-row d-flex align-items-end gap-2 mb-2";
    div.id = `rubric-row-${rowCount}`;

    div.innerHTML = `
        <div style="flex:3">
            <label class="form-label small text-muted mb-1">Step Name</label>
            <input type="text"
                   class="form-control form-control-sm step-name"
                   placeholder="e.g. data_loading"
                   value="${name}"
                   required>
        </div>
        <div style="flex:1">
            <label class="form-label small text-muted mb-1">Marks</label>
            <input type="number"
                   class="form-control form-control-sm step-points"
                   min="1" value="${points}"
                   required>
        </div>
        <div>
            <button type="button"
                    class="btn btn-outline-danger btn-sm"
                    onclick="removeRow('rubric-row-${rowCount}')">✕</button>
        </div>
    `;

    container.appendChild(div);
    updateTotalMarks();
    div.querySelector(".step-points").addEventListener("input", updateTotalMarks);
}

function removeRow(rowId) {
    const row = document.getElementById(rowId);
    if (row) {
        row.remove();
        updateTotalMarks();
    }
}

function updateTotalMarks() {
    let total = 0;
    document.querySelectorAll(".step-points").forEach(input => {
        total += parseInt(input.value) || 0;
    });
    const box     = document.getElementById("rubric-preview-box");
    const totalEl = document.getElementById("total-marks");
    if (totalEl) totalEl.textContent = total;
    if (box)     box.style.display = total > 0 ? "block" : "none";
}

function buildRubricJSON() {
    const rubric = {};
    document.querySelectorAll(".rubric-row").forEach(row => {
        const name   = row.querySelector(".step-name")?.value.trim();
        const points = parseInt(row.querySelector(".step-points")?.value) || 0;
        if (name) {
            rubric[name] = {
                points:     points,
                depends_on: [],
            };
        }
    });
    return rubric;
}

function handleTaskTypeChange(value) {
    const customInput = document.getElementById("customTaskInput");
    const hiddenInput = document.getElementById("taskTypeHidden");

    if (value === "Custom") {
        customInput.style.display = "block";
        customInput.required      = true;
        customInput.name          = "task_type";
        hiddenInput.name          = "";
    } else {
        customInput.style.display = "none";
        customInput.required      = false;
        customInput.name          = "";
        hiddenInput.name          = "task_type";
        hiddenInput.value         = value;
    }
}

document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("assignForm");
    if (!form) return;

    addRubricRow();

    form.addEventListener("submit", function (e) {
        const select      = document.getElementById("taskTypeSelect");
        const customInput = document.getElementById("customTaskInput");

        if (!select.value) {
            e.preventDefault();
            alert("Please select a task type.");
            return;
        }

        if (select.value === "Custom" && !customInput.value.trim()) {
            e.preventDefault();
            alert("Please describe the custom task type.");
            customInput.focus();
            return;
        }

        const rubric = buildRubricJSON();
        if (Object.keys(rubric).length === 0) {
            e.preventDefault();
            alert("Please add at least one rubric step.");
            return;
        }

        document.getElementById("hidden_rubric_json").value = JSON.stringify(rubric);
    });
});