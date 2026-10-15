// Solar Sense - Asynchronous Report Table Pagination Engine
// Maintains state, smooth transitions, and dynamic fetch without page reload

document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".report-result[data-url]").forEach((result) => {
        const previous = result.querySelector('[data-step="-1"]');
        const next = result.querySelector('[data-step="1"]');
        const status = result.querySelector(".page-status");
        const error = result.querySelector(".table-error");

        if (!previous || !next) return;

        result.querySelectorAll("[data-step]").forEach((button) => {
            button.addEventListener("click", async (e) => {
                e.preventDefault();
                e.stopPropagation();

                const currentPage = parseInt(result.dataset.page || "1", 10);
                const step = parseInt(button.dataset.step || "0", 10);
                const targetPage = currentPage + step;

                if (targetPage < 1) return;

                const oldPrevState = previous.disabled;
                const oldNextState = next.disabled;
                previous.disabled = true;
                next.disabled = true;

                if (error) error.textContent = "";
                result.setAttribute("aria-busy", "true");

                try {
                    const dataUrl = new URL(result.dataset.url, window.location.origin);
                    dataUrl.searchParams.set("page", targetPage);

                    const response = await fetch(dataUrl.toString(), {
                        headers: { "Accept": "application/json" }
                    });

                    if (!response.ok) {
                        throw new Error(`Server returned HTTP ${response.status}`);
                    }

                    const data = await response.json();
                    const tbody = result.querySelector("tbody");
                    if (!tbody) throw new Error("Table body not found");

                    const fragment = document.createDocumentFragment();
                    data.rows.forEach((row) => {
                        const tr = document.createElement("tr");
                        row.forEach((cellValue, idx) => {
                            const td = idx === 0 ? document.createElement("th") : document.createElement("td");
                            if (idx === 0) td.setAttribute("scope", "row");
                            td.textContent = cellValue !== null && cellValue !== undefined ? cellValue : "—";
                            tr.appendChild(td);
                        });
                        fragment.appendChild(tr);
                    });

                    tbody.replaceChildren(fragment);
                    result.dataset.page = data.page;

                    if (status) {
                        status.textContent = `Page ${data.page} · Rows ${data.start}–${data.end}`;
                    }

                    previous.disabled = data.page <= 1;
                    next.disabled = !data.has_next;
                } catch (err) {
                    if (error) {
                        error.textContent = `Unable to load page: ${err.message}`;
                    }
                    previous.disabled = oldPrevState;
                    next.disabled = oldNextState;
                } finally {
                    result.removeAttribute("aria-busy");
                }
            });
        });
    });
});
