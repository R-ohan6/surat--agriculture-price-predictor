// Change this to your deployed Render URL once live, e.g.
  // const API_BASE = "https://surat-price-api.onrender.com";
  const API_BASE = "http://127.0.0.1:8000";

  const commoditySelect = document.getElementById("commodity");
  const dateInput = document.getElementById("target_date");
  const errorBox = document.getElementById("error");
  const board = document.getElementById("board");
  const btn = document.getElementById("predictBtn");

  // Default date = today
  dateInput.value = new Date().toISOString().split("T")[0];

  async function loadCommodities() {
    try {
      const res = await fetch(`${API_BASE}/commodities`);
      const commodities = await res.json();
      commoditySelect.innerHTML = commodities
        .map(c => `<option value="${c}">${c}</option>`)
        .join("");
    } catch (e) {
      showError("Couldn't reach the prediction API. Is it running?");
    }
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.add("visible");
  }

  function clearError() {
    errorBox.classList.remove("visible");
  }

  async function predict() {
    clearError();
    btn.disabled = true;
    btn.textContent = "Checking market...";

    const payload = {
      commodity: commoditySelect.value,
      target_date: dateInput.value,
      rainfall_mm: parseFloat(document.getElementById("rainfall_mm").value) || 0,
      rainfall_7day_sum: parseFloat(document.getElementById("rainfall_7day").value) || 0,
      temp_max: parseFloat(document.getElementById("temp_max").value),
      temp_min: parseFloat(document.getElementById("temp_min").value),
    };

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed.");
      }

      const data = await res.json();

      document.getElementById("boardCommodity").textContent = data.commodity;
      document.getElementById("boardPrice").textContent = `₹${Math.round(data.predicted_price).toLocaleString("en-IN")}`;
      document.getElementById("boardLast").textContent = data.last_reported_price
        ? `₹${Math.round(data.last_reported_price).toLocaleString("en-IN")}`
        : "N/A";
      document.getElementById("boardDate").textContent = data.target_date;

      board.classList.add("visible");
    } catch (e) {
      showError(e.message || "Something went wrong. Please try again.");
      board.classList.remove("visible");
    } finally {
      btn.disabled = false;
      btn.textContent = "Check Fair Value";
    }
  }

  loadCommodities();
