// Global state
let currentProductId = "PROD_001";
let productsList = [];
let forecastChartInstance = null;
let simChartInstance = null;
let lossChartInstance = null;

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  loadProducts();
  loadOverview();
  loadModelInfo();
  fetchErpJson();
});

// Tab navigation
function setupTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      
      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");

      // Resize charts if becoming visible
      if (targetId === "tab-forecast" && forecastChartInstance) {
        forecastChartInstance.resize();
      } else if (targetId === "tab-simulator" && simChartInstance) {
        simChartInstance.resize();
      } else if (targetId === "tab-model" && lossChartInstance) {
        lossChartInstance.resize();
      }
    });
  });
}

// Format numbers
function formatMoney(val) {
  return new Intl.NumberFormat("tr-TR", { style: "currency", currency: "TRY", maximumFractionDigits: 0 }).format(val);
}

function formatNumber(val) {
  return new Intl.NumberFormat("tr-TR").format(val);
}

// Load products
async function loadProducts() {
  try {
    const res = await fetch("/api/products");
    productsList = await res.json();
    
    const fSelect = document.getElementById("forecastProductSelect");
    const sSelect = document.getElementById("simProductSelect");
    
    fSelect.innerHTML = "";
    sSelect.innerHTML = "";
    
    productsList.forEach(p => {
      const opt1 = document.createElement("option");
      opt1.value = p.id;
      opt1.textContent = `${p.name} (${p.category})`;
      fSelect.appendChild(opt1);
      
      const opt2 = document.createElement("option");
      opt2.value = p.id;
      opt2.textContent = `${p.name}`;
      sSelect.appendChild(opt2);
    });

    if (productsList.length > 0) {
      currentProductId = productsList[0].id;
      loadForecastData();
    }
  } catch (err) {
    console.error("Failed to load products:", err);
  }
}

// Load overview KPIs & Table
async function loadOverview() {
  try {
    const res = await fetch("/api/overview");
    const data = await res.json();
    const kpis = data.kpis;

    document.getElementById("kpiTotalDemand").innerText = formatNumber(kpis.total_7day_demand_units) + " adet";
    document.getElementById("kpiLostRevenue").innerText = formatMoney(kpis.total_lost_revenue_risk_try);
    document.getElementById("kpiWasteLoss").innerText = formatMoney(kpis.total_waste_risk_try);
    document.getElementById("kpiInventoryVal").innerText = formatMoney(kpis.total_inventory_value_try);
    document.getElementById("headerAccuracy").innerText = "%" + kpis.model_accuracy_pct;

    // Overview Table
    const tbody = document.getElementById("overviewTableBody");
    tbody.innerHTML = "";

    data.products_summary.forEach(p => {
      const tr = document.createElement("tr");

      let statusBadge = `<span class="status-tag status-optimal">Dengeli</span>`;
      if (p.action_type === "STOCKOUT_ALERT") {
        statusBadge = `<span class="status-tag status-danger">Tükenme Riski</span>`;
      } else if (p.action_type === "SPOILAGE_ALERT") {
        statusBadge = `<span class="status-tag status-warning">Zayiat Riski</span>`;
      } else if (p.action_type === "REORDER_WARNING") {
        statusBadge = `<span class="status-tag status-warning">Sipariş Eşiği</span>`;
      }

      tr.innerHTML = `
        <td style="font-weight: 700;">${p.name}</td>
        <td style="color: var(--text-secondary);">${p.category}</td>
        <td style="font-weight: 800;">${p.current_stock} adet</td>
        <td style="color: #38bdf8; font-weight: 800; font-size: 1.05rem;">${p.next_tuesday_sales} adet</td>
        <td style="font-weight: 700;">${p.total_7day_sales} adet</td>
        <td style="color: var(--text-muted);">${p.reorder_point} adet</td>
        <td>${statusBadge}</td>
        <td style="font-size: 0.8rem; max-width: 300px;">${p.action_recommendation}</td>
      `;
      tbody.appendChild(tr);
    });

    // Spotlight on first product Tuesday
    if (data.products_summary.length > 0) {
      const p1 = data.products_summary[0];
      document.getElementById("spotlightTueTotal").innerText = p1.next_tuesday_sales + " adet";
    }

  } catch (err) {
    console.error("Failed to load overview:", err);
  }
}

// Product dropdown change
function onProductChange() {
  currentProductId = document.getElementById("forecastProductSelect").value;
  loadForecastData();
}

// Load forecast details for selected product
async function loadForecastData() {
  try {
    const res = await fetch(`/api/forecast/${currentProductId}`);
    const data = await res.json();

    // 1. Update Tuesday Focus Card
    const tue = data.next_tuesday_forecast;
    document.getElementById("tueTargetTitle").innerText = `${data.product.name} İçin Salı Tahmini: ${tue.predicted_sales} Adet`;
    document.getElementById("tueTargetDetail").innerText = `TensorFlow %95 Güven Aralığı: [${tue.lower_bound} - ${tue.upper_bound} adet] | Tarih: ${tue.date} (Salı)`;
    document.getElementById("tueTargetDate").innerText = tue.date;

    // 2. Render 7 Days Breakdown Grid
    renderDaysGrid(data.future_7days);

    // 3. Render Chart
    renderForecastChart(data.recent_30days, data.future_7days, data.product.name);

    // 4. Render Inventory Analytics Cards
    renderInventoryCards(data.product, data.inventory_analytics);

  } catch (err) {
    console.error("Failed to load forecast data:", err);
  }
}

// Render days grid
function renderDaysGrid(days) {
  const container = document.getElementById("daysGridContainer");
  container.innerHTML = "";

  days.forEach(d => {
    const card = document.createElement("div");
    card.className = `day-card ${d.is_tuesday ? 'highlight-tuesday' : ''}`;
    
    card.innerHTML = `
      <div class="day-name">${d.day_name} ${d.is_tuesday ? '⭐' : ''}</div>
      <div class="day-date">${d.date}</div>
      <div class="day-sales">${d.predicted_sales}</div>
      <div class="day-unit">adet</div>
      <div class="day-ci">± [${d.lower_bound} - ${d.upper_bound}]</div>
    `;
    container.appendChild(card);
  });
}

// Render Forecast Chart with Chart.js
function renderForecastChart(recent30, future7, prodName) {
  const ctx = document.getElementById("forecastChart").getContext("2d");

  const labels = [
    ...recent30.map(r => r.date.substring(5)),
    ...future7.map(f => f.date.substring(5) + ` (${f.day_name.substring(0,3)})`)
  ];

  const actualData = [
    ...recent30.map(r => r.sales),
    ...new Array(future7.length).fill(null)
  ];

  // Bridge last historical point to first forecast point for continuous visualization
  const lastActual = recent30[recent30.length - 1].sales;
  const forecastData = [
    ...new Array(recent30.length - 1).fill(null),
    lastActual,
    ...future7.map(f => f.predicted_sales)
  ];

  const upperData = [
    ...new Array(recent30.length - 1).fill(null),
    lastActual,
    ...future7.map(f => f.upper_bound)
  ];

  const lowerData = [
    ...new Array(recent30.length - 1).fill(null),
    lastActual,
    ...future7.map(f => f.lower_bound)
  ];

  if (forecastChartInstance) {
    forecastChartInstance.destroy();
  }

  forecastChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Gerçekleşen Satışlar (Geçmiş 30 Gün)",
          data: actualData,
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.1)",
          borderWidth: 2,
          pointRadius: 2.5,
          pointHoverRadius: 6,
          fill: true,
          tension: 0.3
        },
        {
          label: "TensorFlow Bi-LSTM Tahmini (Gelecek 7 Gün)",
          data: forecastData,
          borderColor: "#6366f1",
          borderWidth: 3,
          pointRadius: 4,
          pointBackgroundColor: "#818cf8",
          pointHoverRadius: 7,
          tension: 0.3
        },
        {
          label: "Üst Güven Sınırı (%95)",
          data: upperData,
          borderColor: "rgba(56, 189, 248, 0.3)",
          borderDash: [4, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: "+1",
          backgroundColor: "rgba(56, 189, 248, 0.08)",
          tension: 0.3
        },
        {
          label: "Alt Güven Sınırı (%95)",
          data: lowerData,
          borderColor: "rgba(56, 189, 248, 0.3)",
          borderDash: [4, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          backgroundColor: "rgba(15, 23, 42, 0.95)",
          titleColor: "#f8fafc",
          bodyColor: "#94a3b8",
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1,
          padding: 12
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: { color: "#64748b", font: { size: 10 } }
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: { color: "#64748b", font: { size: 11 } }
        }
      }
    }
  });
}

// Render Inventory & Spoilage Cards
function renderInventoryCards(prod, inv) {
  const stockoutEl = document.getElementById("stockoutCardContent");
  const spoilageEl = document.getElementById("spoilageCardContent");

  // Stockout Card
  if (inv.stockout_risk) {
    stockoutEl.innerHTML = `
      <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.3); padding: 18px; border-radius: var(--radius-sm); margin-bottom: 16px;">
        <div style="font-weight: 700; color: #fb7185; font-size: 1rem; margin-bottom: 6px;">
          ⚠️ STOK TÜKENME TEHLİKESİ: ${inv.depletion_day} (${inv.depletion_date})
        </div>
        <div style="color: var(--text-primary); font-size: 0.9rem;">
          Mevcut stok (${inv.current_stock} adet), ${inv.depletion_day} günü tamamen sıfırlanacaktır.
        </div>
        <div style="margin-top: 10px; font-size: 0.85rem; color: #fecdd3;">
          • Karşılanamayan Talep: <strong>${inv.lost_sales_units} adet</strong><br>
          • Kaçırılacak Potansiyel Ciro: <strong>${formatMoney(inv.lost_revenue_try)}</strong>
        </div>
      </div>
      <div style="font-size: 0.9rem;">
        <strong>Önerilen Aksiyon:</strong> ${inv.action_recommendation}
      </div>
    `;
  } else {
    stockoutEl.innerHTML = `
      <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding: 18px; border-radius: var(--radius-sm); margin-bottom: 16px;">
        <div style="font-weight: 700; color: #34d399; font-size: 1rem; margin-bottom: 6px;">
          ✅ STOK YETERLİ (GÜVENLİ BÖLGE)
        </div>
        <div style="color: var(--text-primary); font-size: 0.9rem;">
          Mevcut stok (${inv.current_stock} adet), gelecek 7 günün toplam talebini (${inv.total_7day_demand} adet) eksiksiz karşılamaktadır.
        </div>
        <div style="margin-top: 10px; font-size: 0.85rem; color: #a7f3d0;">
          • Emniyet Stoğu: <strong>${inv.safety_stock} adet</strong><br>
          • Sipariş Noktası (ROP): <strong>${inv.reorder_point} adet</strong>
        </div>
      </div>
      <div style="font-size: 0.9rem;">
        <strong>Durum:</strong> ${inv.action_recommendation}
      </div>
    `;
  }

  // Spoilage Card
  if (inv.waste_risk) {
    spoilageEl.innerHTML = `
      <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); padding: 18px; border-radius: var(--radius-sm); margin-bottom: 16px;">
        <div style="font-weight: 700; color: #fbbf24; font-size: 1rem; margin-bottom: 6px;">
          🗑️ ZAYİAT / İSRAF RİSKİ MEVCUT
        </div>
        <div style="color: var(--text-primary); font-size: 0.9rem;">
          Bu ürünün raf ömrü <strong>${prod.shelf_life_days} gün</strong>dür. Raf ömrü dolmadan satılamayacak fazla stok bulunmaktadır!
        </div>
        <div style="margin-top: 10px; font-size: 0.85rem; color: #fde68a;">
          • Bozulma Tehlikesi Olan Miktar: <strong>${inv.spoiled_units} adet</strong><br>
          • Çöpe Gitme Riski Taşıyan Maliyet: <strong>${formatMoney(inv.waste_loss_try)}</strong>
        </div>
      </div>
      <div style="font-size: 0.9rem;">
        <strong>Önerilen Aksiyon:</strong> ${inv.action_recommendation}
      </div>
    `;
  } else {
    spoilageEl.innerHTML = `
      <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding: 18px; border-radius: var(--radius-sm); margin-bottom: 16px;">
        <div style="font-weight: 700; color: #34d399; font-size: 1rem; margin-bottom: 6px;">
          ✅ ZAYİAT RİSKİ YOK
        </div>
        <div style="color: var(--text-primary); font-size: 0.9rem;">
          Mevcut stok miktarı, raf ömrü (${prod.shelf_life_days} gün) içerisindeki tahmin edilen satış temposuyla güvenli biçimde tüketilecektir.
        </div>
      </div>
      <div style="font-size: 0.9rem;">
        <strong>Raf Ömrü Koruma Durumu:</strong> Zayiat maliyeti sıfır. Stok rotasyonu optimaldir.
      </div>
    `;
  }
}

// Run What-If Simulation
async function runWhatIfSimulation() {
  const pId = document.getElementById("simProductSelect").value;
  const temp = parseFloat(document.getElementById("simTemp").value);
  const weather = parseInt(document.getElementById("simWeather").value);
  const discount = parseFloat(document.getElementById("simDiscount").value) / 100.0;
  const holiday = parseInt(document.getElementById("simHoliday").value);

  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        product_id: pId,
        temperature: temp,
        weather_code: weather,
        discount_pct: discount,
        is_holiday: holiday
      })
    });

    const data = await res.json();
    const comp = data.simulation_comparison;

    document.getElementById("simResultsContainer").style.display = "block";
    document.getElementById("simSummaryHeader").innerText = `📊 Senaryo Sonucu: ${data.product.name} | ${comp.impact_summary}`;
    document.getElementById("simBaseTue").innerText = comp.baseline_tuesday_sales + " adet";
    document.getElementById("simResultTue").innerText = comp.simulated_tuesday_sales + " adet";
    document.getElementById("simDeltaTue").innerText = (comp.tuesday_delta_pct > 0 ? "+" : "") + comp.tuesday_delta_pct + "%";
    document.getElementById("simDeltaTotal").innerText = (comp.total_delta_pct > 0 ? "+" : "") + comp.total_delta_pct + "%";

    // Comparison Chart
    renderSimChart(data.future_7days, comp.baseline_7day_total, data.product.name);

  } catch (err) {
    console.error("Simulation error:", err);
  }
}

function renderSimChart(simDays, baselineTotal, prodName) {
  const ctx = document.getElementById("simComparisonChart").getContext("2d");
  const labels = simDays.map(d => `${d.day_name} (${d.date.substring(5)})`);
  const simValues = simDays.map(d => d.predicted_sales);

  if (simChartInstance) {
    simChartInstance.destroy();
  }

  simChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Simüle Edilen Günlük Talep",
          data: simValues,
          backgroundColor: "rgba(6, 182, 212, 0.65)",
          borderColor: "#06b6d4",
          borderWidth: 1.5,
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
        y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } }
      }
    }
  });
}

// Load Model Info & Loss Chart
async function loadModelInfo() {
  try {
    const res = await fetch("/api/model-info");
    const data = await res.json();
    const m = data.metrics;

    document.getElementById("metricMae").innerText = m.mae_units + " adet";
    document.getElementById("metricRmse").innerText = m.rmse_units + " adet";
    document.getElementById("metricMape").innerText = "%" + m.mape_pct;
    document.getElementById("metricAccuracy").innerText = "%" + m.accuracy_pct;

    // Render Loss Chart
    renderLossChart(data.loss_history, data.val_loss_history);

  } catch (err) {
    console.error("Failed to load model info:", err);
  }
}

function renderLossChart(loss, valLoss) {
  const ctx = document.getElementById("lossChart").getContext("2d");
  const epochs = loss.map((_, i) => `Epoch ${i + 1}`);

  if (lossChartInstance) {
    lossChartInstance.destroy();
  }

  lossChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        {
          label: "Eğitim Kaybı (Train Loss - Huber)",
          data: loss,
          borderColor: "#6366f1",
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.2
        },
        {
          label: "Doğrulama Kaybı (Val Loss - Huber)",
          data: valLoss,
          borderColor: "#f59e0b",
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8" } }
      },
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
        y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } }
      }
    }
  });
}

// Retrain model trigger
async function triggerRetrain() {
  const epochs = parseInt(document.getElementById("retrainEpochs").value) || 25;
  const statusEl = document.getElementById("retrainStatusText");
  const btn = document.getElementById("btnRetrain");

  btn.disabled = true;
  statusEl.innerText = "Model eğitimi arka planda başlatıldı...";

  try {
    const res = await fetch("/api/retrain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: epochs })
    });
    const data = await res.json();
    statusEl.innerText = data.message;
  } catch (err) {
    statusEl.innerText = "Hata oluştu: " + err;
  } finally {
    setTimeout(() => { btn.disabled = false; }, 4000);
  }
}

// Fetch ERP JSON Export
async function fetchErpJson() {
  try {
    const res = await fetch("/api/erp/export");
    const json = await res.json();
    document.getElementById("erpJsonBox").innerText = JSON.stringify(json, null, 2);
  } catch (err) {
    console.error("Failed to fetch ERP JSON:", err);
  }
}
