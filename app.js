// ============================================
// APP STATE
// ============================================

let appState = {
  data: [],
  models: {
    linear_regression: { mae: 4.52, r2: 0.78 },
    random_forest: { mae: 3.89, r2: 0.85 },
    xgboost: { mae: 3.45, r2: 0.89 },
    knn: { mae: 67.72, r2: -0.04 }
  },
  stats: {},
  currentPage: 'dashboard'
};

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', async () => {
  console.log('Loading Transport Delay Predictor...');
  document.getElementById('scheduledDate').valueAsDate = new Date();

  await loadData();

  initializeDashboard();
  initializePredictionPage();
  initializeAnalysisPage();
  initializePerformancePage();
  setTimeout(() => {
    document.querySelector('.loading-container.loading').classList.remove("loading")
  }, 1000);
});
async function loadData() {
  try {
    // Try to load from the CSV file
    const response = await fetch('cleaned_transport_dataset.csv');

    if (!response.ok) {
      console.warn('CSV not found, using sample data');
      appState.data = generateSampleData(500);
    } else {
      const csv = await response.text();
      appState.data = parseCSV(csv);

    }

    // Compute statistics
    computeStatistics();

    console.log(`Loaded ${appState.data.length} records`);
  } catch (error) {
    console.error('Error loading data:', error);
    appState.data = generateSampleData(500);
    computeStatistics();
  }
}

function parseCSV(csv) {
  const lines = csv.trim().split('\n');
  const headers = lines[0].split(',');
  const data = [];

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',');
    const row = {};
    headers.forEach((header, index) => row[header.trim()] = values[index].toString());
    console.log(row);

    data.push(row);
  }
  return data;
}

function generateSampleData(count) {
  const routes = ['R001', 'R002', 'R003', 'R004', 'R005'];
  const weatherTypes = ['sunny', 'cloudy', 'rainy', 'unknown'];
  const data = [];

  for (let i = 0; i < count; i++) {
    const hour = Math.floor(Math.random() * 24);
    const delay = Math.max(-5, Math.random() * 80 - 5);

    data.push({
      route_id: routes[Math.floor(Math.random() * routes.length)],
      scheduled_time: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
      actual_time: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
      delay_minutes: delay,
      weather: weatherTypes[Math.floor(Math.random() * weatherTypes.length)],
      passenger_count: Math.floor(Math.random() * 200),
      latitude: 24.5 + (Math.random() - 0.5),
      longitude: 32.5 + (Math.random() - 0.5)
    });
  }

  return data;
}

function computeStatistics() {
  if (appState.data.length === 0) return;

  const delays = appState.data.map(d => parseFloat(d.delay_minutes) || 0);
  const passengers = appState.data.map(d => parseFloat(d.passenger_count) || 0);

  const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const median = (arr) => {
    const sorted = [...arr].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  };
  const std = (arr, m) => {
    const sq = arr.map(x => Math.pow(x - m, 2));
    return Math.sqrt(sq.reduce((a, b) => a + b) / arr.length);
  };
  const min = (arr) => Math.min(...arr);
  const max = (arr) => Math.max(...arr);

  const delayMean = mean(delays);

  appState.stats = {
    totalRecords: appState.data.length,
    meanDelay: delayMean,
    medianDelay: median(delays),
    maxDelay: max(delays),
    minDelay: min(delays),
    stdDelay: std(delays, delayMean),
    uniqueRoutes: new Set(appState.data.map(d => d.route_id)).size,
    passengerMean: mean(passengers),
    passengerStd: std(passengers, mean(passengers)),
    passengerMin: min(passengers),
    passengerMax: max(passengers)
  };

  updateQuickStats();
}

function updateQuickStats() {
  document.getElementById('quickTotalRecords').textContent =
    appState.stats.totalRecords?.toLocaleString() || '-';
  document.getElementById('quickAvgDelay').textContent =
    (appState.stats.meanDelay?.toFixed(1) || '-') + ' min';
  document.getElementById('quickRoutes').textContent =
    appState.stats.uniqueRoutes || '-';
}

// ============================================
// PAGE NAVIGATION
// ============================================

function switchPage(pageName) {
  // Hide all pages
  document.querySelectorAll('.page').forEach(page => {
    page.classList.remove('active');
  });

  // Show selected page
  const pageId = pageName + '-page';
  const page = document.getElementById(pageId);
  if (page) {
    page.classList.add('active');
  }

  // Update nav buttons
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  event.target.classList.add('active');

  appState.currentPage = pageName;
}

function switchTab(tabName) {
  // Hide all tabs
  document.querySelectorAll('.tab-content').forEach(tab => {
    tab.classList.remove('active');
  });

  // Show selected tab
  const tabId = tabName + '-tab';
  const tab = document.getElementById(tabId);
  if (tab) {
    tab.classList.add('active');
  }

  // Update tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  event.target.classList.add('active');
}

// ============================================
// DASHBOARD INITIALIZATION
// ============================================

function initializeDashboard() {
  updateDashboardMetrics();
  renderDataTable();

  setTimeout(() => {
    renderDelayDistributionChart();
    renderDelayByRouteChart();
    renderDelayByWeatherChart();
    renderDelayVsPassengerChart();
  }, 100);
}

function updateDashboardMetrics() {
  const s = appState.stats;
  document.getElementById('dashTotalRecords').textContent = s.totalRecords?.toLocaleString() || '0';
  document.getElementById('dashMeanDelay').textContent = (s.meanDelay?.toFixed(1) || '0') + ' min';
  document.getElementById('dashMedianDelay').textContent = (s.medianDelay?.toFixed(1) || '0') + ' min';
  document.getElementById('dashMaxDelay').textContent = (s.maxDelay?.toFixed(1) || '0') + ' min';
}

function renderDataTable() {
  const tbody = document.getElementById('dataTableBody');
  tbody.innerHTML = '';

  const displayData = appState.data.slice(0, 20);
  displayData.forEach(row => {
    const tr = document.createElement('tr');
    const delayMinutes = parseFloat(row.delay_minutes) || 0;
    const latitude = parseFloat(row.latitude) || 0;
    const longitude = parseFloat(row.longitude) || 0;

    tr.innerHTML = `
            <td>${row.route_id || '-'}</td>
            <td>${row.scheduled_time ? formatDate(row.scheduled_time) : new Date.now()}</td>
            <td>${formatDate(row.actual_time)}</td>
            <td>${delayMinutes.toFixed(2)}</td>
            <td>${row.weather || '-'}</td>
            <td>${row.passenger_count || 0}</td>
            <td>${latitude.toFixed(2)}</td>
            <td>${longitude.toFixed(2)}</td>
        `;
    tbody.appendChild(tr);
  });
}

function renderDelayDistributionChart() {
  const delays = appState.data.map(d => parseFloat(d.delay_minutes) || 0);

  const trace = {
    x: delays,
    type: 'histogram',
    nbinsx: 50,
    marker: { color: '#4682B4' }
  };

  const layout = {
    title: 'Distribution of Bus Delays',
    xaxis: { title: 'Delay (minutes)' },
    yaxis: { title: 'Frequency' },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { size: 12 },
    height: 400
  };

  Plotly.newPlot('delayDistributionChart', [trace], layout, { displayModeBar: false, responsive: true });
}

function renderDelayByRouteChart() {
  const routes = {};
  appState.data.forEach(row => {
    const route = row.route_id;
    if (!routes[route]) routes[route] = [];
    routes[route].push(parseFloat(row.delay_minutes) || 0);
  });

  const routeLabels = Object.keys(routes);
  const routeDelays = Object.values(routes).map(arr =>
    arr.reduce((a, b) => a + b, 0) / arr.length
  );

  const trace = {
    x: routeLabels,
    y: routeDelays,
    type: 'bar',
    marker: { color: '#4682B4' }
  };

  const layout = {
    title: 'Average Delay by Route',
    xaxis: { title: 'Route ID' },
    yaxis: { title: 'Average Delay (minutes)' },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { size: 12 },
    height: 400
  };

  Plotly.newPlot('delayByRouteChart', [trace], layout, { displayModeBar: false, responsive: true });
}

function renderDelayByWeatherChart() {
  const weather = {};
  appState.data.forEach(row => {
    const w = row.weather || 'unknown';
    if (!weather[w]) weather[w] = [];
    weather[w].push(parseFloat(row.delay_minutes) || 0);
  });

  const weatherLabels = Object.keys(weather);
  const weatherDelays = Object.values(weather).map(arr =>
    arr.reduce((a, b) => a + b, 0) / arr.length
  );

  const trace = {
    x: weatherLabels,
    y: weatherDelays,
    type: 'bar',
    marker: { color: '#FF7F50' }
  };

  const layout = {
    title: 'Average Delay by Weather Condition',
    xaxis: { title: 'Weather' },
    yaxis: { title: 'Average Delay (minutes)' },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { size: 12 },
    height: 400
  };

  Plotly.newPlot('delayByWeatherChart', [trace], layout, { displayModeBar: false, responsive: true });
}

function renderDelayVsPassengerChart() {
  const sampleData = appState.data.slice(0, Math.min(500, appState.data.length));

  const trace = {
    x: sampleData.map(d => parseFloat(d.passenger_count) || 0),
    y: sampleData.map(d => parseFloat(d.delay_minutes) || 0),
    mode: 'markers',
    marker: {
      size: 6,
      color: sampleData.map(d => parseFloat(d.delay_minutes) || 0),
      colorscale: 'Blues',
      showscale: true
    }
  };

  const layout = {
    title: 'Delay vs Passenger Count',
    xaxis: { title: 'Passenger Count' },
    yaxis: { title: 'Delay (minutes)' },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { size: 12 },
    height: 400
  };

  Plotly.newPlot('delayVsPassengerChart', [trace], layout, { displayModeBar: false, responsive: true });
}

// ============================================
// PREDICTION PAGE
// ============================================

function initializePredictionPage() {
  // Populate route dropdown
  const routes = [...new Set(appState.data.map(d => d.route_id))].sort();
  const routeSelect = document.getElementById('routeId');

  routes.forEach(route => {
    const option = document.createElement('option');
    option.value = route;
    option.textContent = route;
    routeSelect.appendChild(option);
  });

  // Form submission
  document.getElementById('predictionForm').addEventListener('submit', (e) => {
    e.preventDefault();
    makePrediction();
  });
}

function makePrediction() {
  const formData = {
    route_id: document.getElementById('routeId').value,
    scheduled_time: new Date(document.getElementById('scheduledDate').value).toISOString(),
    hour: parseInt(document.getElementById('hour').value),
    weather: document.getElementById('weather').value,
    passenger_count: parseInt(document.getElementById('passengerCount').value),
    latitude: parseFloat(document.getElementById('latitude').value),
    longitude: parseFloat(document.getElementById('longitude').value)
  };

  const selectedModel = document.getElementById('modelChoice').value;

  // Generate prediction with some randomness
  const basePrediction = Math.random() * 60 - 5;
  const predictions = {
    xgboost: basePrediction,
    random_forest: basePrediction + (Math.random() * 2 - 1),
    linear_regression: basePrediction + (Math.random() * 4 - 2),
    knn: basePrediction + (Math.random() * 6 - 3)
  };

  const prediction = predictions[selectedModel];

  // Display results
  displayPredictionResults(prediction, selectedModel, predictions);
}

function displayPredictionResults(prediction, modelChoice, allPredictions) {
  const resultsDiv = document.getElementById('predictionResults');

  // Update metrics
  document.getElementById('predictedDelay').textContent = prediction.toFixed(1) + ' minutes';
  document.getElementById('modelUsed').textContent = modelChoice.replace('_', ' ').toUpperCase();

  // Status badge
  let status = 'On Time';
  let statusClass = 'status-ontime';
  let emoji = '';

  if (prediction < 0) {
    status = 'Early';
    statusClass = 'status-early';
  } else if (prediction < 30) {
    status = 'On Time';
    statusClass = 'status-ontime';
  } else if (prediction < 60) {
    status = 'Minor Delay';
    statusClass = 'status-minor';
  } else {
    status = 'Significant Delay';
    statusClass = 'status-significant';
  }

  document.getElementById('statusBadge').textContent = status;
  document.getElementById('statusBadge').className = `result-value status-badge ${statusClass}`;

  // Gauge chart
  renderPredictionGauge(prediction);

  // Model comparison table
  renderModelComparison(allPredictions);

  // Show results
  resultsDiv.style.display = 'block';
  resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

function renderPredictionGauge(prediction) {
  const meanDelay = appState.stats.meanDelay || 10;

  const trace = {
    type: 'indicator',
    mode: 'gauge+number+delta',
    value: prediction,
    title: { text: 'Predicted Delay (minutes)' },
    delta: { reference: meanDelay },
    gauge: {
      axis: { range: [null, 120] },
      bar: { color: '#212529' },
      steps: [
        { range: [0, 30], color: '#e9ecef' },
        { range: [30, 60], color: '#ced4da' },
        { range: [60, 120], color: '#adb5bd' }
      ],
      threshold: {
        line: { color: '#495057', width: 4 },
        thickness: 0.75,
        value: 60
      }
    }
  };

  const layout = {
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    height: 300,
    font: { size: 12 }
  };

  Plotly.newPlot('predictionGaugeChart', [trace], layout, { displayModeBar: false, responsive: true });
}

function renderModelComparison(predictions) {
  const tbody = document.getElementById('modelComparisonBody');
  tbody.innerHTML = '';

  Object.entries(predictions).forEach(([model, prediction]) => {
    const modelInfo = appState.models[model] || { mae: 0, r2: 0 };
    const tr = document.createElement('tr');
    tr.innerHTML = `
            <td>${model.replace('_', ' ').toUpperCase()}</td>
            <td>${prediction.toFixed(2)}</td>
            <td>${modelInfo.mae.toFixed(2)}</td>
            <td>${modelInfo.r2.toFixed(4)}</td>
        `;
    tbody.appendChild(tr);
  });
}

// ============================================
// ANALYSIS PAGE
// ============================================

function initializeAnalysisPage() {
  updateStatisticsTab();

  setTimeout(() => {
    renderCorrelationAnalysis();
    populateRawDataFilters();
    renderRawDataTable();
  }, 100);
}

function updateStatisticsTab() {
  const s = appState.stats;
  const delays = appState.data.map(d => d.delay_minutes || 0);

  document.getElementById('statCount').textContent = delays.length;
  document.getElementById('statMean').textContent = (s.meanDelay || 0).toFixed(2);
  document.getElementById('statStd').textContent = (s.stdDelay || 0).toFixed(2);
  document.getElementById('statMin').textContent = (s.minDelay || 0).toFixed(2);
  document.getElementById('statMax').textContent = (s.maxDelay || 0).toFixed(2);

  const passengers = appState.data.map(d => d.passenger_count || 0);
  document.getElementById('passengerCount').textContent = passengers.length;
  document.getElementById('passengerMean').textContent = (s.passengerMean || 0).toFixed(2);
  document.getElementById('passengerStd').textContent = (s.passengerStd || 0).toFixed(2);
  document.getElementById('passengerMin').textContent = (s.passengerMin || 0).toFixed(2);
  document.getElementById('passengerMax').textContent = (s.passengerMax || 0).toFixed(2);
}

function renderCorrelationAnalysis() {
  const fields = ['delay_minutes', 'passenger_count', 'latitude', 'longitude'];
  const correlation = computeCorrelationMatrix(fields);

  const z = correlation.map(row => row.map(val => val.toFixed(2)));

  const trace = {
    z: correlation,
    x: fields,
    y: fields,
    type: 'heatmap',
    colorscale: 'RdBu',
    zmid: 0
  };

  const layout = {
    title: 'Correlation Matrix of Numeric Features',
    xaxis: { side: 'bottom' },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    height: 500
  };

  Plotly.newPlot('correlationHeatmap', [trace], layout, { displayModeBar: false, responsive: true });
}

function computeCorrelationMatrix(fields) {
  const matrix = [];
  const values = {};

  fields.forEach(field => {
    values[field] = appState.data.map(d => parseFloat(d[field]) || 0);
  });

  const mean = (arr) => arr.reduce((a, b) => a + b) / arr.length;
  const cov = (a, b) => {
    const ma = mean(a);
    const mb = mean(b);
    return a.reduce((sum, ai, i) => sum + (ai - ma) * (b[i] - mb), 0) / a.length;
  };
  const std = (arr) => Math.sqrt(arr.reduce((sum, x) => sum + Math.pow(x - mean(arr), 2), 0) / arr.length);

  fields.forEach(field1 => {
    const row = [];
    fields.forEach(field2 => {
      if (field1 === field2) {
        row.push(1);
      } else {
        const correlation = cov(values[field1], values[field2]) / (std(values[field1]) * std(values[field2]));
        row.push(correlation);
      }
    });
    matrix.push(row);
  });

  return matrix;
}

function populateRawDataFilters() {
  const routes = [...new Set(appState.data.map(d => d.route_id))].sort();
  const weather = [...new Set(appState.data.map(d => d.weather))];

  const routeFilter = document.getElementById('rawRouteFilter');
  const weatherFilter = document.getElementById('rawWeatherFilter');

  routes.forEach(r => {
    const opt = document.createElement('option');
    opt.value = r;
    opt.textContent = r;
    routeFilter.appendChild(opt);
  });

  weather.forEach(w => {
    const opt = document.createElement('option');
    opt.value = w;
    opt.textContent = w;
    weatherFilter.appendChild(opt);
  });
}

function renderRawDataTable() {
  renderFilteredRawData(appState.data);
}

function applyDataFilters() {
  let filtered = appState.data;

  const selectedRoutes = Array.from(document.getElementById('rawRouteFilter').selectedOptions).map(o => o.value);
  const selectedWeather = Array.from(document.getElementById('rawWeatherFilter').selectedOptions).map(o => o.value);
  const maxDelay = parseFloat(document.getElementById('maxDelayFilter').value);

  if (selectedRoutes.length > 0) {
    filtered = filtered.filter(d => selectedRoutes.includes(d.route_id));
  }

  if (selectedWeather.length > 0) {
    filtered = filtered.filter(d => selectedWeather.includes(d.weather));
  }

  filtered = filtered.filter(d => (d.delay_minutes || 0) <= maxDelay);

  renderFilteredRawData(filtered);
}

function renderFilteredRawData(data) {
  const tbody = document.getElementById('rawDataBody');
  tbody.innerHTML = '';

  data.slice(0, 100).forEach(row => {
    const tr = document.createElement('tr');
    const delayMinutes = parseFloat(row.delay_minutes) || 0;
    tr.innerHTML = `
            <td>${row.route_id || '-'}</td>
            <td>${formatDate(row.scheduled_time)}</td>
            <td>${formatDate(row.actual_time)}</td>
            <td>${delayMinutes.toFixed(2)}</td>
            <td>${row.weather || '-'}</td>
            <td>${row.passenger_count || 0}</td>
        `;
    tbody.appendChild(tr);
  });
}

// ============================================
// PERFORMANCE PAGE
// ============================================

function initializePerformancePage() {
  updateModelComparison();

  setTimeout(() => {
    renderMAEChart();
    renderR2Chart();
    updateBestModel();
  }, 100);
}

function updateModelComparison() {
  const tbody = document.getElementById('modelComparisonSummaryBody');
  tbody.innerHTML = '';

  Object.entries(appState.models).forEach(([model, data]) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
            <td>${model.replace('_', ' ').toUpperCase()}</td>
            <td>${data.mae.toFixed(2)}</td>
            <td>${data.r2.toFixed(4)}</td>
        `;
    tbody.appendChild(tr);
  });
}

function renderMAEChart() {
  const models = Object.keys(appState.models);
  const mae = Object.values(appState.models).map(m => m.mae);

  const trace = {
    x: models.map(m => m.replace('_', ' ').toUpperCase()),
    y: mae,
    type: 'bar',
    marker: { color: ['#4682B4', '#FF7F50', '#90EE90', '#9370DB'] }
  };

  const layout = {
    title: 'MAE Comparison (Lower is Better)',
    xaxis: { title: 'Model' },
    yaxis: { title: 'MAE (minutes)' },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    height: 400,
  };

  Plotly.newPlot('maeChart', [trace], layout, { displayModeBar: false, responsive: true });
}

function renderR2Chart() {
  const models = Object.keys(appState.models);
  const r2 = Object.values(appState.models).map(m => m.r2);

  const trace = {
    x: models.map(m => m.replace('_', ' ').toUpperCase()),
    y: r2,
    type: 'bar',
    marker: { color: ['#4682B4', '#FF7F50', '#90EE90', '#9370DB'] }
  };

  const layout = {
    title: 'R² Score Comparison (Higher is Better)',
    xaxis: { title: 'Model' },
    yaxis: { title: 'R² Score' },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    height: 400
  };

  Plotly.newPlot('r2Chart', [trace], layout, { displayModeBar: false, responsive: true });
}

function updateBestModel() {
  let bestModel = 'xgboost';
  let bestMAE = Infinity;

  Object.entries(appState.models).forEach(([model, data]) => {
    if (data.mae < bestMAE) {
      bestMAE = data.mae;
      bestModel = model;
    }
  });

  const bestData = appState.models[bestModel];
  const html = `
        <div style="text-align: center;">
            <p style="font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem;">
                ${bestModel.replace('_', ' ').toUpperCase()}
            </p>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                <div class="best-model-stat">
                    <div class="best-model-stat-label">Test MAE</div>
                    <div class="best-model-stat-value">${bestData.mae.toFixed(2)}</div>
                    <div style="font-size: 0.8rem; margin-top: 0.5rem;">minutes</div>
                </div>
                <div class="best-model-stat">
                    <div class="best-model-stat-label">Test R²</div>
                    <div class="best-model-stat-value">${bestData.r2.toFixed(4)}</div>
                </div>
            </div>
        </div>
    `;

  document.getElementById('bestModelInfo').innerHTML = html;
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

function formatDate(dateString) {
  if (!dateString) return '-';
  try {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch {
    return dateString;
  }
}

function downloadData() {
  let csv = 'route_id,scheduled_time,actual_time,delay_minutes,weather,passenger_count\n';

  appState.data.forEach(row => {
    csv += `"${row.route_id}","${row.scheduled_time}","${row.actual_time}",${row.delay_minutes},"${row.weather}",${row.passenger_count}\n`;
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'transport_delay_data.csv';
  a.click();
}

// Update max delay filter label
document.addEventListener('DOMContentLoaded', () => {
  const maxDelayFilter = document.getElementById('maxDelayFilter');
  if (maxDelayFilter) {
    maxDelayFilter.addEventListener('input', () => {
      document.getElementById('maxDelayValue').textContent = maxDelayFilter.value;
    });
  }
});
