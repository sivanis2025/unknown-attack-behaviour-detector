/**
 * Unknown Attack Behaviour Detector
 * Dashboard JavaScript
 */
'use strict';

document.addEventListener('DOMContentLoaded', () => {
  const $ = (id) => document.getElementById(id);

  const loader = $('globalLoader');
  const loaderText = $('loaderText');
  const statusBadge = $('modelStatusBadge');
  const modelInfoContent = $('modelInfoContent');

  const runDemoBtn = $('runDemoBtn');
  const uploadBtn = $('uploadAnalyseBtn');
  const retrainBtn = $('retrainBtn');

  const csvInput = $('csvFile');
  const trainInput = $('trainCsvFile');

  const charts = {};

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    })[char]);
  }

  function formatNumber(value) {
    const number = Number(value);
    return Number.isFinite(number)
      ? number.toLocaleString()
      : '—';
  }

  function showToast(message, type = 'info') {
    const container = $('toastContainer');

    if (!container) {
      console.log(message);
      return;
    }

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.setAttribute('role', 'alert');
    toast.textContent = message;

    container.appendChild(toast);

    setTimeout(() => {
      toast.remove();
    }, 5000);
  }

  function setLoading(show, message = 'Processing…') {
    if (loaderText) {
      loaderText.textContent = message;
    }

    if (loader) {
      loader.hidden = !show;
    }
  }

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);

    let data;

    try {
      data = await response.json();
    } catch {
      throw new Error(
        `Invalid server response. HTTP ${response.status}`
      );
    }

    if (!response.ok) {
      throw new Error(
        data.error || `Request failed. HTTP ${response.status}`
      );
    }

    return data;
  }

  function setButtonBusy(button, busy, label) {
    if (!button) return () => {};

    const originalText = button.textContent;

    if (busy) {
      button.disabled = true;
      button.textContent = label;
    }

    return () => {
      button.disabled = false;
      button.textContent = originalText;
    };
  }

  function destroyChart(id) {
    if (charts[id]) {
      charts[id].destroy();
      delete charts[id];
    }
  }

  function createChart(id, type, labels, datasets, options = {}) {
    const canvas = $(id);

    if (!canvas || typeof window.Chart === 'undefined') {
      return;
    }

    destroyChart(id);

    charts[id] = new Chart(canvas.getContext('2d'), {
      type,
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: type === 'doughnut',
        plugins: {
          legend: {
            display: type === 'doughnut'
          }
        },
        ...options
      }
    });
  }

  function renderTable(records, columns) {
    const head = $('tableHead');
    const body = $('tableBody');
    const table = $('suspiciousTable');
    const empty = $('tableEmpty');

    if (!head || !body || !table || !empty) {
      return;
    }

    if (!Array.isArray(records) || records.length === 0) {
      table.hidden = true;
      empty.hidden = false;
      return;
    }

    table.hidden = false;
    empty.hidden = true;

    const headers = [
      '#',
      'Severity',
      'Anomaly Score',
      ...columns,
      'Explanation (Heuristic)'
    ];

    head.innerHTML = `<tr>${
      headers.map((header) =>
        `<th>${escapeHtml(header)}</th>`
      ).join('')
    }</tr>`;

    body.innerHTML = '';

    records.forEach((record, index) => {
      const row = document.createElement('tr');

      const values = [
        index + 1,
        record.severity || 'Low',
        record.anomaly_score ?? '—',
        ...columns.map((column) => record[column] ?? '—'),
        record.explanation ||
          'Statistical anomaly; expert review required.'
      ];

      values.forEach((value, cellIndex) => {
        const cell = document.createElement('td');

        if (cellIndex === 1) {
          const badge = document.createElement('span');
          badge.className =
            `severity-badge ${record.severity || 'Low'}`;
          badge.textContent = String(value);
          cell.appendChild(badge);
        } else {
          cell.textContent = String(value);
        }

        row.appendChild(cell);
      });

      body.appendChild(row);
    });
  }

  function renderResults(data) {
    if (!data || typeof data !== 'object') {
      throw new Error('The analysis response is empty.');
    }

    if ($('resultsSection')) {
      $('resultsSection').hidden = false;
    }

    if ($('emptyState')) {
      $('emptyState').hidden = true;
    }

    if ($('dataSourceLabel')) {
      $('dataSourceLabel').textContent =
        data.source || 'Unknown';
    }

    if ($('statTotal')) {
      $('statTotal').textContent = formatNumber(data.total);
    }

    if ($('statNormal')) {
      $('statNormal').textContent = formatNumber(data.n_normal);
    }

    if ($('statAnomalies')) {
      $('statAnomalies').textContent =
        formatNumber(data.n_anomaly);
    }

    if ($('statPct')) {
      $('statPct').textContent =
        `${data.anomaly_pct ?? 0}%`;
    }

    const normal = Number(data.n_normal || 0);
    const anomalies = Number(data.n_anomaly || 0);

    createChart(
      'distributionChart',
      'doughnut',
      ['Normal Traffic', 'Anomalous Traffic'],
      [{
        data: [normal, anomalies],
        backgroundColor: [
          'rgba(0,255,135,0.7)',
          'rgba(255,71,87,0.7)'
        ],
        borderWidth: 2
      }],
      { cutout: '70%' }
    );

    if ($('distLegend')) {
      $('distLegend').textContent =
        `Normal: ${normal} | Anomalies: ${anomalies}`;
    }

    if (
      data.score_chart &&
      Array.isArray(data.score_chart.labels) &&
      Array.isArray(data.score_chart.values)
    ) {
      createChart(
        'scoreHistChart',
        'bar',
        data.score_chart.labels,
        [{
          label: 'Record Count',
          data: data.score_chart.values
        }],
        { maintainAspectRatio: false }
      );
    }

    if (
      Array.isArray(data.sample_scores) &&
      data.sample_scores.length
    ) {
      createChart(
        'scoreLineChart',
        'line',
        data.sample_scores.map((_, index) => index + 1),
        [{
          label: 'Anomaly Score',
          data: data.sample_scores,
          borderColor: '#00d4ff',
          pointRadius: 2,
          tension: 0.3
        }],
        { maintainAspectRatio: false }
      );
    }

    renderTable(
      data.suspicious_records || [],
      data.display_cols || []
    );

    $('resultsSection')?.scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    });
  }

  async function loadModelInfo() {
    try {
      const info = await fetchJson('/api/model-info', {
        cache: 'no-store'
      });

      if (statusBadge) {
        statusBadge.textContent = '● MODEL READY';
      }

      if (!modelInfoContent) return;

      let html = `
        <div class="model-info-grid">
          <div class="model-info-item">
            <span class="model-info-label">Model Type</span>
            <span class="model-info-value">
              ${escapeHtml(info.model_type || 'Isolation Forest')}
            </span>
          </div>

          <div class="model-info-item">
            <span class="model-info-label">Training Records</span>
            <span class="model-info-value">
              ${formatNumber(info.n_records)}
            </span>
          </div>

          <div class="model-info-item">
            <span class="model-info-label">Features Used</span>
            <span class="model-info-value">
              ${formatNumber(info.n_features)}
            </span>
          </div>

          <div class="model-info-item">
            <span class="model-info-label">Estimators</span>
            <span class="model-info-value">
              ${formatNumber(info.n_estimators)}
            </span>
          </div>

          <div class="model-info-item">
            <span class="model-info-label">Contamination</span>
            <span class="model-info-value">
              ${escapeHtml(info.contamination)}
            </span>
          </div>
        </div>
      `;

      if (info.has_evaluation) {
        const percentage = (value) =>
          `${(Number(value || 0) * 100).toFixed(1)}%`;

        html += `
          <div class="model-metrics-row">
            <div class="metric-card">
              <div class="metric-name">Precision</div>
              <div class="metric-value">
                ${percentage(info.precision)}
              </div>
            </div>

            <div class="metric-card">
              <div class="metric-name">Recall</div>
              <div class="metric-value">
                ${percentage(info.recall)}
              </div>
            </div>

            <div class="metric-card">
              <div class="metric-name">F1-Score</div>
              <div class="metric-value">
                ${percentage(info.f1_score)}
              </div>
            </div>
          </div>
        `;
      }

      if (Array.isArray(info.confusion_matrix)) {
        html += `
          <p>Confusion Matrix</p>
          <pre>${escapeHtml(
            JSON.stringify(info.confusion_matrix, null, 2)
          )}</pre>
        `;
      }

      html += `
        <p class="model-disclaimer">
          ⚠ ${escapeHtml(
            info.disclaimer ||
            'Metrics are preliminary and require independent testing.'
          )}
        </p>
      `;

      if (
        Array.isArray(info.feature_names) &&
        info.feature_names.length
      ) {
        html += `
          <p>Features used for detection:</p>
          <div class="features-list">
            ${info.feature_names.map((feature) =>
              `<span class="feature-tag">${
                escapeHtml(feature)
              }</span>`
            ).join('')}
          </div>
        `;
      }

      modelInfoContent.innerHTML = html;

    } catch (error) {
      console.error('Model information error:', error);

      if (statusBadge) {
        statusBadge.textContent = 'MODEL INFO ERROR';
      }

      if (modelInfoContent) {
        modelInfoContent.textContent =
          `Unable to load model information: ${error.message}`;
      }
    }
  }

  // Update the CSV filename when a file is selected.
  csvInput?.addEventListener('change', () => {
    const file = csvInput.files[0];

    if ($('fileNameDisplay')) {
      $('fileNameDisplay').textContent =
        file ? file.name : 'No file chosen';
    }

    if (uploadBtn) {
      uploadBtn.disabled = !file;
    }
  });

  // Update the training CSV filename.
  trainInput?.addEventListener('change', () => {
    const file = trainInput.files[0];

    if ($('trainFileNameDisplay')) {
      $('trainFileNameDisplay').textContent =
        file ? file.name : 'No file chosen';
    }

    if (retrainBtn) {
      retrainBtn.disabled = !file;
    }
  });

  // Run the built-in demonstration.
  runDemoBtn?.addEventListener('click', async () => {
    const restore = setButtonBusy(
      runDemoBtn, true, 'Running…'
    );

    setLoading(true, 'Analysing demo dataset…');

    try {
      const data = await fetchJson('/api/demo');
      renderResults(data);

      showToast(
        'Demo analysis completed. The dataset is synthetic.',
        'success'
      );
    } catch (error) {
      console.error(error);
      showToast(`Demo failed: ${error.message}`, 'error');
    } finally {
      restore();
      setLoading(false);
    }
  });

  // Analyse an uploaded CSV.
  uploadBtn?.addEventListener('click', async () => {
    const file = csvInput?.files?.[0];

    if (!file) {
      showToast('Please select a CSV file first.');
      return;
    }

    const restore = setButtonBusy(
      uploadBtn, true, 'Analysing…'
    );

    setLoading(true, `Analysing ${file.name}…`);

    try {
      const form = new FormData();
      form.append('file', file);

      const data = await fetchJson('/api/upload', {
        method: 'POST',
        body: form
      });

      renderResults(data);
      showToast('CSV analysis completed.', 'success');

    } catch (error) {
      console.error(error);
      showToast(`CSV analysis failed: ${error.message}`, 'error');
    } finally {
      restore();
      setLoading(false);
    }
  });

  // Retrain the model with an uploaded CSV.
  retrainBtn?.addEventListener('click', async () => {
    const file = trainInput?.files?.[0];

    if (!file) {
      showToast('Please select a training CSV file.');
      return;
    }

    if (!window.confirm(
      `Retrain the model using "${file.name}"?`
    )) {
      return;
    }

    const restore = setButtonBusy(
      retrainBtn, true, 'Training…'
    );

    setLoading(true, 'Retraining model…');

    try {
      const form = new FormData();
      form.append('file', file);

      const result = await fetchJson('/api/train', {
        method: 'POST',
        body: form
      });

      showToast(
        result.message || 'Retraining completed.',
        'success'
      );

      await loadModelInfo();

    } catch (error) {
      console.error(error);
      showToast(`Retraining failed: ${error.message}`, 'error');
    } finally {
      restore();
      setLoading(false);
    }
  });

  // Optional report download, if the backend supports this route.
  $('downloadReportBtn')?.addEventListener('click', () => {
    window.location.href = '/api/download';
  });

  // Start the dashboard without leaving the loading overlay visible.
  setLoading(false);
  loadModelInfo().finally(() => setLoading(false));
});
