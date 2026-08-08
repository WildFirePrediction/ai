/*
 * Wildfire spread prediction - interactive web demo
 *
 * Flow:
 *   click map      -> ignition point (validated against data coverage)
 *   run prediction -> one POST /api/predict per point, rendered as it arrives
 *   timeline       -> shows cumulative burned cells up to the selected timestep
 */
(function () {
  'use strict';

  // ------------------------------------------------------------------
  // Constants
  // ------------------------------------------------------------------
  var KOREA_CENTER = [36.4, 127.9];
  var KOREA_BOUNDS = L.latLngBounds([32.8, 124.2], [39.2, 132.2]);
  var STEP_MINUTES = 10;
  var METERS_PER_DEG_LAT = 111320;

  // Heat ramp: early steps yellow, later steps deep red
  var RAMP = [
    '#ffe066', '#ffd43b', '#ffc078', '#ffa94d', '#ff922b', '#ff7b28',
    '#ff6b35', '#fa5252', '#f03e3e', '#e03131', '#c92a2a', '#a51111'
  ];

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------
  var state = {
    points: [],          // {id, lat, lon, marker}
    results: [],         // prediction payloads (with .ok / .error)
    layers: [],          // {pointId, stepLayers: [LayerGroup], gridLayer, originLayer}
    nextPointId: 1,
    maxSteps: 3,
    currentStep: 3,
    playing: false,
    playTimer: null,
    running: false,
    health: null
  };

  // ------------------------------------------------------------------
  // Elements
  // ------------------------------------------------------------------
  function $(id) { return document.getElementById(id); }

  var el = {
    statusBox: $('engine-status'),
    statusText: $('engine-status-text'),
    metaModel: $('meta-model'),
    metaDevice: $('meta-device'),
    metaGrid: $('meta-grid'),
    metaStep: $('meta-step'),
    pointList: $('point-list'),
    pointCount: $('point-count'),
    presetSelect: $('preset-select'),
    ignitionTime: $('ignition-time'),
    btnNow: $('btn-now'),
    horizon: $('horizon'),
    horizonLabel: $('horizon-label'),
    showGrid: $('show-grid'),
    autoFit: $('auto-fit'),
    btnRun: $('btn-run'),
    btnClearResults: $('btn-clear-results'),
    btnClearPoints: $('btn-clear-points'),
    btnExport: $('btn-export'),
    progress: $('progress'),
    progressFill: $('progress-fill'),
    progressText: $('progress-text'),
    summary: $('summary'),
    statFires: $('stat-fires'),
    statCells: $('stat-cells'),
    statArea: $('stat-area'),
    statRuntime: $('stat-runtime'),
    resultList: $('result-list'),
    resultCount: $('result-count'),
    legendSteps: $('legend-steps'),
    timeline: $('timeline'),
    btnPlay: $('btn-play'),
    playIcon: $('play-icon'),
    timeSlider: $('time-slider'),
    timeLabel: $('time-label'),
    overlay: $('overlay'),
    overlayText: $('overlay-text'),
    toasts: $('toasts')
  };

  // ------------------------------------------------------------------
  // Map
  // ------------------------------------------------------------------
  var map = L.map('map', {
    center: KOREA_CENTER,
    zoom: 7,
    minZoom: 5,
    maxZoom: 17,
    maxBounds: KOREA_BOUNDS.pad(0.35),
    zoomControl: false,
    attributionControl: true
  });

  L.control.zoom({ position: 'topleft' }).addTo(map);
  L.control.scale({ imperial: false, position: 'bottomleft' }).addTo(map);

  var baseLayers = {
    'Dark': L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
      { maxZoom: 19, attribution: 'CARTO / OpenStreetMap contributors' }
    ),
    'Terrain': L.tileLayer(
      'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
      { maxZoom: 17, attribution: 'OpenTopoMap / OpenStreetMap contributors' }
    ),
    'Satellite': L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 18, attribution: 'Esri World Imagery' }
    ),
    'Street': L.tileLayer(
      'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      { maxZoom: 19, attribution: 'OpenStreetMap contributors' }
    )
  };

  baseLayers.Dark.addTo(map);
  L.control.layers(baseLayers, {}, { position: 'topleft', collapsed: true }).addTo(map);

  // Frame the whole peninsula regardless of window size
  map.fitBounds(L.latLngBounds([33.0, 125.6], [38.7, 130.0]), { padding: [20, 20] });

  var markerLayer = L.layerGroup().addTo(map);
  var resultLayer = L.layerGroup().addTo(map);

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  function toast(message, kind) {
    var node = document.createElement('div');
    node.className = 'toast' + (kind ? ' toast-' + kind : '');
    node.textContent = message;
    el.toasts.appendChild(node);
    setTimeout(function () {
      node.style.opacity = '0';
      node.style.transition = 'opacity 0.3s';
      setTimeout(function () { node.remove(); }, 320);
    }, kind === 'err' ? 5200 : 3200);
  }

  function stepColor(index) { return RAMP[Math.min(index, RAMP.length - 1)]; }

  function stepLabel(step) {
    var minutes = step * STEP_MINUTES;
    if (minutes === 0) { return 'ignition'; }
    if (minutes < 60) { return 't+' + minutes + ' min'; }
    var hours = Math.floor(minutes / 60);
    var rest = minutes % 60;
    return 't+' + hours + 'h' + (rest ? ' ' + rest + 'm' : '');
  }

  function fmt(value, digits, fallback) {
    if (value === null || value === undefined || isNaN(value)) {
      return fallback === undefined ? '-' : fallback;
    }
    return Number(value).toFixed(digits === undefined ? 1 : digits);
  }

  function localIsoNow(date) {
    var d = date || new Date();
    var pad = function (n) { return String(n).padStart(2, '0'); };
    return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) +
      'T' + pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
  }

  function api(path, options) {
    return fetch(path, options).then(function (response) {
      return response.json().catch(function () {
        throw new Error('Server returned a non-JSON response (HTTP ' + response.status + ')');
      }).then(function (body) {
        if (!response.ok) {
          throw new Error(body.error || body.message || ('HTTP ' + response.status));
        }
        return body;
      });
    });
  }

  // 400 m cell centre -> square polygon bounds (local metric approximation)
  function cellBounds(lat, lon, cellSize) {
    var half = cellSize / 2;
    var dLat = half / METERS_PER_DEG_LAT;
    var dLon = half / (METERS_PER_DEG_LAT * Math.cos(lat * Math.PI / 180));
    return [[lat - dLat, lon - dLon], [lat + dLat, lon + dLon]];
  }

  // ------------------------------------------------------------------
  // Ignition points
  // ------------------------------------------------------------------
  function addPoint(lat, lon) {
    if (state.running) { return; }

    lat = Math.round(lat * 1e6) / 1e6;
    lon = Math.round(lon * 1e6) / 1e6;

    api('/api/coverage', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat: lat, lon: lon })
    }).then(function (body) {
      if (!body.ok) {
        toast(body.message + ' (' + fmt(lat, 4) + ', ' + fmt(lon, 4) + ')', 'err');
        return;
      }
      commitPoint(lat, lon);
    }).catch(function (error) {
      toast('Coverage check failed: ' + error.message, 'err');
    });
  }

  function commitPoint(lat, lon) {
    var point = { id: state.nextPointId++, lat: lat, lon: lon, marker: null };

    var marker = L.marker([lat, lon], {
      draggable: true,
      icon: makeIcon(state.points.length + 1),
      title: 'Ignition point ' + (state.points.length + 1)
    });

    marker.on('click', function () { removePoint(point.id); });
    marker.on('dragend', function () {
      var pos = marker.getLatLng();
      point.lat = Math.round(pos.lat * 1e6) / 1e6;
      point.lon = Math.round(pos.lng * 1e6) / 1e6;
      renderPointList();
    });

    point.marker = marker;
    markerLayer.addLayer(marker);
    state.points.push(point);
    renderPointList();
  }

  function makeIcon(number) {
    return L.divIcon({
      className: '',
      html: '<div class="ignition-marker">' + number + '</div>',
      iconSize: [26, 26],
      iconAnchor: [13, 13]
    });
  }

  function removePoint(id) {
    var index = state.points.findIndex(function (p) { return p.id === id; });
    if (index < 0) { return; }
    markerLayer.removeLayer(state.points[index].marker);
    state.points.splice(index, 1);
    // Renumber remaining markers
    state.points.forEach(function (p, i) { p.marker.setIcon(makeIcon(i + 1)); });
    renderPointList();
  }

  function clearPoints() {
    markerLayer.clearLayers();
    state.points = [];
    renderPointList();
  }

  function renderPointList() {
    el.pointCount.textContent = String(state.points.length);
    el.btnRun.disabled = state.running || state.points.length === 0 || !state.health;

    if (state.points.length === 0) {
      el.pointList.innerHTML = '<li class="empty-note">No ignition points yet</li>';
      return;
    }

    el.pointList.innerHTML = '';
    state.points.forEach(function (point, index) {
      var item = document.createElement('li');
      item.className = 'point-item';
      item.innerHTML =
        '<span class="point-index">' + (index + 1) + '</span>' +
        '<span class="point-coords">' + point.lat.toFixed(4) + ', ' + point.lon.toFixed(4) + '</span>' +
        '<button class="point-remove" title="Remove">x</button>';

      item.addEventListener('mouseenter', function () {
        item.classList.add('active');
        point.marker.setZIndexOffset(1000);
      });
      item.addEventListener('mouseleave', function () {
        item.classList.remove('active');
        point.marker.setZIndexOffset(0);
      });
      item.addEventListener('click', function (event) {
        if (event.target.classList.contains('point-remove')) {
          removePoint(point.id);
        } else {
          map.setView([point.lat, point.lon], Math.max(map.getZoom(), 12));
        }
      });

      el.pointList.appendChild(item);
    });
  }

  // ------------------------------------------------------------------
  // Prediction run
  // ------------------------------------------------------------------
  function runPrediction() {
    if (state.running || state.points.length === 0) { return; }

    clearResults();
    state.running = true;
    setBusy(true);

    var horizon = parseInt(el.horizon.value, 10);
    var timestamp = el.ignitionTime.value || localIsoNow();
    var points = state.points.slice();
    var completed = 0;

    // Show every step of the new horizon while results stream in
    state.maxSteps = horizon;
    state.currentStep = horizon;
    buildLegend(horizon);

    el.progress.classList.remove('hidden');
    updateProgress(0, points.length);

    // Sequential requests: the GPU rollout is serialised server side anyway and
    // this gives honest per-fire progress feedback during the demo.
    var chain = Promise.resolve();
    points.forEach(function (point, index) {
      chain = chain.then(function () {
        el.overlayText.textContent = 'Running inference ' + (index + 1) + ' of ' +
          points.length + '...';
        return api('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            lat: point.lat,
            lon: point.lon,
            timestamp: timestamp,
            timesteps: horizon
          })
        }).then(function (result) {
          result.ok = true;
          result.point_index = index + 1;
          state.results.push(result);
          renderResult(result, index);
        }).catch(function (error) {
          state.results.push({
            ok: false,
            point_index: index + 1,
            fire_location: { lat: point.lat, lon: point.lon },
            error: error.message
          });
          renderResultError(index + 1, point, error.message);
          toast('Fire ' + (index + 1) + ' failed: ' + error.message, 'err');
        }).then(function () {
          completed += 1;
          updateProgress(completed, points.length);
        });
      });
    });

    chain.then(function () {
      state.running = false;
      setBusy(false);
      el.progress.classList.add('hidden');
      finishRun(horizon);
    });
  }

  function updateProgress(done, total) {
    el.progressFill.style.width = (total ? (done / total) * 100 : 0) + '%';
    el.progressText.textContent = 'Predicted ' + done + ' of ' + total + ' ignition points';
  }

  function setBusy(busy) {
    el.overlay.classList.toggle('hidden', !busy);
    el.btnRun.disabled = busy || state.points.length === 0 || !state.health;
    el.btnClearPoints.disabled = busy;
    el.btnClearResults.disabled = busy;
  }

  function finishRun(horizon) {
    var successes = state.results.filter(function (r) { return r.ok; });
    el.btnExport.disabled = successes.length === 0;

    if (successes.length === 0) {
      toast('No predictions produced', 'err');
      return;
    }

    setupTimeline(horizon);
    updateSummary();

    if (el.autoFit.checked) {
      var bounds = L.latLngBounds([]);
      state.layers.forEach(function (entry) {
        if (entry.bounds) { bounds.extend(entry.bounds); }
      });
      if (bounds.isValid()) { map.fitBounds(bounds, { padding: [60, 60], maxZoom: 14 }); }
    }

    toast(successes.length + ' prediction(s) completed', 'ok');
  }

  // ------------------------------------------------------------------
  // Rendering
  // ------------------------------------------------------------------
  function renderResult(result, index) {
    drawResultOnMap(result, index);
    appendResultCard(result, index);
    updateSummary();
  }

  function drawResultOnMap(result, index) {
    var cellSize = result.grid.cell_size_m;
    var origin = result.fire_location;
    var entry = {
      pointId: result.fire_id,
      stepLayers: [],
      gridLayer: null,
      originLayer: L.layerGroup(),
      bounds: L.latLngBounds([])
    };

    // Model domain outline
    var b = result.grid.bounds;
    entry.gridLayer = L.rectangle([[b.south, b.west], [b.north, b.east]], {
      color: '#4a5b70',
      weight: 1,
      dashArray: '5 5',
      fill: false,
      interactive: false
    });
    entry.bounds.extend([[b.south, b.west], [b.north, b.east]]);

    // Ignition cell
    var originCell = L.rectangle(cellBounds(origin.lat, origin.lon, cellSize), {
      color: '#fff3e0',
      weight: 1.5,
      fillColor: '#ff3b30',
      fillOpacity: 0.85
    }).bindPopup(
      '<b>Ignition ' + (index + 1) + '</b><br>' +
      'Fire ID: ' + result.fire_id + '<br>' +
      origin.lat.toFixed(5) + ', ' + origin.lon.toFixed(5) + '<br>' +
      'Start: ' + result.fire_timestamp.replace('T', ' ')
    );
    entry.originLayer.addLayer(originCell);

    // Predicted cells per timestep
    result.predictions.forEach(function (step, stepIndex) {
      var group = L.layerGroup();
      var color = stepColor(stepIndex);

      step.predicted_cells.forEach(function (cell) {
        var rect = L.rectangle(cellBounds(cell.lat, cell.lon, cellSize), {
          color: color,
          weight: 0.6,
          opacity: 0.9,
          fillColor: color,
          fillOpacity: 0.5
        });
        rect.bindPopup(
          '<b>' + stepLabel(step.timestep) + '</b><br>' +
          'Fire ' + (index + 1) + ' (' + result.fire_id + ')<br>' +
          'Cell: ' + cell.lat.toFixed(5) + ', ' + cell.lon.toFixed(5) + '<br>' +
          'Predicted at: ' + step.timestamp.replace('T', ' ')
        );
        group.addLayer(rect);
        entry.bounds.extend(rect.getBounds());
      });

      entry.stepLayers.push(group);
    });

    state.layers.push(entry);
    applyStepVisibility();
  }

  function appendResultCard(result, index) {
    if (el.resultList.querySelector('.empty-note')) { el.resultList.innerHTML = ''; }

    var stats = result.statistics;
    var weather = result.weather || {};
    var terrain = result.terrain || {};

    var card = document.createElement('div');
    card.className = 'result-card';

    var rows = stats.per_timestep.map(function (step) {
      return '<tr><td>' + stepLabel(step.timestep) + '</td><td>' + step.new_cells +
        '</td><td>' + step.cumulative_cells + '</td><td>' +
        (step.area_km2 * 100).toFixed(1) + '</td></tr>';
    }).join('');

    var windText = weather.available && weather.wind_speed_ms !== null
      ? fmt(weather.wind_speed_ms, 1) + ' m/s from ' + fmt(weather.wind_direction_deg, 0) + ' deg'
      : 'unavailable';

    card.innerHTML =
      '<div class="result-head">' +
        '<span class="result-swatch" style="background:' + stepColor(0) + '"></span>' +
        '<span class="result-title">Fire ' + (index + 1) + '</span>' +
        '<span class="result-sub">' + stats.total_area_ha.toFixed(1) + ' ha / ' +
          stats.total_cells + ' cells</span>' +
      '</div>' +
      '<div class="result-body">' +
        '<dl class="kv">' +
          '<dt>Fire ID</dt><dd>' + result.fire_id + '</dd>' +
          '<dt>Origin</dt><dd>' + result.fire_location.lat.toFixed(4) + ', ' +
            result.fire_location.lon.toFixed(4) + '</dd>' +
          '<dt>Ignition</dt><dd>' + result.fire_timestamp.replace('T', ' ').slice(0, 19) + '</dd>' +
          '<dt>Horizon</dt><dd>' + stats.horizon_minutes + ' min</dd>' +
          '<dt>Max spread</dt><dd>' + fmt(stats.max_spread_m, 0) + ' m</dd>' +
          '<dt>Rate of spread</dt><dd>' + fmt(stats.mean_rate_of_spread_m_per_min, 2) + ' m/min</dd>' +
          '<dt>Compute time</dt><dd>' + result.runtime_ms + ' ms</dd>' +
          '<dt>Temperature</dt><dd>' + (weather.available ? fmt(weather.temperature_c, 1) + ' C' : '-') + '</dd>' +
          '<dt>Humidity</dt><dd>' + (weather.available ? fmt(weather.humidity_pct, 0) + ' %' : '-') + '</dd>' +
          '<dt>Wind</dt><dd>' + windText + '</dd>' +
          '<dt>KMA station</dt><dd>' + (weather.available
            ? weather.station_id + ' (' + fmt(weather.station_distance_km, 1) + ' km)' : '-') + '</dd>' +
          '<dt>Slope</dt><dd>' + fmt(terrain.slope_deg_center, 1) + ' deg</dd>' +
          '<dt>NDVI</dt><dd>' + fmt(terrain.ndvi_center, 3) + '</dd>' +
          '<dt>Fuel model</dt><dd>' + (terrain.fuel_model || '-') + '</dd>' +
        '</dl>' +
        '<table class="steps-table">' +
          '<thead><tr><th>Step</th><th>New</th><th>Total</th><th>ha</th></tr></thead>' +
          '<tbody>' + rows + '</tbody>' +
        '</table>' +
      '</div>';

    card.querySelector('.result-head').addEventListener('click', function () {
      card.classList.toggle('open');
    });
    card.querySelector('.result-head').addEventListener('dblclick', function () {
      map.setView([result.fire_location.lat, result.fire_location.lon], 13);
    });

    el.resultList.appendChild(card);
    el.resultCount.textContent = String(el.resultList.querySelectorAll('.result-card').length);
  }

  function renderResultError(number, point, message) {
    if (el.resultList.querySelector('.empty-note')) { el.resultList.innerHTML = ''; }
    var card = document.createElement('div');
    card.className = 'result-card';
    card.innerHTML =
      '<div class="result-head">' +
        '<span class="result-swatch" style="background:#f85149"></span>' +
        '<span class="result-title">Fire ' + number + '</span>' +
        '<span class="result-sub">failed</span>' +
      '</div>' +
      '<div class="result-error">' + message + '</div>';
    el.resultList.appendChild(card);
  }

  function updateSummary() {
    var successes = state.results.filter(function (r) { return r.ok; });
    if (successes.length === 0) {
      el.summary.classList.add('hidden');
      return;
    }

    var cells = 0, area = 0, runtime = 0;
    successes.forEach(function (r) {
      cells += r.statistics.total_cells;
      area += r.statistics.total_area_ha;
      runtime += r.runtime_ms;
    });

    el.summary.classList.remove('hidden');
    el.statFires.textContent = String(successes.length);
    el.statCells.textContent = String(cells);
    el.statArea.textContent = area.toFixed(0);
    el.statRuntime.textContent = String(runtime);
  }

  function buildLegend(horizon) {
    el.legendSteps.innerHTML = '';
    for (var i = 0; i < horizon; i++) {
      var row = document.createElement('div');
      row.className = 'legend-row';
      row.innerHTML = '<span class="swatch" style="background:' + stepColor(i) + '"></span>' +
        stepLabel(i + 1);
      el.legendSteps.appendChild(row);
    }
  }

  function clearResults() {
    stopPlayback();
    resultLayer.clearLayers();
    state.results = [];
    state.layers = [];
    el.resultList.innerHTML = '<p class="empty-note">Run a prediction to see spread statistics</p>';
    el.resultCount.textContent = '0';
    el.summary.classList.add('hidden');
    el.timeline.classList.add('hidden');
    el.legendSteps.innerHTML = '';
    el.btnExport.disabled = true;
  }

  // ------------------------------------------------------------------
  // Timeline
  // ------------------------------------------------------------------
  function setupTimeline(horizon) {
    el.timeSlider.max = String(horizon);
    el.timeSlider.value = String(horizon);
    state.currentStep = horizon;
    el.timeline.classList.remove('hidden');
    el.timeLabel.textContent = stepLabel(horizon);
    applyStepVisibility();
  }

  function applyStepVisibility() {
    resultLayer.clearLayers();

    state.layers.forEach(function (entry) {
      if (el.showGrid.checked) { resultLayer.addLayer(entry.gridLayer); }
      resultLayer.addLayer(entry.originLayer);
      entry.stepLayers.forEach(function (group, index) {
        if (index < state.currentStep) { resultLayer.addLayer(group); }
      });
    });
  }

  function setStep(step) {
    state.currentStep = step;
    el.timeSlider.value = String(step);
    el.timeLabel.textContent = stepLabel(step);
    applyStepVisibility();
  }

  function togglePlayback() {
    if (state.layers.length === 0) { return; }
    if (state.playing) { stopPlayback(); return; }

    state.playing = true;
    el.playIcon.className = 'icon-pause';
    if (state.currentStep >= state.maxSteps) { setStep(0); }

    state.playTimer = setInterval(function () {
      if (state.currentStep >= state.maxSteps) { stopPlayback(); return; }
      setStep(state.currentStep + 1);
    }, 750);
  }

  function stopPlayback() {
    state.playing = false;
    el.playIcon.className = 'icon-play';
    if (state.playTimer) { clearInterval(state.playTimer); state.playTimer = null; }
  }

  // ------------------------------------------------------------------
  // Export
  // ------------------------------------------------------------------
  function exportJson() {
    var successes = state.results.filter(function (r) { return r.ok; });
    if (successes.length === 0) { return; }

    var payload = {
      exported_at: new Date().toISOString(),
      source: 'wildfire prediction web demo',
      engine: state.health,
      scenario: {
        ignition_time: el.ignitionTime.value,
        timesteps: parseInt(el.horizon.value, 10),
        points: state.points.map(function (p) { return { lat: p.lat, lon: p.lon }; })
      },
      predictions: successes
    };

    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var link = document.createElement('a');
    link.href = url;
    link.download = 'wildfire_demo_' + localIsoNow().replace(/[-:T]/g, '').slice(0, 15) + '.json';
    link.click();
    URL.revokeObjectURL(url);
    toast('Scenario exported', 'ok');
  }

  // ------------------------------------------------------------------
  // Engine health + presets
  // ------------------------------------------------------------------
  function pollHealth() {
    api('/api/health').then(function (body) {
      state.health = body;
      el.statusBox.className = 'status status-ok';
      el.statusText.textContent = 'Engine ready';
      el.metaModel.textContent = body.model;
      el.metaModel.title = body.checkpoint;
      el.metaDevice.textContent = body.device;
      el.metaGrid.textContent = body.grid_size + 'x' + body.grid_size +
        ' @ ' + body.cell_size_m + 'm';
      el.metaStep.textContent = body.timestep_minutes + ' min';
      el.horizon.max = String(body.max_timesteps || 12);
      renderPointList();
    }).catch(function () {
      state.health = null;
      el.statusBox.className = 'status status-err';
      el.statusText.textContent = 'Engine unavailable';
      renderPointList();
    });
  }

  function loadPresets() {
    api('/api/presets').then(function (body) {
      body.presets.forEach(function (preset) {
        var option = document.createElement('option');
        option.value = preset.lat + ',' + preset.lon;
        option.textContent = preset.name;
        el.presetSelect.appendChild(option);
      });
    }).catch(function () { /* presets are optional */ });
  }

  // ------------------------------------------------------------------
  // Wiring
  // ------------------------------------------------------------------
  map.on('click', function (event) { addPoint(event.latlng.lat, event.latlng.lng); });

  el.presetSelect.addEventListener('change', function () {
    if (!this.value) { return; }
    var parts = this.value.split(',');
    var lat = parseFloat(parts[0]);
    var lon = parseFloat(parts[1]);
    addPoint(lat, lon);
    map.setView([lat, lon], 12);
    this.value = '';
  });

  el.btnNow.addEventListener('click', function () {
    el.ignitionTime.value = localIsoNow();
  });

  el.horizon.addEventListener('input', function () {
    el.horizonLabel.textContent = stepLabel(parseInt(this.value, 10));
  });

  el.showGrid.addEventListener('change', applyStepVisibility);
  el.btnRun.addEventListener('click', runPrediction);
  el.btnClearPoints.addEventListener('click', clearPoints);
  el.btnClearResults.addEventListener('click', clearResults);
  el.btnExport.addEventListener('click', exportJson);
  el.btnPlay.addEventListener('click', togglePlayback);

  el.timeSlider.addEventListener('input', function () {
    stopPlayback();
    setStep(parseInt(this.value, 10));
  });

  document.addEventListener('keydown', function (event) {
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') { return; }
    if (event.code === 'Space') { event.preventDefault(); togglePlayback(); }
    if (event.key === 'Enter' && !state.running) { runPrediction(); }
    if (event.key === 'c' || event.key === 'C') { clearPoints(); }
    if (event.key === 'r' || event.key === 'R') { clearResults(); }
    if (event.key === 'ArrowRight' && state.currentStep < state.maxSteps) {
      stopPlayback(); setStep(state.currentStep + 1);
    }
    if (event.key === 'ArrowLeft' && state.currentStep > 0) {
      stopPlayback(); setStep(state.currentStep - 1);
    }
  });

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------
  el.ignitionTime.value = localIsoNow();
  el.horizonLabel.textContent = stepLabel(parseInt(el.horizon.value, 10));
  renderPointList();
  loadPresets();
  pollHealth();
  setInterval(pollHealth, 15000);
})();
