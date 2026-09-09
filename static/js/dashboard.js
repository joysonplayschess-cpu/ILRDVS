/* Dashboard charts (Chart.js is vendored at /static/vendor/chart.umd.min.js). */
(function () {
  "use strict";
  var el = document.getElementById("chart-data");
  if (!el || typeof Chart === "undefined") return;
  var D = JSON.parse(el.textContent);

  Chart.defaults.font.family =
    "Inter, system-ui, 'Segoe UI', Roboto, sans-serif";
  Chart.defaults.color = "#67728a";
  var PALETTE = ["#2f6bff", "#14b8a6", "#f59e0b", "#8a5cf6", "#ef6a6a",
                 "#22a7f0", "#84cc16", "#f472b6", "#94a3b8"];

  function doughnut(id, d) {
    var c = document.getElementById(id);
    if (!c || !d.data.length) return;
    new Chart(c, { type: "doughnut",
      data: { labels: d.labels,
        datasets: [{ data: d.data, backgroundColor: PALETTE,
                     borderWidth: 2, borderColor: "#fff" }] },
      options: { maintainAspectRatio: false, cutout: "62%",
        plugins: { legend: { position: "right" } } } });
  }

  function bar(id, d, datasets) {
    var c = document.getElementById(id);
    if (!c || !d.labels.length) return;
    new Chart(c, { type: "bar",
      data: { labels: d.labels, datasets: datasets },
      options: { maintainAspectRatio: false,
        plugins: { legend: { position: "top" } },
        scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } } });
  }

  doughnut("ch-docstatus", D.doc_status);
  doughnut("ch-lang", D.languages);

  bar("ch-states", D.states, [
    { label: "Documents processed", data: D.states.processed, backgroundColor: "#2f6bffcc",
      borderRadius: 6 },
    { label: "Records verified", data: D.states.verified, backgroundColor: "#14b8a6cc",
      borderRadius: 6 }
  ]);

  (function () {  // accuracy trend
    var c = document.getElementById("ch-accuracy");
    if (!c || !D.accuracy.data.length) return;
    var ctx = c.getContext("2d");
    var grad = ctx.createLinearGradient(0, 0, 0, 240);
    grad.addColorStop(0, "rgba(47,107,255,.30)");
    grad.addColorStop(1, "rgba(47,107,255,0)");
    new Chart(c, { type: "line",
      data: { labels: D.accuracy.labels,
        datasets: [{ label: "Confidence %", data: D.accuracy.data,
          borderColor: "#2f6bff", backgroundColor: grad, fill: true,
          tension: .35, pointRadius: 2.5 }] },
      options: { maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { min: 0, max: 100,
          title: { display: true, text: "%" } } } } });
  })();

  bar("ch-issues", D.issues, [
    { label: "Open findings", data: D.issues.data, backgroundColor: "#f59e0bcc",
      borderRadius: 6 }
  ]);
  bar("ch-monthly", D.monthly, [
    { label: "Documents uploaded", data: D.monthly.data, backgroundColor: "#8a5cf6cc",
      borderRadius: 6 }
  ]);
})();
