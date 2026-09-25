/* Draws the plotly figures that dashboard/build.py wrote into #dash-data,
   swaps them when a select changes, re-colours them for dark mode, and makes
   the tables sortable. If plotly.js didn't load, the static PNGs show instead
   (see .has-plotly in style.css). */
(function () {
  "use strict";

  var DATA = JSON.parse(document.getElementById("dash-data").textContent);
  var hasPlotly = typeof window.Plotly !== "undefined";
  var root = document.documentElement;
  var media = window.matchMedia("(prefers-color-scheme: dark)");

  function isDark() {
    var t = root.getAttribute("data-theme");
    return t ? t === "dark" : media.matches;
  }

  // one pass over the figure JSON, light colour string -> dark colour string
  function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
  var darkKeys = Object.keys(DATA.dark).sort(function (a, b) { return b.length - a.length; });
  var darkRe = new RegExp(darkKeys.map(escapeRe).join("|"), "g");

  function narrow() { return window.innerWidth < 640; }

  function themed(fig) {
    var s = JSON.stringify(fig);
    if (isDark()) s = s.replace(darkRe, function (m) { return DATA.dark[m]; });
    var f = JSON.parse(s);
    if (narrow()) {
      f.layout.dragmode = false;  // let a finger scroll the page
      if (f.layout.showlegend !== false && f.layout.height) f.layout.height += 70;
    }
    return f;
  }

  function config(name) {
    return {
      displaylogo: false,
      responsive: true,
      scrollZoom: false,
      displayModeBar: "hover",
      modeBarButtonsToRemove: ["lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "toggleSpikelines"],
      toImageButtonOptions: { format: "png", filename: "rl-dosing-" + name, scale: 2 }
    };
  }

  function groupKey(group) {
    var sels = document.querySelectorAll('select[data-group="' + group + '"]');
    return Array.prototype.map.call(sels, function (s) { return s.value; }).join("|");
  }

  function draw(group) {
    var key = groupKey(group);
    document.querySelectorAll('[data-group="' + group + '"][data-key]').forEach(function (el) {
      el.hidden = el.getAttribute("data-key") !== key;
    });
    if (!hasPlotly) return;
    document.querySelectorAll('.plot[data-group="' + group + '"]').forEach(function (node) {
      var variants = DATA.charts[node.getAttribute("data-chart")] || {};
      var fig = variants[key];
      if (!fig) return;
      var f = themed(fig);
      window.Plotly.react(node, f.data, f.layout, config(node.getAttribute("data-chart")));
    });
  }

  var groups = [];
  document.querySelectorAll("[data-group]").forEach(function (el) {
    var g = el.getAttribute("data-group");
    if (groups.indexOf(g) < 0) groups.push(g);
  });
  function drawAll() { groups.forEach(draw); }

  document.querySelectorAll("select[data-group]").forEach(function (sel) {
    sel.addEventListener("change", function () { draw(sel.getAttribute("data-group")); });
  });

  // theme: follows the system unless the button was used
  var btn = document.getElementById("theme");
  function paintButton() {
    var dark = isDark();
    btn.textContent = dark ? "Light mode" : "Dark mode";
    btn.setAttribute("aria-pressed", dark ? "true" : "false");
  }
  btn.addEventListener("click", function () {
    root.setAttribute("data-theme", isDark() ? "light" : "dark");
    paintButton();
    drawAll();
  });
  var onMedia = function () { if (!root.hasAttribute("data-theme")) { paintButton(); drawAll(); } };
  if (media.addEventListener) media.addEventListener("change", onMedia); else media.addListener(onMedia);

  // re-draw when crossing the phone breakpoint (legend room, drag mode)
  var wasNarrow = narrow();
  window.addEventListener("resize", function () {
    if (narrow() !== wasNarrow) { wasNarrow = narrow(); drawAll(); }
  });

  // sortable tables: click a header, click again to reverse
  document.querySelectorAll("table.sortable").forEach(function (table) {
    var heads = table.querySelectorAll("thead th");
    heads.forEach(function (th, i) {
      th.tabIndex = 0;
      function sort() {
        var asc = th.getAttribute("aria-sort") !== "ascending";
        heads.forEach(function (h) { h.removeAttribute("aria-sort"); });
        th.setAttribute("aria-sort", asc ? "ascending" : "descending");
        var body = table.tBodies[0];
        var rows = Array.prototype.slice.call(body.rows);
        rows.sort(function (a, b) {
          var x = a.cells[i].getAttribute("data-sort"), y = b.cells[i].getAttribute("data-sort");
          var nx = parseFloat(x), ny = parseFloat(y);
          var cmp;
          if (x === "" && y === "") cmp = 0;
          else if (x === "") return 1;       // blanks last either way
          else if (y === "") return -1;
          else if (!isNaN(nx) && !isNaN(ny)) cmp = nx - ny;
          else cmp = x.localeCompare(y);
          return asc ? cmp : -cmp;
        });
        rows.forEach(function (r) { body.appendChild(r); });
      }
      th.addEventListener("click", sort);
      th.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sort(); }
      });
    });
  });

  paintButton();
  drawAll();
})();
