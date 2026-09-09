/* ILRDVS front-end helpers (no external dependencies). */
(function () {
  "use strict";

  /* ---- clickable table rows ------------------------------------------ */
  document.querySelectorAll("tr.rowlink").forEach(function (tr) {
    tr.addEventListener("click", function (e) {
      if (e.target.closest("a,button,input,select")) return;
      window.location = tr.dataset.href;
    });
  });

  /* ---- original / enhanced toggle on document detail ----------------- */
  var bo = document.getElementById("btn-orig"),
      be = document.getElementById("btn-enha"),
      po = document.getElementById("pane-orig"),
      pe = document.getElementById("pane-enha");
  if (bo && be) {
    bo.addEventListener("click", function () {
      po.classList.remove("hidden"); pe.classList.add("hidden");
      bo.classList.add("active"); be.classList.remove("active");
    });
    be.addEventListener("click", function () {
      pe.classList.remove("hidden"); po.classList.add("hidden");
      be.classList.add("active"); bo.classList.remove("active");
    });
  }

  /* ---- verification console: multi-page tab switcher ------------------
     One LandRecord/one review form covers the whole document; this just
     lets the officer flip between page scans instead of scrolling past a
     tall stack of images (which read as "one review per page"). ---- */
  var pageTabs = document.querySelectorAll(".page-tab");
  if (pageTabs.length) {
    var pageBlocks = document.querySelectorAll(".page-block");
    pageTabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        var target = tab.dataset.page;
        pageTabs.forEach(function (t) {
          t.classList.toggle("active", t === tab);
        });
        pageBlocks.forEach(function (b) {
          b.classList.toggle("hidden", b.dataset.page !== target);
        });
      });
    });
  }

  /* ---- upload page: dropzone + gamma slider -------------------------- */
  var dz = document.getElementById("dropzone"),
      fi = document.getElementById("file-input"),
      fl = document.getElementById("file-list");
  function renderFiles(files) {
    if (!fl) return;
    fl.innerHTML = "";
    Array.prototype.forEach.call(files, function (f, i) {
      var li = document.createElement("li");
      li.innerHTML = "<span class='mono'>" + f.name + "</span>" +
        "<span class='muted'>" + (f.size / 1024).toFixed(0) + " KB</span>" +
        "<span class='st' data-i='" + i + "'></span>";
      fl.appendChild(li);
    });
  }
  if (dz && fi) {
    fi.addEventListener("change", function () { renderFiles(fi.files); });
    ["dragover", "dragenter"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add("drag"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove("drag"); });
    });
    dz.addEventListener("drop", function (e) {
      fi.files = e.dataTransfer.files;
      renderFiles(fi.files);
    });
  }

  var gm = document.getElementById("gamma-mode"),
      gw = document.getElementById("gamma-slider-wrap"),
      gs = document.getElementById("gamma-slider"),
      go = document.getElementById("gamma-out");
  if (gm && gw) {
    gm.addEventListener("change", function () {
      gw.classList.toggle("hidden", gm.value !== "manual");
    });
    if (gs && go) gs.addEventListener("input", function () {
      go.textContent = parseFloat(gs.value).toFixed(2);
    });
  }

  /* ---- verification console: “⟲ OCR value” paste chips --------------- */
  document.querySelectorAll("button[data-copy]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var input = btn.closest(".vfield").querySelector("input");
      input.value = btn.dataset.copy;
      input.focus();
    });
  });
})();
