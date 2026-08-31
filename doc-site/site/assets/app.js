(function () {
  "use strict";
  var data = null;
  var results = [];

  function qs(sel) { return document.querySelector(sel); }

  // --- theme toggle (persisted in localStorage; default dark) -------------
  var themeBtn = qs("#theme-toggle");
  if (themeBtn) {
    function applyTheme(t) {
      document.documentElement.setAttribute("data-theme", t);
      themeBtn.textContent = t === "light" ? "☀" : "☾";
      try { localStorage.setItem("doc-theme", t); } catch (e) {}
    }
    themeBtn.addEventListener("click", function () {
      var cur = document.documentElement.getAttribute("data-theme");
      applyTheme(cur === "light" ? "dark" : "light");
    });
    applyTheme(document.documentElement.getAttribute("data-theme") || "dark");
  }

  // --- collapsible TOC groups --------------------------------------------
  document.querySelectorAll(".toc-head").forEach(function (head) {
    var caret = head.querySelector(".toc-caret");
    if (!caret) return;
    caret.addEventListener("click", function (e) {
      var li = head.parentNode;
      var subs = li && li.querySelector(".toc-subs");
      if (subs) {
        var open = subs.classList.toggle("open");
        caret.classList.toggle("open", open);
      }
      e.preventDefault();
    });
  });

  // --- truncated-title preview -------------------------------------------
  var pv = document.createElement("div");
  pv.className = "toc-preview";
  document.body.appendChild(pv);
  var isTrunc = function (el) { return el.scrollWidth > el.clientWidth + 1; };
  function showPreview(a) {
    pv.textContent = a.textContent.replace(/\s+/g, " ").trim();
    pv.classList.add("show");
    var r = a.getBoundingClientRect();
    var pw = pv.offsetWidth, ph = pv.offsetHeight;
    var left = Math.max(8, r.right + 10);
    if (left + pw > window.innerWidth - 8) left = Math.max(8, r.left - pw - 10);
    pv.style.left = left + "px";
    pv.style.top = Math.max(8, Math.min(r.top, window.innerHeight - ph - 8)) + "px";
  }
  function hidePreview() { pv.classList.remove("show"); }
  document.querySelectorAll(".toc a").forEach(function (a) {
    a.addEventListener("mouseenter", function () { if (isTrunc(a)) showPreview(a); });
    a.addEventListener("mouseleave", hidePreview);
    a.addEventListener("focus", function () { if (isTrunc(a)) showPreview(a); });
    a.addEventListener("blur", hidePreview);
  });
  var sbEl = qs(".sidebar");
  if (sbEl) sbEl.addEventListener("scroll", hidePreview);
  window.addEventListener("resize", hidePreview);

  // sidebar toggle (mobile)
  var toggle = qs("#nav-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var sb = qs(".sidebar");
      if (sb) sb.classList.toggle("open");
    });
  }

  // --- search -------------------------------------------------------------
  var input = qs("#search");
  if (input) {
    var box = document.createElement("div");
    box.id = "search-results";
    var host = qs("#search-host") || document.body;
    host.insertBefore(box, host.firstChild);

    fetch("data.json").then(function (r) { return r.json(); }).then(function (d) {
      data = d;
      d.sections.forEach(function (s) {
        var hay = (s.tabTitle + " " + s.title + " " +
          (s.subs || []).map(function (x) { return x.title; }).join(" ") + " " + s.text).toLowerCase();
        s._hay = hay;
      });
    }).catch(function () {});

    function render(list) {
      box.innerHTML = "";
      list.forEach(function (s) {
        var a = document.createElement("a");
        a.className = "sr";
        a.href = s.slug + ".html";
        var t = document.createElement("span");
        t.className = "t";
        t.textContent = s.title;
        var tab = document.createElement("span");
        tab.className = "tab";
        tab.textContent = s.tabTitle;
        var sn = document.createElement("span");
        sn.className = "sn";
        sn.textContent = (s.text || "").slice(0, 160);
        a.appendChild(t); a.appendChild(tab); a.appendChild(sn);
        box.appendChild(a);
      });
    }

    function search() {
      var val = input.value.trim().toLowerCase();
      if (!val || !data) { box.innerHTML = ""; return; }
      var tokens = val.split(/\s+/);
      var out = [];
      data.sections.forEach(function (s) {
        if (tokens.every(function (tk) { return s._hay.indexOf(tk) !== -1; })) {
          var title = s.title.toLowerCase();
          var score = 0;
          if (title.indexOf(val) === 0) score -= 200;
          else if (title.indexOf(val) !== -1) score -= 100;
          score += s._hay.indexOf(tokens[0]);
          out.push({ s: s, score: score });
        }
      });
      out.sort(function (a, b) { return a.score - b.score; });
      render(out.slice(0, 25).map(function (o) { return o.s; }));
    }

    input.addEventListener("input", search);
    document.addEventListener("keydown", function (e) {
      if (e.key === "/" && document.activeElement !== input) {
        e.preventDefault(); input.focus();
      }
      if (e.key === "Escape") { input.value = ""; search(); input.blur(); }
    });
  }
})();
