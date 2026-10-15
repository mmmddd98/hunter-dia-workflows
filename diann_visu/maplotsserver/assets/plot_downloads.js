(function () {
  function closestButton(target) {
    if (!target || !target.closest) return null;
    return target.closest("[data-plot-download]");
  }

  function graphDiv(graphId) {
    const wrapper = document.getElementById(graphId);
    if (!wrapper) return null;
    return wrapper.querySelector(".js-plotly-plot");
  }

  function safeFilename(value) {
    return String(value || "maplot_figure")
      .replace(/[^A-Za-z0-9._-]+/g, "_")
      .replace(/^_+|_+$/g, "") || "maplot_figure";
  }

  function graphFilename(gd, fallback) {
    const layout = gd && gd.layout ? gd.layout : {};
    const title = layout.title && (layout.title.text || layout.title);
    return safeFilename(gd && gd.id ? gd.id : title || fallback);
  }

  function exportDimensions(gd, requestedWidth) {
    const layout = gd && gd._fullLayout ? gd._fullLayout : {};
    const renderedWidth = Number(layout.width || (gd && gd.clientWidth) || 1400);
    const renderedHeight = Number(layout.height || (gd && gd.clientHeight) || 900);
    const width = Math.max(1, Number(requestedWidth) || renderedWidth);
    const aspectRatio = renderedWidth > 0 && renderedHeight > 0
      ? renderedWidth / renderedHeight
      : 14 / 9;
    return {
      width: Math.round(width),
      height: Math.max(1, Math.round(width / aspectRatio))
    };
  }

  function downloadPlot(gd, format, filename, requestedWidth, scale) {
    const dimensions = exportDimensions(gd, requestedWidth);
    return window.Plotly.downloadImage(gd, {
      format,
      filename: safeFilename(filename),
      width: dimensions.width,
      height: dimensions.height,
      scale: Number(scale) || 1
    });
  }

  function downloadBlob(text, filename, mimeType) {
    const blob = new Blob([text], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(function () {
      URL.revokeObjectURL(url);
    }, 1000);
  }

  function downloadHtml(gd, filename) {
    const data = gd.data || [];
    const layout = gd.layout || {};
    const config = { responsive: true, displayModeBar: true };
    const plotlyVersion = window.Plotly && window.Plotly.version ? window.Plotly.version : "latest";
    const title = layout.title && (layout.title.text || layout.title) ? layout.title.text || layout.title : filename;
    const html = [
      "<!DOCTYPE html>",
      "<html>",
      "<head>",
      '<meta charset="utf-8">',
      "<title>" + escapeHtml(title) + "</title>",
      '<script src="https://cdn.plot.ly/plotly-' + plotlyVersion + '.min.js"><\/script>',
      '<style>html,body{margin:0;width:100%;height:100%;font-family:Arial,sans-serif;}#plot{width:100vw;height:100vh;}</style>',
      "</head>",
      "<body>",
      '<div id="plot"></div>',
      "<script>",
      "const data = " + JSON.stringify(data) + ";",
      "const layout = " + JSON.stringify(layout) + ";",
      "const config = " + JSON.stringify(config) + ";",
      "Plotly.newPlot('plot', data, layout, config);",
      "<\/script>",
      "</body>",
      "</html>"
    ].join("\n");
    downloadBlob(html, safeFilename(filename) + ".html", "text/html;charset=utf-8");
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  document.addEventListener("click", function (event) {
    const button = closestButton(event.target);
    if (!button) return;
    const graphId = button.dataset.graphId;
    const gd = graphDiv(graphId);
    if (!gd || !window.Plotly) return;
    const format = button.dataset.plotDownload || "svg";
    const filename = button.dataset.filename || graphId || "maplot_figure";
    if (format === "html") {
      downloadHtml(gd, filename);
      return;
    }
    downloadPlot(gd, format, filename, button.dataset.width, button.dataset.scale);
  });

  function modebarButton(label, title, className, onClick) {
    const button = document.createElement("a");
    button.className = "modebar-btn " + className;
    button.setAttribute("rel", "tooltip");
    button.setAttribute("data-title", title);
    button.setAttribute("aria-label", title);
    button.innerHTML = '<span class="maplot-modebar-download-label">' + label + "</span>";
    button.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      onClick();
    });
    return button;
  }

  function addDownloadModebarButtons(gd) {
    if (!gd) return;
    const modebar = gd.querySelector(".modebar-group");
    if (!modebar) return;
    if (!gd.querySelector(".maplot-svg-download")) {
      modebar.appendChild(modebarButton("SVG", "Download SVG", "maplot-svg-download", function () {
        downloadPlot(gd, "svg", graphFilename(gd, "maplot_figure"), 1600, 1);
      }));
    }
    if (!gd.querySelector(".maplot-html-download")) {
      modebar.appendChild(modebarButton("HTML", "Download interactive HTML", "maplot-html-download", function () {
        downloadHtml(gd, graphFilename(gd, "maplot_interactive_plot"));
      }));
    }
  }

  function installDownloadModebarButtons() {
    document.querySelectorAll(".js-plotly-plot").forEach(addDownloadModebarButtons);
  }

  document.addEventListener("plotly_afterplot", function (event) {
    addDownloadModebarButtons(event.target);
  });
  setInterval(installDownloadModebarButtons, 1200);
  installDownloadModebarButtons();
})();
