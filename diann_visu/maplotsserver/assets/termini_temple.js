(function () {
  function templeMeta(gd) {
    const layoutMeta = gd && gd.layout && gd.layout.meta;
    const fullMeta = gd && gd._fullLayout && gd._fullLayout.meta;
    const meta = layoutMeta || fullMeta || {};
    if (meta.plot_type !== "termini_temple") return null;
    return meta;
  }

  function hiddenSet(gd) {
    if (!gd.__terminiTempleHidden) gd.__terminiTempleHidden = {};
    return gd.__terminiTempleHidden;
  }

  function isHidden(hidden, key) {
    return !!hidden[key];
  }

  function legendOpacity(hidden, key) {
    return isHidden(hidden, key) ? 0.28 : 1;
  }

  function fadedColor(color, opacity) {
    if (!color || opacity >= 1) return color;
    const hex = color.charAt(0) === "#" ? color.slice(1) : color;
    if (hex.length !== 6) return color;
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return "rgba(" + r + "," + g + "," + b + "," + opacity + ")";
  }

  function makeLegendTrace(name, group, color, hidden, groupTitle) {
    const opacity = legendOpacity(hidden, group);
    const trace = {
      type: "bar",
      x: [null],
      y: [null],
      name: name,
      legendgroup: group,
      showlegend: true,
      hoverinfo: "skip",
      opacity: opacity,
      marker: { color: fadedColor(color, opacity), opacity: opacity },
      meta: { role: "temple_legend" },
    };
    if (groupTitle) trace.legendgrouptitle = { text: groupTitle };
    return trace;
  }

  function shortSpecificityLabel(value) {
    const labels = {
      tryptic: "Tryptic",
      semi_Nterm: "Semi-N",
      semi_Cterm: "Semi-C",
    };
    return labels[value] || value;
  }

  function separatedPositions(desired, minDistance, minCenter) {
    if (desired.length <= 1) return desired.slice();
    const blocks = [];
    desired.forEach(function (value, index) {
      blocks.push({ start: index, end: index, weight: 1, mean: value - index * minDistance });
      while (blocks.length >= 2 && blocks[blocks.length - 2].mean > blocks[blocks.length - 1].mean) {
        const last = blocks.pop();
        const previous = blocks[blocks.length - 1];
        const weight = previous.weight + last.weight;
        previous.mean = (previous.mean * previous.weight + last.mean * last.weight) / weight;
        previous.weight = weight;
        previous.end = last.end;
      }
    });
    const positions = new Array(desired.length);
    blocks.forEach(function (block) {
      for (let index = block.start; index <= block.end; index += 1) {
        positions[index] = block.mean + index * minDistance;
      }
    });
    if (positions[0] < minCenter) {
      const offset = minCenter - positions[0];
      for (let index = 0; index < positions.length; index += 1) positions[index] += offset;
    }
    return positions;
  }

  function boundedSeparatedPositions(desired, minDistance, minCenter, maxCenter) {
    if (desired.length <= 1) {
      if (!desired.length) return [];
      return [Math.min(Math.max(desired[0], minCenter), maxCenter)];
    }
    const available = maxCenter - minCenter;
    const required = (desired.length - 1) * minDistance;
    if (available + 1e-9 < required) return separatedPositions(desired, minDistance, minCenter);

    const positions = separatedPositions(desired, minDistance, minCenter);
    if (positions[positions.length - 1] <= maxCenter + 1e-9) return positions;

    const span = positions[positions.length - 1] - positions[0];
    if (span <= available + 1e-9) {
      const shift = Math.min(positions[positions.length - 1] - maxCenter, positions[0] - minCenter);
      return positions.map(function (position) { return position - shift; });
    }

    const desiredMidpoint = (desired[0] + desired[desired.length - 1]) / 2;
    let start = desiredMidpoint - required / 2;
    start = Math.min(Math.max(start, minCenter), maxCenter - required);
    return desired.map(function (_value, index) { return start + index * minDistance; });
  }

  function rebuild(gd) {
    const meta = templeMeta(gd);
    if (!meta) return;
    const model = meta.temple_model || {};
    const hidden = hiddenSet(gd);
    const classes = model.classes || [];
    const specificityOrder = model.specificity_order || [];
    const modificationOrder = model.modification_order || [];
    const specificityColors = model.specificity_colors || {};
    const modificationColors = model.modification_colors || {};
    const classGap = model.class_gap || 0.92;
    const barGap = model.bar_gap || 0.18;
    const barWidth = model.bar_width || 0.22;
    const masterBarWidth = model.master_bar_width || 0.28;
    const masterWidth = model.master_width || (specificityOrder.length * masterBarWidth + Math.max(specificityOrder.length - 1, 0) * barGap);
    const masterY0 = model.master_y0 || 118;
    const masterY1 = model.master_y1 || 143;
    const lowerScale = model.lower_scale || 100;

    const traces = [];
    const annotations = [];
    const xTickValues = [];
    const xTickLabels = [];
    let xMax = 0;

    const visibleClasses = classes.map(function (cls) {
      return {
        cls: cls,
        specs: (cls.specificities || []).filter(function (item) {
          return item.count > 0 && !isHidden(hidden, "specificity:" + item.name);
        }),
      };
    }).filter(function (entry) {
      return entry.specs.length > 0;
    });

    visibleClasses.forEach(function (entry, classIndex) {
      const cls = entry.cls;
      const groupX0 = xMax + (classIndex ? classGap : 0);
      const allSpecs = entry.specs;

      const groupWidth = masterWidth;
      const masterX0 = groupX0;
      const masterX1 = groupX0 + groupWidth;
      const visibleTotal = allSpecs.reduce(function (total, item) { return total + item.count; }, 0);
      let cursor = masterX0;
      const masterSegments = [];

      allSpecs.forEach(function (item) {
        const share = visibleTotal ? item.count / visibleTotal : 0;
        const width = groupWidth * share;
        if (width <= 0) return;
        const segmentX0 = cursor;
        const segmentX1 = cursor + width;
        masterSegments.push({ item: item, x0: segmentX0, x1: segmentX1 });
        traces.push({
          type: "bar",
          orientation: "h",
          x: [width],
          y: [(masterY0 + masterY1) / 2],
          base: [cursor],
          width: [masterY1 - masterY0],
          marker: { color: specificityColors[item.name], line: { color: "white", width: 0.8 } },
          showlegend: false,
          hovertemplate:
            "<b>%{customdata[0]}</b><br>" +
            "Termini class: %{customdata[1]}<br>" +
            "Items: %{customdata[2]:,}<br>" +
            "Share: %{customdata[3]:.1%}<extra></extra>",
          customdata: [[cls.name, item.name, item.count, share]],
          meta: { role: "master", specificity: item.name },
        });
        cursor += width;
      });

      const lowerCenters = boundedSeparatedPositions(
        masterSegments.map(function (segment) { return (segment.x0 + segment.x1) / 2; }),
        barWidth + barGap,
        masterX0 + barWidth / 2,
        masterX1 - barWidth / 2
      );

      masterSegments.forEach(function (segment, specificityIndex) {
        const item = segment.item;
        const x = lowerCenters[specificityIndex];
        let base = 0;
        const visibleMods = modificationOrder.filter(function (modification) {
          return !isHidden(hidden, "mod:" + modification);
        });
        const modTotal = visibleMods.reduce(function (total, modification) {
          return total + ((item.mod_counts || {})[modification] || 0);
        }, 0);
        visibleMods.forEach(function (modification) {
          const count = ((item.mod_counts || {})[modification] || 0);
          if (count <= 0 || modTotal <= 0) return;
          const height = lowerScale * count / modTotal;
          traces.push({
            type: "bar",
            x: [x],
            y: [height],
            base: [base],
            width: [barWidth],
            marker: { color: modificationColors[modification], line: { color: "white", width: 0.8 } },
            showlegend: false,
            hovertemplate:
              "<b>%{customdata[0]}</b><br>" +
              "Termini class: %{customdata[1]}<br>" +
              "Modification: " + modification + "<br>" +
              "Items: %{customdata[2]:,}<br>" +
              "Share within class: %{customdata[4]:.1%}<extra></extra>",
            customdata: [[cls.name, item.name, count, modTotal, count / modTotal]],
            meta: { role: "detail", modification: modification, specificity: item.name },
          });
          base += height;
        });

        xTickValues.push(x);
        xTickLabels.push(item.name);

        [
          [segment.x0, x - barWidth / 2],
          [segment.x1, x + barWidth / 2],
        ].forEach(function (points) {
          traces.push({
            type: "scatter",
            mode: "lines",
            x: points,
            y: [masterY0, lowerScale],
            line: { color: "rgba(100, 116, 139, 0.55)", dash: "dot", width: 1.2 },
            hoverinfo: "skip",
            showlegend: false,
            meta: { role: "connector" },
          });
        });
      });

      annotations.push({
        text: "<b>" + cls.name + "</b>",
        x: (masterX0 + masterX1) / 2,
        y: masterY1 + 6,
        showarrow: false,
        xanchor: "center",
        yanchor: "bottom",
        font: { size: 12, color: "#111827" },
      });
      const lowerX1 = lowerCenters.length ? lowerCenters[lowerCenters.length - 1] + barWidth / 2 : masterX1;
      xMax = Math.max(masterX1, lowerX1);
    });

    specificityOrder.forEach(function (specificity, index) {
      traces.push(makeLegendTrace("Master: " + shortSpecificityLabel(specificity), "specificity:" + specificity, specificityColors[specificity], hidden, index === 0 ? "Specificity" : ""));
    });
    modificationOrder.forEach(function (modification, index) {
      traces.push(makeLegendTrace("Mod: " + modification, "mod:" + modification, modificationColors[modification], hidden, index === 0 ? "N-term modification" : ""));
    });
    annotations.push({
      text: "Specificity",
      x: -0.06,
      y: (masterY0 + masterY1) / 2,
      xref: "paper",
      showarrow: false,
      textangle: -90,
      xanchor: "center",
      yanchor: "middle",
      font: { size: 14, color: "#111827" },
    });

    const layout = Object.assign({}, gd.layout || {});
    layout.meta = meta;
    layout.annotations = annotations;
    layout.xaxis = Object.assign({}, layout.xaxis || {}, {
      tickmode: "array",
      tickvals: xTickValues,
      ticktext: xTickLabels.map(shortSpecificityLabel),
      tickangle: 30,
      range: [-0.18, xMax + 0.24],
      showline: false,
      gridcolor: "rgba(0,0,0,0)",
      zeroline: false,
    });
    layout.yaxis = Object.assign({}, layout.yaxis || {}, {
      tickvals: [0, 25, 50, 75, 100],
      ticktext: ["0%", "25%", "50%", "75%", "100%"],
      range: [0, masterY1 + 14],
      showline: false,
      gridcolor: "#E8E8E8",
      zeroline: false,
    });
    layout.shapes = [
      {
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 0,
        yref: "y",
        y0: 0,
        y1: lowerScale,
        line: { color: "#111111", width: 1 },
      },
      {
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 0.018,
        yref: "y",
        y0: lowerScale,
        y1: lowerScale,
        line: { color: "#111111", width: 1 },
      },
    ];
    layout.title = Object.assign({}, layout.title || {}, {
      text: "Termini specificity x N-term modification",
      font: { size: 18 },
    });
    layout.xaxis.title = { text: "Class" };
    layout.yaxis.title = { text: "Mod. share" };
    layout.legend = Object.assign({}, layout.legend || {}, {
      title: { text: "" },
      font: { size: 12 },
      traceorder: "normal",
      tracegroupgap: 8,
      y: 0.86,
      yanchor: "top",
    });
    layout.margin = Object.assign({}, layout.margin || {}, { l: 72, r: 148, t: 58, b: 66 });
    layout.font = Object.assign({}, layout.font || {}, { size: 12 });
    layout.uirevision = "termini-temple";

    Plotly.react(gd, traces, layout, gd._context || {}).then(function () {
      attach(gd);
      if (Plotly && Plotly.Plots && Plotly.Plots.resize) {
        Plotly.Plots.resize(gd);
      }
      window.dispatchEvent(new Event("resize"));
      window.setTimeout(function () {
        if (Plotly && Plotly.Plots && Plotly.Plots.resize) {
          Plotly.Plots.resize(gd);
        }
        window.dispatchEvent(new Event("resize"));
      }, 60);
    });
  }

  function attach(gd) {
    if (!gd || gd.__terminiTempleAttached || !templeMeta(gd)) return;
    gd.__terminiTempleAttached = true;
    gd.on("plotly_legendclick", function (eventData) {
      const meta = templeMeta(gd);
      if (!meta) return true;
      const trace = gd.data && gd.data[eventData.curveNumber];
      const group = trace && trace.legendgroup;
      if (!group || (!group.startsWith("mod:") && !group.startsWith("specificity:"))) return true;
      const hidden = hiddenSet(gd);
      hidden[group] = !hidden[group];
      window.setTimeout(function () { rebuild(gd); }, 0);
      return false;
    });
    gd.on("plotly_legenddoubleclick", function (eventData) {
      const meta = templeMeta(gd);
      if (!meta) return true;
      const trace = gd.data && gd.data[eventData.curveNumber];
      const group = trace && trace.legendgroup;
      if (!group || (!group.startsWith("mod:") && !group.startsWith("specificity:"))) return true;
      const hidden = hiddenSet(gd);
      Object.keys(hidden).forEach(function (key) { delete hidden[key]; });
      hidden[group] = true;
      window.setTimeout(function () { rebuild(gd); }, 0);
      return false;
    });
  }

  function scan() {
    document.querySelectorAll(".js-plotly-plot").forEach(attach);
  }

  document.addEventListener("DOMContentLoaded", scan);
  const observer = new MutationObserver(scan);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  window.setInterval(scan, 1000);
})();
