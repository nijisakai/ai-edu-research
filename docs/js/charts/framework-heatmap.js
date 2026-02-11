/* Chart 4: Framework Cross-Tab Heatmap with dropdown selector */
async function renderFrameworkHeatmap(containerId) {
  const data = await DataLoader.load('framework_stats.json');

  const crossTabs = {
    '三赋能 × iSTAR协同层级': data['三赋能_x_iSTAR'],
    '三赋能 × 技术代际': data['三赋能_x_技术代际'],
    'iSTAR × 技术路径': data['iSTAR_x_技术路径']
  };

  function renderHeatmap(tabName) {
    const crossTab = crossTabs[tabName];
    const yLabels = Object.keys(crossTab);
    const xLabels = [...new Set(yLabels.flatMap(k => Object.keys(crossTab[k])))];
    const zValues = yLabels.map(y => xLabels.map(x => crossTab[y][x] || 0));

    // Create annotation text
    const annotations = [];
    for (let i = 0; i < yLabels.length; i++) {
      for (let j = 0; j < xLabels.length; j++) {
        const val = zValues[i][j];
        annotations.push({
          x: xLabels[j], y: yLabels[i],
          text: val > 0 ? val.toString() : '',
          showarrow: false,
          font: { color: val > 500 ? 'white' : '#333', size: 12 }
        });
      }
    }

    const trace = {
      type: 'heatmap',
      z: zValues,
      x: xLabels,
      y: yLabels,
      colorscale: [[0, '#f7fbff'], [0.2, '#c6dbef'], [0.4, '#6baed6'], [0.6, '#2171b5'], [0.8, '#08519c'], [1, '#08306b']],
      hovertemplate: '%{y} × %{x}<br>数量: %{z}<extra></extra>',
      showscale: true,
      colorbar: { title: '数量', thickness: 15 }
    };

    Plotly.newPlot(containerId, [trace], {
      annotations: annotations,
      height: 400,
      margin: { l: 140, r: 80, t: 10, b: 100 },
      xaxis: { tickangle: -30, tickfont: { size: 11 } },
      yaxis: { tickfont: { size: 11 } },
      plot_bgcolor: 'white',
      paper_bgcolor: 'white'
    }, { responsive: true, displayModeBar: false });
  }

  // Create dropdown
  const panel = document.getElementById(containerId).closest('.chart-panel');
  const selectContainer = panel.querySelector('.chart-select-container');
  if (selectContainer) {
    const select = document.createElement('select');
    select.className = 'chart-select';
    Object.keys(crossTabs).forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      select.appendChild(opt);
    });
    select.addEventListener('change', () => renderHeatmap(select.value));
    selectContainer.appendChild(select);
  }

  renderHeatmap('三赋能 × iSTAR协同层级');
}
