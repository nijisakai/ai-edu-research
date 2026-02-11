/* Chart 6: Innovation Depth Distribution */
async function renderInnovationChart(containerId) {
  const data = await DataLoader.load('framework_stats.json');
  const scores = data['创新深度评分'];
  const mean = data['创新深度评分_mean'];

  const levels = Object.keys(scores);
  const counts = Object.values(scores);
  const total = counts.reduce((a, b) => a + b, 0);
  const pcts = counts.map(c => (c / total * 100).toFixed(1));

  const barColors = ['#c6dbef', '#6baed6', '#2171b5', '#08519c', '#08306b'];

  const trace = {
    type: 'bar',
    x: levels.map(l => `${l}级`),
    y: counts,
    marker: { color: barColors, line: { width: 0 } },
    text: counts.map((c, i) => `${c} (${pcts[i]}%)`),
    textposition: 'outside',
    textfont: { size: 12 },
    hovertemplate: '创新深度 %{x}<br>数量: %{y}<br>占比: %{text}<extra></extra>'
  };

  const maxY = Math.max(...counts) * 1.2;

  const layout = {
    height: 400,
    xaxis: { title: '创新深度等级' },
    yaxis: { title: '案例数量', range: [0, maxY], gridcolor: '#f0f0f0' },
    margin: { l: 60, r: 30, t: 30, b: 60 },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    shapes: [{
      type: 'line',
      x0: mean - 0.5, x1: mean - 0.5,
      y0: 0, y1: maxY * 0.9,
      line: { color: '#e8563a', width: 2, dash: 'dash' }
    }],
    annotations: [{
      x: mean - 0.5, y: maxY * 0.92,
      text: `均值 = ${mean}`,
      showarrow: false,
      font: { color: '#e8563a', size: 13, weight: 'bold' }
    }]
  };

  Plotly.newPlot(containerId, [trace], layout, { responsive: true, displayModeBar: false });
}
