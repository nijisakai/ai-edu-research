/* Chart 1: Province Distribution - Bar chart + map toggle */
async function renderProvinceChart(containerId) {
  const data = await DataLoader.load('province_distribution.json');
  const caseData = data.case_count_by_province;
  const toolData = data.tool_count_by_province;

  // Sort by case count descending, exclude 未提及
  const entries = Object.entries(caseData)
    .filter(([k]) => k !== '未提及')
    .sort((a, b) => b[1] - a[1]);

  const provinces = entries.map(e => e[0].replace(/省|市|自治区|壮族|维吾尔|回族/g, ''));
  const fullNames = entries.map(e => e[0]);
  const caseCounts = entries.map(e => e[1]);
  const toolCounts = entries.map(e => toolData[e[0]] || 0);
  const total = caseCounts.reduce((a, b) => a + b, 0);
  const pcts = caseCounts.map(c => (c / total * 100).toFixed(1));

  const trace = {
    type: 'bar',
    y: provinces,
    x: caseCounts,
    orientation: 'h',
    marker: {
      color: caseCounts,
      colorscale: [[0, '#fee5d9'], [0.25, '#fcae91'], [0.5, '#fb6a4a'], [0.75, '#de2d26'], [1, '#a50f15']],
      line: { width: 0 }
    },
    customdata: toolCounts.map((t, i) => [t, pcts[i], fullNames[i]]),
    hovertemplate: '%{customdata[2]}<br>案例数: %{x}<br>工具数: %{customdata[0]}<br>占比: %{customdata[1]}%<extra></extra>',
    name: '案例数'
  };

  const layout = {
    xaxis: { title: '案例数量', gridcolor: '#f0f0f0' },
    yaxis: { autorange: 'reversed', dtick: 1, tickfont: { size: 11 } },
    height: 700,
    margin: { l: 100, r: 30, t: 10, b: 50 },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white'
  };

  Plotly.newPlot(containerId, [trace], layout, { responsive: true, displayModeBar: false });
}
