/* Chart 2: Tool Ecosystem - Top 20 horizontal bar */
async function renderToolChart(containerId) {
  const data = await DataLoader.load('tool_product_distribution.json');
  const top30 = data.top30_tools;
  const totalMentions = data.total_tool_mentions;

  const entries = Object.entries(top30).slice(0, 20).reverse();
  const tools = entries.map(e => e[0]);
  const counts = entries.map(e => e[1]);
  const pcts = counts.map(c => (c / totalMentions * 100).toFixed(1));

  const trace = {
    type: 'bar',
    y: tools,
    x: counts,
    orientation: 'h',
    marker: {
      color: counts,
      colorscale: 'Viridis',
      line: { width: 0 }
    },
    text: pcts.map(p => p + '%'),
    textposition: 'outside',
    textfont: { size: 11 },
    customdata: pcts,
    hovertemplate: '%{y}<br>使用次数: %{x}<br>占比: %{customdata}%<extra></extra>'
  };

  const layout = {
    xaxis: { title: '使用次数', gridcolor: '#f0f0f0' },
    yaxis: { tickfont: { size: 11 } },
    height: 550,
    margin: { l: 130, r: 60, t: 10, b: 50 },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white'
  };

  Plotly.newPlot(containerId, [trace], layout, { responsive: true, displayModeBar: false });
}
