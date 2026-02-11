/* Chart 5: Semantic Clusters - Bubble chart */
async function renderClusterChart(containerId) {
  const data = await DataLoader.load('nlp_clusters.json');
  const clusters = data.clusters;
  const ids = Object.keys(clusters);

  // Arrange in circular layout
  const n = ids.length;
  const xCoords = ids.map((_, i) => Math.cos(2 * Math.PI * i / n) * 4);
  const yCoords = ids.map((_, i) => Math.sin(2 * Math.PI * i / n) * 4);
  const sizes = ids.map(id => clusters[id].size);
  const maxSize = Math.max(...sizes);

  const clusterColors = [
    '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
    '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac'
  ];

  const hoverTexts = ids.map(id => {
    const c = clusters[id];
    const terms = c.top_terms.slice(0, 5).map(t => t.word).join('、');
    return `聚类 ${id} (n=${c.size})<br>关键词: ${terms}`;
  });

  const labelTexts = ids.map(id => {
    const terms = clusters[id].top_terms.slice(0, 2).map(t => t.word).join('/');
    return `C${id}: ${terms}`;
  });

  const trace = {
    type: 'scatter',
    mode: 'markers+text',
    x: xCoords,
    y: yCoords,
    marker: {
      size: sizes.map(s => Math.sqrt(s / maxSize) * 80 + 20),
      color: ids.map(Number),
      colorscale: clusterColors.map((c, i) => [i / (n - 1), c]),
      opacity: 0.75,
      line: { width: 2, color: 'white' }
    },
    text: labelTexts,
    textposition: 'top center',
    textfont: { size: 11 },
    hovertext: hoverTexts,
    hoverinfo: 'text',
    customdata: ids.map(id => ({
      size: clusters[id].size,
      terms: clusters[id].top_terms.slice(0, 8)
    }))
  };

  const layout = {
    height: 500,
    xaxis: { visible: false, range: [-7, 7] },
    yaxis: { visible: false, range: [-7, 7], scaleanchor: 'x' },
    margin: { l: 20, r: 20, t: 10, b: 10 },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    showlegend: false,
    annotations: ids.map((id, i) => ({
      x: xCoords[i], y: yCoords[i],
      text: `<b>${clusters[id].size}</b>`,
      showarrow: false,
      font: { size: 14, color: '#333' }
    }))
  };

  Plotly.newPlot(containerId, [trace], layout, { responsive: true, displayModeBar: false });
}
