/* Chart 7: Five-Education (五育) Radar Chart */
async function renderWuyuRadar(containerId) {
  const data = await DataLoader.load('nlp_cultivation.json');

  const fiveEdu = ['智育', '德育', '美育', '体育', '劳育'];

  // Aggregate from overall data - handle compound categories
  function aggregateFiveEdu(src) {
    const result = { '智育': 0, '德育': 0, '美育': 0, '体育': 0, '劳育': 0 };
    for (const [key, count] of Object.entries(src)) {
      for (const edu of fiveEdu) {
        const base = edu.replace('育', '');
        if (key === edu || key.includes(base)) {
          result[edu] += count;
        }
      }
    }
    return result;
  }

  // Overall trace
  const overall = aggregateFiveEdu(data.overall);
  const overallMax = Math.max(...Object.values(overall));

  // Normalize to 0-100 for radar
  function normalize(vals, max) {
    return fiveEdu.map(e => (vals[e] / max * 100).toFixed(1));
  }

  const traces = [];

  // Per-stage traces
  const stageColors = {
    '小学': { line: '#4e79a7', fill: 'rgba(78,121,167,0.1)' },
    '初中': { line: '#f28e2b', fill: 'rgba(242,142,43,0.1)' },
    '高中': { line: '#e15759', fill: 'rgba(225,87,89,0.1)' },
    '幼儿园': { line: '#76b7b2', fill: 'rgba(118,183,178,0.1)' }
  };

  for (const [stage, colors] of Object.entries(stageColors)) {
    const stageData = data.by_school_level[stage];
    if (!stageData) continue;
    const agg = aggregateFiveEdu(stageData);
    const vals = normalize(agg, overallMax);
    traces.push({
      type: 'scatterpolar',
      r: [...vals, vals[0]],
      theta: [...fiveEdu, fiveEdu[0]],
      name: stage,
      fill: 'toself',
      fillcolor: colors.fill,
      line: { color: colors.line, width: 2 },
      customdata: fiveEdu.map(e => agg[e]),
      hovertemplate: `${stage} - %{theta}: %{customdata}条<extra></extra>`
    });
  }

  // Overall trace (bold)
  const overallVals = normalize(overall, overallMax);
  traces.push({
    type: 'scatterpolar',
    r: [...overallVals, overallVals[0]],
    theta: [...fiveEdu, fiveEdu[0]],
    name: '全部',
    fill: 'toself',
    fillcolor: 'rgba(26,58,92,0.05)',
    line: { color: '#1a3a5c', width: 3 },
    customdata: fiveEdu.map(e => overall[e]),
    hovertemplate: '全部 - %{theta}: %{customdata}条<extra></extra>'
  });

  const layout = {
    polar: {
      radialaxis: {
        visible: true,
        range: [0, 100],
        ticksuffix: '%',
        tickfont: { size: 10 },
        gridcolor: '#e0e0e0'
      },
      angularaxis: {
        tickfont: { size: 13 }
      }
    },
    height: 480,
    margin: { l: 60, r: 60, t: 30, b: 30 },
    paper_bgcolor: 'white',
    legend: { x: 0, y: -0.15, orientation: 'h', font: { size: 12 } },
    showlegend: true
  };

  Plotly.newPlot(containerId, traces, layout, { responsive: true, displayModeBar: false });
}
