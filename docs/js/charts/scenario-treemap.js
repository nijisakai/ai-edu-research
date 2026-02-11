/* Chart 3: Application Scenarios - Treemap / Sunburst toggle */
async function renderScenarioChart(containerId) {
  const data = await DataLoader.load('scenario_analysis.json');
  const l1 = data.scenario_level1;
  const l2 = data.scenario_level2;

  // Map L2 to L1 based on known relationships
  const l2ToL1Map = {
    '智能辅导系统': '助学', '情境式学习': '助学', '游戏化学习': '助学',
    '智能阅读辅助': '助学', '智能学情诊断与个性化辅导': '助学',
    '教学分析': '助教', '教师备课': '助教', '课堂管理': '助教',
    '教学设计': '助教', '智能备课': '助教', '教学辅助': '助教',
    '智能教师专业发展平台': '助教',
    '综合素质评价': '助评', '学生评估': '助评', '学情分析': '助评',
    '智能德育与综合素质评价': '助评', '学生成长数据分析与预警': '助评',
    '校园安全智能监控': '助管', '学生信息智能管理': '助管',
    '教务管理智能化': '助管',
    '智能心理支持': '助育', '智能美育教育': '助育',
    '智能体育健康': '助育', '智能体育助手': '助育',
    '虚拟实验与实践教学': '助学', '课程设计': '助教',
    '智能作业设计与批改': '助教', '作业管理': '助教'
  };

  // Build hierarchy
  const labels = ['全部场景'];
  const parents = [''];
  const values = [0];
  const colors = {
    '助学': '#4e79a7', '助教': '#f28e2b', '助育': '#e15759',
    '助评': '#76b7b2', '助管': '#59a14f', '助研': '#edc948'
  };

  // Add L1 nodes
  for (const [name, count] of Object.entries(l1)) {
    if (name === '未提及') continue;
    labels.push(name + ' (' + count + ')');
    parents.push('全部场景');
    values.push(count);
  }

  // Add L2 nodes under their L1 parent
  for (const [name, count] of Object.entries(l2)) {
    if (name === '未提及' || name.includes('；') || count < 5) continue;
    const parent = l2ToL1Map[name];
    if (parent && l1[parent]) {
      labels.push(name);
      parents.push(parent + ' (' + l1[parent] + ')');
      values.push(count);
    }
  }

  const colorArr = labels.map((l, i) => {
    if (i === 0) return '#ddd';
    for (const [k, c] of Object.entries(colors)) {
      if (l.startsWith(k)) return c;
      const parentLabel = parents[i];
      if (parentLabel && parentLabel.startsWith(k)) return c;
    }
    return '#999';
  });

  function renderView(viewType) {
    const trace = {
      type: viewType,
      labels: labels,
      parents: parents,
      values: values,
      marker: { colors: colorArr },
      textinfo: 'label+percent parent',
      hovertemplate: '%{label}<br>数量: %{value}<extra></extra>',
      branchvalues: 'total'
    };

    if (viewType === 'treemap') {
      trace.tiling = { pad: 2 };
    }

    Plotly.newPlot(containerId, [trace], {
      height: 500,
      margin: { l: 10, r: 10, t: 10, b: 10 },
      paper_bgcolor: 'white'
    }, { responsive: true, displayModeBar: false });
  }

  renderView('treemap');

  // Set up toggle buttons
  const panel = document.getElementById(containerId).closest('.chart-panel');
  const buttons = panel.querySelectorAll('.chart-toggle button');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderView(btn.dataset.view);
    });
  });
}
