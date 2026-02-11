/* Chart 8: Case Browser - Searchable/filterable table */
async function renderCaseBrowser(containerId) {
  const data = await DataLoader.load('case_deep_analysis.json');
  const cases = data.cases || [];

  if (!cases.length) {
    document.getElementById(containerId).innerHTML = '<p style="color:#999;text-align:center;padding:40px">暂无案例数据</p>';
    return;
  }

  // Collect filter options
  const stages = [...new Set(cases.map(c => c.stage).filter(Boolean))].sort();
  const provinces = [...new Set(cases.map(c => c.province).filter(Boolean))].sort();

  // Populate filter dropdowns
  const stageSelect = document.getElementById('case-filter-stage');
  const provSelect = document.getElementById('case-filter-province');

  if (stageSelect) {
    stages.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s; opt.textContent = s;
      stageSelect.appendChild(opt);
    });
  }

  if (provSelect) {
    provinces.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p; opt.textContent = p;
      provSelect.appendChild(opt);
    });
  }

  const container = document.getElementById(containerId);
  container.innerHTML = `
    <div id="case-table-wrapper">
      <table class="case-table">
        <thead><tr>
          <th>编号</th><th>标题</th><th>学段</th>
          <th>省份</th><th>工具</th><th>iSTAR</th><th>五育</th>
        </tr></thead>
        <tbody id="case-tbody"></tbody>
      </table>
    </div>
    <div id="case-detail-panel"></div>
  `;

  let expandedId = null;

  function renderRows(filtered) {
    const tbody = document.getElementById('case-tbody');
    tbody.innerHTML = filtered.map(c => {
      const tools = Array.isArray(c.tools_used) ? c.tools_used.join('、') : (c.tools_used || '');
      return `
        <tr data-id="${c.case_id}" class="case-row">
          <td>${c.case_id || ''}</td>
          <td class="case-title" title="${c.title || ''}">${c.title || ''}</td>
          <td>${c.stage || ''}</td>
          <td>${c.province || ''}</td>
          <td>${tools}</td>
          <td>${c.istar_level || ''}</td>
          <td>${c.wuyu_category || ''}</td>
        </tr>`;
    }).join('');

    // Click to expand
    tbody.querySelectorAll('.case-row').forEach(row => {
      row.addEventListener('click', () => {
        const id = row.dataset.id;
        const detailPanel = document.getElementById('case-detail-panel');
        if (expandedId === id) {
          detailPanel.innerHTML = '';
          expandedId = null;
          return;
        }
        expandedId = id;
        const c = cases.find(x => String(x.case_id) === String(id));
        if (c) {
          detailPanel.innerHTML = `
            <div class="case-detail">
              <strong>案例 ${c.case_id}：${c.title || ''}</strong><br>
              ${c.actual_usage_description ? `<strong>实际使用描述：</strong>${c.actual_usage_description}<br>` : ''}
              ${c.pedagogical_innovation ? `<strong>教学创新：</strong>${c.pedagogical_innovation}<br>` : ''}
              ${c.overall_assessment ? `<strong>总体评价：</strong>${c.overall_assessment}<br>` : ''}
              ${c.deep_learning_evidence ? `<strong>深度学习证据：</strong>${c.deep_learning_evidence}<br>` : ''}
            </div>`;
        }
      });
    });
  }

  function filterCases() {
    const search = (document.getElementById('case-search')?.value || '').toLowerCase();
    const stage = document.getElementById('case-filter-stage')?.value || '';
    const province = document.getElementById('case-filter-province')?.value || '';

    const filtered = cases.filter(c => {
      if (stage && c.stage !== stage) return false;
      if (province && c.province !== province) return false;
      if (search) {
        const haystack = [c.title, c.school, c.province, c.stage,
          Array.isArray(c.tools_used) ? c.tools_used.join(' ') : '',
          c.actual_usage_description || ''].join(' ').toLowerCase();
        if (!haystack.includes(search)) return false;
      }
      return true;
    });
    renderRows(filtered);
  }

  // Bind events
  document.getElementById('case-search')?.addEventListener('input', filterCases);
  document.getElementById('case-filter-stage')?.addEventListener('change', filterCases);
  document.getElementById('case-filter-province')?.addEventListener('change', filterCases);

  // Initial render
  renderRows(cases);
}
