/* Main Application Controller */
document.addEventListener('DOMContentLoaded', () => {
  const loading = document.getElementById('loading-overlay');

  // Check authentication
  if (!AUTH.require()) {
    // Not authenticated — wait for auth-success event
    document.addEventListener('auth-success', () => initDashboard(loading), { once: true });
  } else {
    initDashboard(loading);
  }
});

async function initDashboard(loading) {
  try {
    // Preload critical data files in parallel
    await DataLoader.loadAll([
      'province_distribution.json',
      'tool_product_distribution.json',
      'scenario_analysis.json',
      'framework_stats.json',
      'nlp_clusters.json',
      'nlp_cultivation.json',
      'case_deep_analysis.json'
    ]);

    // Render all charts in parallel
    await Promise.all([
      renderProvinceChart('chart-province'),
      renderToolChart('chart-tools'),
      renderScenarioChart('chart-scenario'),
      renderFrameworkHeatmap('chart-framework'),
      renderClusterChart('chart-clusters'),
      renderInnovationChart('chart-innovation'),
      renderWuyuRadar('chart-wuyu'),
      renderCaseBrowser('chart-cases')
    ]);

    // Initialize figure gallery
    if (typeof Gallery !== 'undefined') {
      await Gallery.init('figure-gallery');
    }

  } catch (err) {
    console.error('Dashboard initialization failed:', err);
    const main = document.querySelector('.dashboard-grid');
    if (main) {
      main.innerHTML = `<div class="chart-panel full-width" style="text-align:center;padding:60px">
        <h2 style="color:#e15759">数据加载失败</h2>
        <p style="color:#666;margin-top:12px">${err.message}</p>
        <p style="color:#999;margin-top:8px;font-size:13px">
          请确保从 Web 服务器访问此页面（非 file:// 协议）。<br>
          本地测试可使用: <code>python -m http.server 8000</code>
        </p>
      </div>`;
    }
  } finally {
    if (loading) loading.style.display = 'none';
  }
}
