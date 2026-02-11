(function() {
  // Detect if we're in a subdirectory (docs/docs/*.html)
  var path = window.location.pathname;
  var inSubdir = path.indexOf('/docs/') > -1 && path.split('/docs/').length > 2
                 || path.match(/\/docs\/[^/]+\/[^/]+\.html/);
  // More reliable: check if current page is inside docs/ subdir
  var segments = path.split('/');
  var filename = segments[segments.length - 1] || 'index.html';
  var parentDir = segments[segments.length - 2] || '';
  var isSubdir = (parentDir === 'docs' && filename.endsWith('.html'));

  var prefix = isSubdir ? '../' : '';

  var nav = document.createElement('nav');
  nav.className = 'site-nav';
  nav.innerHTML =
    '<div class="nav-inner">' +
      '<a href="' + prefix + 'index.html" class="nav-brand">' +
        '<span class="nav-title">AI赋能基础教育研究</span>' +
        '<span class="nav-author">陈虹宇</span>' +
      '</a>' +
      '<div class="nav-links" id="nav-links">' +
        '<a href="' + prefix + 'index.html" data-page="index">图表画廊</a>' +
        '<a href="' + prefix + 'paper.html" data-page="paper">学术论文</a>' +
        '<a href="' + prefix + 'report.html" data-page="report">研究报告</a>' +
        '<a href="' + prefix + 'huang-search.html" data-page="huang-search">黄老师检索</a>' +
        '<div class="nav-dropdown" id="nav-dropdown">' +
          '<button class="nav-dropdown-toggle" data-page="docs">研究文档 <span class="dropdown-arrow">▾</span></button>' +
          '<div class="nav-dropdown-menu">' +
            '<div class="nav-dropdown-group">' +
              '<div class="nav-dropdown-heading">分析报告（中文）</div>' +
              '<a href="' + prefix + 'docs/case-analysis-zh.html">案例深度分析摘要</a>' +
              '<a href="' + prefix + 'docs/causal-analysis-zh.html">因果与统计分析报告</a>' +
              '<a href="' + prefix + 'docs/deep-insights-zh.html">深度洞察挖掘报告</a>' +
              '<a href="' + prefix + 'docs/industry-research-zh.html">产业研究简报</a>' +
            '</div>' +
            '<div class="nav-dropdown-group">' +
              '<div class="nav-dropdown-heading">Analysis Reports (EN)</div>' +
              '<a href="' + prefix + 'docs/case-analysis-en.html">Case Analysis Summary</a>' +
              '<a href="' + prefix + 'docs/causal-analysis-en.html">Causal Analysis Summary</a>' +
              '<a href="' + prefix + 'docs/deep-insights-en.html">Deep Insights Summary</a>' +
              '<a href="' + prefix + 'docs/industry-research-en.html">Industry Research</a>' +
            '</div>' +
            '<div class="nav-dropdown-group">' +
              '<div class="nav-dropdown-heading">参考资料</div>' +
              '<a href="' + prefix + 'docs/huang-ronghuai.html">黄荣怀理论研究</a>' +
              '<a href="' + prefix + 'docs/paper-review.html">论文审读报告</a>' +
              '<a href="' + prefix + 'docs/report-review.html">报告审读报告</a>' +
              '<a href="' + prefix + 'docs/tech-guide.html">技术说明与图表解读指南</a>' +
            '</div>' +
            '<div class="nav-dropdown-group nav-dropdown-divider">' +
              '<a href="' + prefix + 'json-viewer.html">JSON数据浏览器</a>' +
            '</div>' +
          '</div>' +
        '</div>' +
        '<a href="' + prefix + 'api.html" data-page="api">数据API</a>' +
      '</div>' +
      '<div class="nav-actions">' +
        '<a href="https://github.com/nijisakai/ai-edu-research" target="_blank" rel="noopener" class="nav-github" title="GitHub">' +
          '<svg height="20" width="20" viewBox="0 0 16 16" fill="currentColor">' +
            '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>' +
          '</svg>' +
        '</a>' +
        '<button class="nav-mobile-toggle" id="nav-toggle" aria-label="菜单">' +
          '<span></span><span></span><span></span>' +
        '</button>' +
      '</div>' +
    '</div>';
  document.body.prepend(nav);

  // Active link detection
  var page = filename.replace('.html', '') || 'index';
  if (page === '') page = 'index';

  // Check if we're on a docs/* or json-viewer page
  var isDocsPage = isSubdir || page === 'json-viewer';

  nav.querySelectorAll('.nav-links > a').forEach(function(a) {
    var dp = a.getAttribute('data-page');
    if (dp && dp === page) a.classList.add('active');
    if (dp === 'index' && (page === '' || page === 'index')) a.classList.add('active');
  });

  // Highlight dropdown toggle if on a docs page
  if (isDocsPage) {
    var toggle = nav.querySelector('.nav-dropdown-toggle');
    if (toggle) toggle.classList.add('active');
  }

  // Highlight active link within dropdown
  nav.querySelectorAll('.nav-dropdown-menu a').forEach(function(a) {
    var href = a.getAttribute('href');
    if (href && path.endsWith(href.replace(prefix, '').replace('../', ''))) {
      a.classList.add('active');
    }
  });

  // Desktop: hover to open dropdown
  var dropdown = document.getElementById('nav-dropdown');
  if (dropdown) {
    dropdown.addEventListener('mouseenter', function() {
      this.classList.add('open');
    });
    dropdown.addEventListener('mouseleave', function() {
      this.classList.remove('open');
    });

    // Click toggle for mobile/touch
    dropdown.querySelector('.nav-dropdown-toggle').addEventListener('click', function(e) {
      e.preventDefault();
      dropdown.classList.toggle('open');
    });
  }

  // Mobile toggle
  document.getElementById('nav-toggle').addEventListener('click', function() {
    document.getElementById('nav-links').classList.toggle('open');
  });

  // Close dropdown when clicking outside
  document.addEventListener('click', function(e) {
    if (dropdown && !dropdown.contains(e.target)) {
      dropdown.classList.remove('open');
    }
  });
})();
