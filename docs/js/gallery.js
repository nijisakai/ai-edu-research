const Gallery = {
  figures: [],
  filtered: [],
  currentIndex: 0,

  async init(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    try {
      const resp = await fetch('data/figures-index.json');
      const data = await resp.json();

      // Flatten all figures from all series
      this.figures = [];
      data.series.forEach(s => {
        s.figures.forEach(f => {
          this.figures.push({ ...f, seriesId: s.id, seriesName: s.name });
        });
      });
      this.filtered = [...this.figures];

      // Build tabs
      const tabsHtml = `
        <div class="gallery-tabs" id="gallery-tabs">
          <button class="active" data-series="all">全部 (${this.figures.length})</button>
          ${data.series.map(s =>
            `<button data-series="${s.id}">${s.id}: ${s.name} (${s.figures.length})</button>`
          ).join('')}
        </div>
      `;

      // Build grid
      const gridHtml = `
        <div class="gallery-grid" id="gallery-grid">
          ${this.figures.map((f, i) => `
            <div class="gallery-card" data-series="${f.seriesId}" data-index="${i}">
              <div class="gallery-thumb">
                <img loading="lazy" src="figures/${f.file}" alt="${f.title}">
              </div>
              <div class="gallery-info">
                <span class="gallery-series-badge">${f.seriesId}</span>
                <h4>${f.title}</h4>
              </div>
            </div>
          `).join('')}
        </div>
      `;

      // Lightbox
      const lightboxHtml = `
        <div class="lightbox" id="lightbox">
          <button class="lightbox-close" id="lightbox-close">&times;</button>
          <button class="lightbox-prev" id="lightbox-prev">&#8249;</button>
          <img class="lightbox-img" id="lightbox-img">
          <button class="lightbox-next" id="lightbox-next">&#8250;</button>
          <div class="lightbox-caption" id="lightbox-caption"></div>
        </div>
      `;

      container.innerHTML = tabsHtml + gridHtml + lightboxHtml;
      this.bindEvents();
    } catch (err) {
      container.innerHTML = '<p style="color:#999;text-align:center">图表数据加载失败</p>';
      console.error('Gallery init error:', err);
    }
  },

  bindEvents() {
    // Tab filtering
    document.getElementById('gallery-tabs').addEventListener('click', e => {
      if (e.target.tagName !== 'BUTTON') return;
      const series = e.target.dataset.series;

      // Update active tab
      document.querySelectorAll('#gallery-tabs button').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');

      // Filter cards
      document.querySelectorAll('.gallery-card').forEach(card => {
        if (series === 'all' || card.dataset.series === series) {
          card.classList.remove('hidden');
        } else {
          card.classList.add('hidden');
        }
      });

      // Update filtered list for lightbox navigation
      if (series === 'all') {
        this.filtered = [...this.figures];
      } else {
        this.filtered = this.figures.filter(f => f.seriesId === series);
      }
    });

    // Card click → lightbox
    document.getElementById('gallery-grid').addEventListener('click', e => {
      const card = e.target.closest('.gallery-card');
      if (!card) return;
      const idx = parseInt(card.dataset.index);
      const fig = this.figures[idx];
      // Find index in filtered list
      const filteredIdx = this.filtered.findIndex(f => f.file === fig.file);
      if (filteredIdx >= 0) this.openLightbox(filteredIdx);
    });

    // Lightbox controls
    document.getElementById('lightbox-close').addEventListener('click', () => this.closeLightbox());
    document.getElementById('lightbox-prev').addEventListener('click', () => this.navigate(-1));
    document.getElementById('lightbox-next').addEventListener('click', () => this.navigate(1));
    document.getElementById('lightbox').addEventListener('click', e => {
      if (e.target.id === 'lightbox') this.closeLightbox();
    });

    // Keyboard
    document.addEventListener('keydown', e => {
      const lb = document.getElementById('lightbox');
      if (!lb || !lb.classList.contains('active')) return;
      if (e.key === 'Escape') this.closeLightbox();
      if (e.key === 'ArrowLeft') this.navigate(-1);
      if (e.key === 'ArrowRight') this.navigate(1);
    });
  },

  openLightbox(index) {
    this.currentIndex = index;
    const fig = this.filtered[index];
    const lb = document.getElementById('lightbox');
    document.getElementById('lightbox-img').src = 'figures/' + fig.file;
    document.getElementById('lightbox-caption').textContent =
      `[${fig.seriesId}] ${fig.title} (${this.currentIndex + 1}/${this.filtered.length})`;
    lb.classList.add('active');
    document.body.style.overflow = 'hidden';
  },

  closeLightbox() {
    document.getElementById('lightbox').classList.remove('active');
    document.body.style.overflow = '';
  },

  navigate(dir) {
    this.currentIndex = (this.currentIndex + dir + this.filtered.length) % this.filtered.length;
    const fig = this.filtered[this.currentIndex];
    document.getElementById('lightbox-img').src = 'figures/' + fig.file;
    document.getElementById('lightbox-caption').textContent =
      `[${fig.seriesId}] ${fig.title} (${this.currentIndex + 1}/${this.filtered.length})`;
  }
};
