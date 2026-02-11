/* Centralized data loader with caching and auth integration */
const DataLoader = {
  cache: {},
  basePath: 'data/',

  async load(filename) {
    if (this.cache[filename]) return this.cache[filename];

    // Wait for auth if not yet authenticated
    if (typeof AUTH !== 'undefined' && !AUTH.isAuthenticated()) {
      await new Promise(resolve => {
        document.addEventListener('auth-success', resolve, { once: true });
      });
    }

    const resp = await fetch(this.basePath + filename);
    if (!resp.ok) throw new Error(`Failed to load ${filename}: ${resp.status}`);
    const data = await resp.json();
    this.cache[filename] = data;
    return data;
  },

  async loadAll(filenames) {
    return Promise.all(filenames.map(f => this.load(f)));
  }
};
