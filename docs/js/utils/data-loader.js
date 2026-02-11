/* Centralized data loader with caching */
const DataLoader = {
  cache: {},
  basePath: 'data/',

  async load(filename) {
    if (this.cache[filename]) return this.cache[filename];
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
