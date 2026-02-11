const AUTH = {
  STORAGE_KEY: 'ai-edu-api-key',
  VALID_KEY: 'BNUSLI',

  isAuthenticated() {
    return sessionStorage.getItem(this.STORAGE_KEY) === this.VALID_KEY;
  },

  authenticate(key) {
    if (key === this.VALID_KEY) {
      sessionStorage.setItem(this.STORAGE_KEY, key);
      return true;
    }
    return false;
  },

  showGate() {
    const overlay = document.createElement('div');
    overlay.id = 'auth-gate';
    overlay.innerHTML = `
      <div class="auth-dialog">
        <h2>数据访问验证</h2>
        <p>请输入API密钥以访问研究数据</p>
        <input type="text" id="auth-key-input" placeholder="请输入密钥" autocomplete="off">
        <button id="auth-submit">验证</button>
        <p class="auth-error" id="auth-error" style="display:none">密钥无效，请重试</p>
      </div>
    `;
    document.body.appendChild(overlay);

    const submit = () => {
      const key = document.getElementById('auth-key-input').value.trim();
      if (this.authenticate(key)) {
        overlay.remove();
        document.dispatchEvent(new Event('auth-success'));
      } else {
        document.getElementById('auth-error').style.display = 'block';
        document.getElementById('auth-key-input').classList.add('error');
        setTimeout(() => document.getElementById('auth-key-input').classList.remove('error'), 600);
      }
    };

    document.getElementById('auth-submit').addEventListener('click', submit);
    document.getElementById('auth-key-input').addEventListener('keydown', e => {
      if (e.key === 'Enter') submit();
    });

    // Auto-focus the input
    setTimeout(() => document.getElementById('auth-key-input').focus(), 100);
  },

  require() {
    if (!this.isAuthenticated()) {
      this.showGate();
      return false;
    }
    return true;
  }
};
