(function () {
  const THEME_KEY = 'theme';

  function applyTheme(theme) {
    const body = document.body;
    const icon = document.getElementById('themeIcon');
    const isDark = theme === 'dark';

    if (isDark) {
      body.setAttribute('data-theme', 'dark');
      body.classList.add('dark-mode');
    } else {
      body.removeAttribute('data-theme');
      body.classList.remove('dark-mode');
    }

    if (icon) {
      icon.className = isDark ? 'fas fa-sun' : 'fas fa-moon';
    }
  }

  function getSavedTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === 'dark' || saved === 'light') return saved;

    const legacy = localStorage.getItem('darkMode');
    if (legacy === 'true') return 'dark';
    return 'light';
  }

  function toggleTheme() {
    const nextTheme = document.body.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    applyTheme(nextTheme);
    localStorage.setItem(THEME_KEY, nextTheme);
    localStorage.setItem('darkMode', String(nextTheme === 'dark'));
  }

  function initTheme() {
    applyTheme(getSavedTheme());
  }

  function initAOS() {
    if (!window.AOS) return;
    AOS.init({
      duration: 700,
      easing: 'ease-out-cubic',
      once: true,
      offset: 80
    });
  }

  function initNavbarScroll() {
    const navbar = document.querySelector('.navbar-custom');
    if (!navbar) return;

    const update = () => {
      navbar.classList.toggle('nav-scrolled', window.scrollY > 30);
    };

    update();
    window.addEventListener('scroll', update);
  }

  function initButtonLoading(selector) {
    document.querySelectorAll(selector).forEach((btn) => {
      btn.addEventListener('click', function () {
        const originalContent = this.innerHTML;
        this.style.pointerEvents = 'none';
        this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
        setTimeout(() => {
          this.innerHTML = originalContent;
          this.style.pointerEvents = 'auto';
        }, 900);
      });
    });
  }

  function saveHistoryEntries(predictions) {
    if (!Array.isArray(predictions)) return;

    const existing = JSON.parse(localStorage.getItem('analysisHistory') || '[]');
    const stamped = predictions.map((item) => ({
      ...item,
      timestamp: item.timestamp || new Date().toISOString()
    }));
    const merged = [...stamped, ...existing].slice(0, 50);
    localStorage.setItem('analysisHistory', JSON.stringify(merged));
  }

  function initPage(options = {}) {
    initTheme();
    initAOS();
    initNavbarScroll();

    if (options.enableButtonLoading) {
      initButtonLoading(options.enableButtonLoading);
    }

    if (window.M) {
      M.AutoInit();
    }
  }

  window.toggleDarkMode = toggleTheme;
  window.AppUI = {
    initTheme,
    initAOS,
    initNavbarScroll,
    initButtonLoading,
    initPage,
    saveHistoryEntries
  };

  document.addEventListener('DOMContentLoaded', initTheme);
})();
