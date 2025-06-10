// Apply saved theme on load
document.addEventListener('DOMContentLoaded', () => {
  if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
  }
  const tooltipped = document.querySelectorAll('.tooltipped');
  M.Tooltip.init(tooltipped);
  const modals = document.querySelectorAll('.modal');
  M.Modal.init(modals);
});

function toggleDarkMode() {
  document.body.classList.toggle('dark-mode');
  localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
}

function openHelp() {
  const modal = document.getElementById('helpModal');
  if (modal) {
    const instance = M.Modal.getInstance(modal) || M.Modal.init(modal);
    instance.open();
  }
}
