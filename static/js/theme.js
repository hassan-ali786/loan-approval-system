// Theme toggle
const toggleBtn = document.getElementById('themeToggle');
const root = document.documentElement;

// Load saved theme
const saved = localStorage.getItem('theme') || 'dark';
root.setAttribute('data-theme', saved);
if (toggleBtn) toggleBtn.textContent = saved === 'dark' ? '☀️' : '🌙';

if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
        const current = root.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        root.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
        toggleBtn.textContent = next === 'dark' ? '☀️' : '🌙';
    });
}

// Loading animation on form submit
const form = document.getElementById('loanForm');
const overlay = document.getElementById('loadingOverlay');

if (form && overlay) {
    form.addEventListener('submit', () => {
        overlay.classList.add('active');
    });
}