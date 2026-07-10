// ─── Theme toggle ───────────────────────────────
const saved = localStorage.getItem('loaniq-theme') || 'dark';
document.documentElement.setAttribute('data-theme', saved);

const themeBtn = document.getElementById('themeToggle');
if (themeBtn) {
    themeBtn.textContent = saved === 'dark' ? '☀' : '🌙';
    themeBtn.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next    = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('loaniq-theme', next);
        themeBtn.textContent = next === 'dark' ? '☀' : '🌙';
    });
}

// ─── Loading overlay on form submit ─────────────
const form    = document.getElementById('loanForm');
const overlay = document.getElementById('loadingOverlay');
if (form && overlay) {
    form.addEventListener('submit', () => {
        overlay.classList.add('active');
    });
}

// ─── Mobile sidebar toggle ──────────────────────
const sidebar     = document.getElementById('sidebar');
const sidebarBtn  = document.getElementById('sidebarToggle');
if (sidebarBtn && sidebar) {
    sidebarBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });
    document.addEventListener('click', e => {
        if (!e.target.closest('.sidebar') && !e.target.closest('#sidebarToggle')) {
            sidebar.classList.remove('open');
        }
    });
}