// Client-side interactions for Land Acquisition UI

document.addEventListener('DOMContentLoaded', () => {
    // Active navigation item highlighting based on current path
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link-gov').forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPath || (href !== '/' && currentPath.startsWith(href))) {
            link.classList.add('active');
        } else if (href === '/' && currentPath === '/') {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });

    // Light / Dark Theme Toggle for Landing Page
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const themeToggleIcon = document.getElementById('themeToggleIcon');

    function applyTheme(theme) {
        if (theme === 'light') {
            document.body.classList.add('light-theme');
            if (themeToggleIcon) {
                themeToggleIcon.className = 'bi bi-moon-stars-fill';
            }
            if (themeToggleBtn) {
                themeToggleBtn.setAttribute('title', 'Switch to Dark Mode');
            }
        } else {
            document.body.classList.remove('light-theme');
            if (themeToggleIcon) {
                themeToggleIcon.className = 'bi bi-sun-fill';
            }
            if (themeToggleBtn) {
                themeToggleBtn.setAttribute('title', 'Switch to Light Mode');
            }
        }
    }

    // Initialize theme from localStorage (default: dark on landing page)
    const savedTheme = localStorage.getItem('landpredict-home-theme') || 'dark';
    applyTheme(savedTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const isLight = document.body.classList.contains('light-theme');
            const newTheme = isLight ? 'dark' : 'light';
            applyTheme(newTheme);
            localStorage.setItem('landpredict-home-theme', newTheme);
        });
    }

    const areaAcres = document.getElementById('areaAcres');
    const areaHectares = document.getElementById('areaHectares');

    // Auto-sync between Acres and Hectares
    if (areaAcres && areaHectares) {
        areaAcres.addEventListener('input', () => {
            const val = parseFloat(areaAcres.value);
            if (!isNaN(val)) {
                areaHectares.value = (val * 0.404686).toFixed(2);
            }
        });

        areaHectares.addEventListener('input', () => {
            const val = parseFloat(areaHectares.value);
            if (!isNaN(val)) {
                areaAcres.value = (val * 2.47105).toFixed(2);
            }
        });
    }
});
