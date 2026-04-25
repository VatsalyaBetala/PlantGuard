// Shared bootstrap helpers for PlantGuard pages.
// Each page also carries its own inline theme logic, so this file mainly
// serves as a compatibility shim + AOS bootstrap for any page that loads it.
document.addEventListener('DOMContentLoaded', () => {
    // AOS init — safe no-op if AOS isn't loaded.
    if (window.AOS && typeof window.AOS.init === 'function') {
        try {
            window.AOS.init({
                duration: 650,
                easing: 'ease-out-cubic',
                once: true,
                offset: 60
            });
        } catch (_) { /* noop */ }
    }

    // Restore saved theme for any page that hasn't already.
    if (!document.body.hasAttribute('data-theme') && localStorage.getItem('theme') === 'dark') {
        document.body.setAttribute('data-theme', 'dark');
    }

    // Scroll-aware navbar fallback — every page now has an inline handler,
    // but keep this as a belt-and-braces for older pages.
    const navbar = document.querySelector('.navbar-custom');
    if (navbar) {
        const onScroll = () => navbar.classList.toggle('is-scrolled', window.scrollY > 8);
        window.addEventListener('scroll', onScroll, { passive: true });
        onScroll();
    }
});

// Legacy dark-mode toggle kept so any stray onclick="toggleDarkMode()" still
// works on pages I didn't touch in the future.
if (typeof window.toggleDarkMode !== 'function') {
    window.toggleDarkMode = function () {
        const body = document.body;
        const icon = document.getElementById('themeIcon');
        const isDark = body.getAttribute('data-theme') === 'dark';
        if (isDark) {
            body.removeAttribute('data-theme');
            if (icon) icon.className = 'fas fa-moon';
            localStorage.setItem('theme', 'light');
        } else {
            body.setAttribute('data-theme', 'dark');
            if (icon) icon.className = 'fas fa-sun';
            localStorage.setItem('theme', 'dark');
        }
    };
}
