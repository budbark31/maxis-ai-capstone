/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./public/index.html",
    "./src/**/*.{js,jsx,ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        "marywood-green": "#005C49",
        "marywood-green-soft": "#0B6F5A",
        "marywood-gold": "#B5A36A",
        "dark-bg": "#131B1A",
        "dark-surface": "#1E2725",
        "dark-surface-alt": "#24312E",
        "dark-border": "#2F3C39",
        "dark-text": "#E4E7E6",
        "dark-subtext": "#98A5A2"
      },
      boxShadow: {
        "glow-green": "0 0 0 2px rgba(0,92,73,0.35), 0 0 18px -4px rgba(0,92,73,0.55)"
      },
      keyframes: {
        fadeIn: { "0%": { opacity: 0 }, "100%": { opacity: 1 } },
        slideUp: { "0%": { opacity: 0, transform: "translateY(16px)" }, "100%": { opacity: 1, transform: "translateY(0)" } }
      },
      animation: {
        "fade-in": "fadeIn .35s ease-out both",
        "slide-up": "slideUp .45s cubic-bezier(0.16,0.8,0.32,1) both"
      }
    }
  },
  plugins: []
};
