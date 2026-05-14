/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        mono: ["JetBrains Mono", "SF Mono", "Fira Code", "Consolas", "Menlo", "monospace"],
      },
      colors: {
        // Status palette (matches render_comparison.png)
        pass:    { DEFAULT: "#a3d8a3", dark: "#5a9c5a" },
        fail:    { DEFAULT: "#f0928b", dark: "#b35047" },
        build:   { DEFAULT: "#ed9d4a", dark: "#a86826" },
        partial: { DEFAULT: "#f5d870", dark: "#b89c3d" },
      },
    },
  },
  plugins: [],
};
