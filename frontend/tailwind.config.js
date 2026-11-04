/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        navy: { DEFAULT: "#0f2744", 700: "#1b3a5f", 500: "#4a6380", 300: "#8da0b6" },
        canvas: "#f4f6f9",
        line: "#e3e8ef",
        brand: { DEFAULT: "#2f6fd6", soft: "#e8f0fc", 600: "#245bb3" },
        teal: { DEFAULT: "#0e8f86", soft: "#e3f5f3", 600: "#0b7770" },
      },
      fontFamily: { sans: ["Inter", "system-ui", "Segoe UI", "Arial", "sans-serif"] },
      boxShadow: { card: "0 1px 2px rgba(15,39,68,.05), 0 4px 16px rgba(15,39,68,.06)" },
      borderRadius: { xl2: "1.1rem" },
    },
  },
  plugins: [],
};
