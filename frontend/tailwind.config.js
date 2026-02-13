/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        gc: {
          red: '#D3002D',      // Standard Gov Canada Red
          blue: '#26374A',     // "Federal Blue" (Headers/Buttons)
          dark: '#1C2B3A',     // Darker text/footer
          gray: '#F5F5F5',     // Background grays
          link: '#284162',     // Link colors
          border: '#DCDCDC'    // Input borders
        }
      },
      fontFamily: {
        sans: ['"Noto Sans"', 'Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}