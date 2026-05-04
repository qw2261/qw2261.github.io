/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#2c3e50',
        accent: '#42b883'
      },
      fontFamily: {
        sans: ['PT Sans Narrow', 'sans-serif'],
        serif: ['PT Serif', 'serif']
      }
    },
  },
  plugins: [],
}
