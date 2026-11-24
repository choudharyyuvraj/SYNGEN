import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// During development the UI runs on :5173 and forwards /api calls to the
// FastAPI server on :8000 (python server/app.py).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
    fs: { allow: ['..'] }, // lets the Gallery import images from ../assets
  },
})
