import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// In dev the API runs separately (`refiner serve`); proxy /api to it.
// In production the Sanic app serves web/dist itself, so relative /api URLs keep working.
export default defineConfig({
  plugins: [svelte()],
  server: {
    port: 5173,
    proxy: { '/api': { target: process.env.REFINER_API ?? 'http://127.0.0.1:8000', changeOrigin: true } },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
