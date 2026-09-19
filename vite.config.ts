import { fileURLToPath, URL } from 'node:url'

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react-swc'
import { defineConfig } from 'vite'

import { staticHostingPreview } from './plugins/static-hosting-preview.ts'

export default defineConfig({
  base: '/',
  plugins: [react(), tailwindcss(), staticHostingPreview()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
})
