import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  define: {
    global: 'globalThis',
    'process.env': {}
  },
  resolve: {
    alias: {
      'events': 'events',
      'styled-components': path.resolve('./node_modules/styled-components')
    }
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'redux', 'react-redux', 'react-palm']
  },
  build: {
    chunkSizeWarningLimit: 15000
  }
});
