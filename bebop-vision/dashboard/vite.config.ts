import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev mode proxies /api to the FastAPI backend
// (python tools/dashboard_api.py --port 8099).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.DASHBOARD_API ?? "http://127.0.0.1:8099",
        changeOrigin: true,
      },
    },
  },
});
