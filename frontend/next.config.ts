import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  // Suppress the deprecated middleware warning
  // See: https://nextjs.org/docs/messages/middleware-to-proxy
  compiler: {
    // This won't suppress the middleware warning, but it's here for reference
    removeConsole: process.env.NODE_ENV === 'production',
  },
};

export default nextConfig;
