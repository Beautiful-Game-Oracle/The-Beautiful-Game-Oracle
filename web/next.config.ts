import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    // @ts-expect-error - outputFileTracingIncludes is valid but missing from types in this version
    outputFileTracingIncludes: {
      "/api/**/*": ["./public/**/*"],
    },
  },
};

export default nextConfig;
