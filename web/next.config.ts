import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    // @ts-expect-error - outputFileTracingIncludes is valid but missing from types in this version
    outputFileTracingIncludes: {
      "/api/**/*": [
        "./public/**/*",
        "./node_modules/onnxruntime-node/bin/napi-v6/linux/x64/libonnxruntime.so.1",
      ],
    },
  },
  serverComponentsExternalPackages: ["onnxruntime-node"],
};

export default nextConfig;
