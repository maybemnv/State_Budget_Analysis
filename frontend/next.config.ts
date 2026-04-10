import type { NextConfig } from "next";
import fs from 'fs';
import path from 'path';

const envPath = path.resolve(__dirname, '../.env');
const env: Record<string, string> = {};

if (fs.existsSync(envPath)) {
  const envFile = fs.readFileSync(envPath, 'utf8');
  envFile.split('\n').forEach((line: string) => {
    const [key, value] = line.split('=');
    if (key && value && key.trim().startsWith('NEXT_PUBLIC_')) {
      env[key.trim()] = value.trim();
    }
  });
}

const nextConfig: NextConfig = {
  /* config options here */
  output: "standalone",
  env: env,
};

export default nextConfig;
