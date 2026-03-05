# Stage 1: Build
FROM node:20-slim AS builder

WORKDIR /app

# Install build dependencies if needed
# RUN apt-get update && apt-get install -y --no-install-recommends ...

# Copy package management files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy the rest of the frontend source
COPY . .

# Build the Next.js application
RUN npm run build

# Stage 2: Production
FROM node:20-slim AS runner

WORKDIR /app

# Set environment to production
ENV NODE_ENV=production

# Copy necessary files from builder
# We only need the public folder, the static assets, and the standalone build
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# Expose the port
EXPOSE 3000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 
  CMD curl -f http://localhost:3000/api/health || exit 1

# Start the application
CMD ["node", "server.js"]
