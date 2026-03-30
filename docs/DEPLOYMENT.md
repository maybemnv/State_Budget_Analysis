# Deployment Guide

Production deployment instructions for DataLens AI.

## Overview

DataLens AI consists of:
- **Backend**: FastAPI (Python) with PostgreSQL and Redis
- **Frontend**: Next.js 14 (React + TypeScript)
- **Infrastructure**: Docker, cloud PostgreSQL, Redis/Upstash

## Prerequisites

- Docker and Docker Compose
- PostgreSQL 15+ database (managed or self-hosted)
- Redis instance (local, cloud, or Upstash)
- Google Gemini API key
- Domain name (for production)

## Environment Variables

Create `.env.production`:

```env
# Required API Keys
GEMINI_API_KEY=your_gemini_api_key

# Database (use connection string or individual vars)
DB_HOST=your-db-host.amazonaws.com
DB_PORT=5432
DB_USER=datalens
DB_PASSWORD=your_secure_password
DB_NAME=datalens_production

# Redis (choose one)
# Option 1: Standard Redis
REDIS_URL=redis://your-redis-host:6379/0

# Option 2: Upstash (recommended for serverless)
UPSTASH_REDIS_REST_URL=https://your-url.upstash.io
UPSTASH_REDIS_REST_TOKEN=your_token

# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
MAX_UPLOAD_MB=100
SESSION_TTL_SECONDS=7200

# CORS (restrict to your domain)
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

## Docker Deployment

### Single Server

1. **Clone and configure:**
```bash
git clone https://github.com/your-org/datalens-ai.git
cd datalens-ai
cp .env.example .env
# Edit .env with production values
```

2. **Build and run:**
```bash
docker-compose -f docker-compose.prod.yaml up -d
```

3. **Verify:**
```bash
curl http://localhost:8000/health
```

### Docker Compose Production

The `docker-compose.prod.yaml` includes:
- Backend API container
- Frontend Next.js container (static export or SSR)
- Nginx reverse proxy (SSL termination)
- Optional: PostgreSQL and Redis containers

```yaml
# Key production settings
services:
  backend:
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
      - WORKERS=4  # Uvicorn workers
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
    restart: unless-stopped
    
  frontend:
    environment:
      - NODE_ENV=production
    restart: unless-stopped
    
  nginx:
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./ssl:/etc/nginx/ssl:ro
```

## Cloud Deployment

### Railway/Render (Easy)

1. Connect GitHub repo
2. Add environment variables in dashboard
3. Deploy automatically on push

### AWS ECS/Fargate

```bash
# Build image
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URL
docker build -t datalens-backend ./backend
docker tag datalens-backend:latest $ECR_URL/datalens-backend:latest
docker push $ECR_URL/datalens-backend:latest

# Deploy to ECS
aws ecs update-service --cluster datalens --service backend --force-new-deployment
```

### Google Cloud Run

```bash
# Deploy backend
gcloud run deploy datalens-backend \
  --source ./backend \
  --set-env-vars GEMINI_API_KEY=xxx,DB_HOST=xxx \
  --max-instances=10 \
  --memory=2Gi

# Deploy frontend
gcloud run deploy datalens-frontend \
  --source ./frontend \
  --set-env-vars NEXT_PUBLIC_API_URL=https://datalens-backend-url
```

### Vercel (Frontend only)

```bash
cd frontend
vercel --prod
```

## Database Migration

For production database setup:

```bash
# Run migrations (if using Alembic)
cd backend
alembic upgrade head

# Or initialize fresh
docker-compose exec backend python -c "from backend.db import init_db; import asyncio; asyncio.run(init_db())"
```

## SSL/TLS

### Let's Encrypt with Certbot

```bash
# On server with nginx
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

### Cloudflare (recommended)

1. Add site to Cloudflare
2. Enable proxy (orange cloud)
3. SSL/TLS mode: Full (strict)
4. Always Use HTTPS: On

## Monitoring

### Health Checks

Configure uptime monitoring for:
- `GET /health` — API health
- WebSocket connection test
- Database connectivity check

### Logging

Structured JSON logging is enabled in production:

```json
{
  "timestamp": "2026-03-30T14:30:00Z",
  "level": "INFO",
  "logger": "backend.routes.chat",
  "message": "WebSocket accepted",
  "session_id": "sess_abc123",
  "request_id": "req_xyz789"
}
```

### Recommended Tools

| Tool | Purpose |
|------|---------|
| Datadog / New Relic | APM and tracing |
| Grafana + Prometheus | Metrics and dashboards |
| Sentry | Error tracking |
| PagerDuty | Alerting |

### Key Metrics to Monitor

- Request latency (p50, p95, p99)
- Error rate (4xx, 5xx)
- Database connection pool usage
- Redis cache hit rate
- Agent execution time
- WebSocket connection count

## Security Checklist

- [ ] Use strong database passwords (32+ chars)
- [ ] Enable SSL/TLS for all connections
- [ ] Restrict CORS to production domain only
- [ ] Set `DEBUG=false` and `ENVIRONMENT=production`
- [ ] Use secrets manager for API keys (AWS SSM, GCP Secret Manager)
- [ ] Enable database encryption at rest
- [ ] Configure firewall rules (port 443 only)
- [ ] Set up log aggregation and retention
- [ ] Enable rate limiting
- [ ] Run security scans (Snyk, Trivy)

## Backup Strategy

### Database

```bash
# Automated daily backups (PostgreSQL)
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d).sql

# Or use managed DB automated backups
```

### Sessions

Sessions are ephemeral by design (TTL: 2 hours). No backup needed for session data.

## Scaling

### Horizontal Scaling

```yaml
# docker-compose scale
services:
  backend:
    deploy:
      replicas: 3
```

### Load Balancer Config

```nginx
upstream backend {
    least_conn;
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    location /ws/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Troubleshooting

### Common Issues

**Database connection errors:**
```bash
# Check connectivity
docker-compose exec backend pg_isready -h $DB_HOST -p 5432

# Check migrations
alembic current
```

**Redis connection errors:**
```bash
# Test Redis
redis-cli -u $REDIS_URL ping
```

**High memory usage:**
- Reduce `MAX_UPLOAD_MB`
- Decrease `SESSION_TTL_SECONDS`
- Enable swap or increase container memory

## Rollback

```bash
# Rollback to previous Docker image
docker-compose -f docker-compose.prod.yaml pull
docker-compose -f docker-compose.prod.yaml up -d --no-deps backend

# Database rollback (if migration failed)
alembic downgrade -1
```
