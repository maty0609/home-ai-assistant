# Project Agent Instructions

This file provides custom instructions for the agent when working on this project.

## Frontend Testing

When running end-to-end frontend tests with Playwright:

1. **Check if servers are running first:**
   - Backend: Check with `lsof -ti:PORT` (use the backend port from `docker-compose.yml`)
   - Frontend: Check with `lsof -ti:3000` (or the port configured in `playwright.config.ts`)

2. **If servers are running:**
   - Run tests directly without starting servers
   - Example: `cd frontend && npm run test:e2e`

3. **If servers are NOT running:**
   - Start backend server first
   - For frontend, if it's a Next.js app, you can either:
     - Let Playwright start it automatically (recommended - configured in `playwright.config.ts` with `reuseExistingServer: false`)
     - Run it in background using `bg_bash` tool instead of `&` (e.g., `bg_bash "npm run dev"` or `bg_bash "npm run dev > /tmp/frontend-dev.log 2>&1"`)

4. **Timeout warning:**
   - E2E tests can take 1-5 minutes to complete
   - DO NOT run `npm run test:e2e` if you're going to wait for completion
   - If stuck or hanging, you may need to:
     - Kill the test process (Ctrl+C)
     - Check logs in `/tmp/frontend-dev.log`
     - Verify server health before retrying

5. **After tests:**
   - Check test results in `frontend/test-results/`
   - View HTML report: `npx playwright show-report`

## Backend Testing

When testing backend services:

1. Always use the virtual environment: `source .venv/bin/activate`
2. Run tests with: `pytest` or the appropriate test command
3. If testing with uvicorn, ensure you terminate after 5 seconds
