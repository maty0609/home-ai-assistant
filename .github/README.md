# CI/CD Pipeline

This repository uses GitHub Actions for continuous integration and deployment.

## Workflows

### 1. CI (`ci.yml`)
Runs on every push and pull request to `main` and `develop` branches.

**Jobs:**
- **Backend**: Python linting, type checking, and tests
- **Frontend**: Node.js linting, type checking, and build
- **Docker**: Build and test Docker images
- **Integration**: End-to-end tests
- **Security**: Vulnerability scanning with Trivy
- **Dependencies**: Security audit of dependencies

### 2. Deploy (`deploy.yml`)
Runs on pushes to `main` and when tags are created.

**Jobs:**
- **Build and Push**: Build Docker images and push to GitHub Container Registry
- **Deploy Staging**: Deploy to staging environment
- **Deploy Production**: Deploy to production environment (only on tags)
- **Notify**: Send deployment notifications

### 3. Dependencies (`dependencies.yml`)
Runs weekly and manually to update dependencies.

**Jobs:**
- **Update Python**: Update Python dependencies and create PR
- **Update Node.js**: Update Node.js dependencies and create PR
- **Security Audit**: Audit updated dependencies

## Environment Variables

Set these secrets in your GitHub repository:

### Required for CI:
- `GITHUB_TOKEN` (automatically provided)

### Required for Deployment:
- `SLACK_WEBHOOK` (optional, for notifications)

### Required for Dependabot:
- None (uses `GITHUB_TOKEN`)

## Local Development

### Running Tests
```bash
# Backend tests
cd backend
pip install pytest pytest-cov
pytest

# Frontend tests
cd frontend
npm test
```

### Running Linters
```bash
# Backend linting
cd backend
black .
flake8 .
mypy .

# Frontend linting
cd frontend
npm run lint
```

## Deployment

This ie being work in progress. Not fully function yet.

### Staging
- Automatically deploys on pushes to `main`
- Can be triggered manually via workflow dispatch

### Production
- Deploys on tag creation (e.g., `v1.0.0`)
- Can be triggered manually via workflow dispatch

## Security

- **Trivy**: Scans for vulnerabilities in dependencies and Docker images
- **Safety**: Audits Python dependencies
- **npm audit**: Audits Node.js dependencies
- **CodeQL**: Static analysis for security issues

## Monitoring

- **Code Coverage**: Reports uploaded to Codecov
- **Security Findings**: Uploaded to GitHub Security tab
- **Deployment Status**: Notifications sent to Slack (if configured)

## Customization

1. Update environment names in `deploy.yml`
2. Add your deployment logic in the deploy steps
3. Configure Slack webhook for notifications
4. Update Dependabot configuration with your username
5. Add more test files in `backend/tests/` 