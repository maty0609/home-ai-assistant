# CI/CD Pipeline Documentation

This repository uses GitHub Actions for continuous integration and deployment, inspired by the [Ghostty](https://github.com/ghostty-org/ghostty) project structure.

## Workflows

### 1. CI (`ci.yml`)
The main continuous integration workflow that runs on every push and pull request.

**Jobs:**
- **Lint**: Code formatting and linting checks for both Python and TypeScript
- **Test Backend**: Unit tests for the FastAPI backend with PostgreSQL service
- **Test Frontend**: Unit tests and build verification for the Next.js frontend
- **Docker**: Docker image building and basic health checks
- **Security**: Security scanning with Safety, Bandit, and npm audit
- **Integration**: End-to-end integration tests
- **Build and Push**: Docker image publishing to Docker Hub (main branch only)

### 2. Deploy (`deploy.yml`)
Deployment workflow that runs after successful CI completion.

**Jobs:**
- **Deploy Staging**: Deploy to staging environment
- **Deploy Production**: Deploy to production environment (after staging)
- **Notify**: Send deployment notifications to Slack

### 3. Release (`release.yml`)
Release workflow triggered by version tags.

**Features:**
- Automatic changelog generation from git commits
- GitHub release creation
- Docker image tagging with version numbers

## Setup Requirements

### GitHub Secrets
Configure these secrets in your repository settings:

```bash
# Docker Hub credentials
DOCKERHUB_USERNAME=your-dockerhub-username
DOCKERHUB_TOKEN=your-dockerhub-access-token

# Slack notifications (optional)
SLACK_WEBHOOK=your-slack-webhook-url

# Environment-specific secrets
STAGING_API_KEY=your-staging-api-key
PRODUCTION_API_KEY=your-production-api-key
```

### Environment Protection
Set up environment protection rules for `staging` and `production`:
1. Go to Settings → Environments
2. Create environments: `staging` and `production`
3. Add protection rules (required reviewers, deployment branches)

## Usage

### Running Tests Locally
```bash
# Backend tests
cd backend
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx
python -m pytest tests/ -v

# Frontend tests
cd frontend
npm ci
npm test
npm run build
```

### Creating a Release
```bash
# Create and push a new tag
git tag v1.0.0
git push origin v1.0.0
```

### Manual Workflow Trigger
You can manually trigger workflows from the Actions tab in GitHub.

## Customization

### Adding New Jobs
1. Edit the appropriate workflow file in `.github/workflows/`
2. Add your job under the `jobs` section
3. Configure dependencies using `needs:`

### Environment Variables
- Use `env` section for workflow-level variables
- Use `secrets` for sensitive data
- Use `${{ env.VARIABLE_NAME }}` syntax in steps

### Caching
- Python: Uses pip cache automatically
- Node.js: Configured with npm cache
- Docker: Uses GitHub Actions cache for layer caching

## Troubleshooting

### Common Issues

1. **Database Connection Failures**
   - Ensure PostgreSQL service is properly configured
   - Check database credentials in secrets

2. **Docker Build Failures**
   - Verify Dockerfile syntax
   - Check for missing dependencies

3. **Test Failures**
   - Run tests locally first
   - Check for environment-specific issues

### Debugging
- Enable debug logging by setting `ACTIONS_STEP_DEBUG=true` in repository secrets
- Check workflow run logs for detailed error messages
- Use `if: always()` to continue workflow even if some steps fail

## Best Practices

1. **Branch Protection**: Enable branch protection rules for main branch
2. **Code Review**: Require pull request reviews before merging
3. **Testing**: Write comprehensive tests for all new features
4. **Security**: Regularly update dependencies and run security scans
5. **Documentation**: Keep this README updated with any changes

## Contributing

When contributing to the CI/CD pipeline:

1. Test changes locally first
2. Create a feature branch
3. Submit a pull request with detailed description
4. Ensure all checks pass before merging
5. Update documentation as needed 