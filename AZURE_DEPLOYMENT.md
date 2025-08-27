# Azure App Service Deployment Guide

This guide provides step-by-step instructions for deploying VeriDoc AI to Azure App Service using GitHub Actions.

## Prerequisites

- Azure subscription
- Azure CLI installed and configured
- GitHub repository with the project
- OpenAI API key

## Step 1: Azure Infrastructure Setup

### 1.1 Install Azure CLI
```bash
# Windows (using winget)
winget install Microsoft.AzureCLI

# Or download from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
```

### 1.2 Login to Azure
```bash
az login
```

### 1.3 Create Resource Group
```bash
az group create --name veridoc-rg --location eastus
```

### 1.4 Create App Service Plan
```bash
az appservice plan create \
  --name veridoc-plan \
  --resource-group veridoc-rg \
  --sku B1 \
  --is-linux
```

### 1.5 Create Web Apps
```bash
# Create staging web app
az webapp create \
  --name veridoc-staging \
  --resource-group veridoc-rg \
  --plan veridoc-plan \
  --runtime "PYTHON|3.11"

# Create production web app
az webapp create \
  --name veridoc-prod \
  --resource-group veridoc-rg \
  --plan veridoc-plan \
  --runtime "PYTHON|3.11"
```

### 1.6 Configure App Settings
```bash
# Configure staging environment
az webapp config appsettings set \
  --name veridoc-staging \
  --resource-group veridoc-rg \
  --settings \
    WEBSITES_PORT=8001 \
    SCM_DO_BUILD_DURING_DEPLOYMENT=true \
    PYTHON_VERSION=3.11

# Configure production environment
az webapp config appsettings set \
  --name veridoc-prod \
  --resource-group veridoc-rg \
  --settings \
    WEBSITES_PORT=8001 \
    SCM_DO_BUILD_DURING_DEPLOYMENT=true \
    PYTHON_VERSION=3.11
```

## Step 2: Create Service Principal

### 2.1 Generate Service Principal
```bash
az ad sp create-for-rbac \
  --name "veridoc-deploy" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/veridoc-rg \
  --sdk-auth
```

**Important**: Replace `{subscription-id}` with your actual Azure subscription ID.

### 2.2 Get Subscription ID
```bash
az account show --query id --output tsv
```

### 2.3 Example Output
The command will output JSON similar to this:
```json
{
  "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "subscriptionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

**Save this JSON output** - you'll need it for GitHub secrets.

## Step 3: Configure GitHub Secrets

### 3.1 Access Repository Settings
1. Go to your GitHub repository: `https://github.com/oleggninenko/VeriDoc`
2. Click on "Settings" tab
3. In the left sidebar, click "Secrets and variables" → "Actions"

### 3.2 Add Required Secrets

#### AZURE_CREDENTIALS
- **Name**: `AZURE_CREDENTIALS`
- **Value**: The entire JSON output from the service principal creation
- **Type**: Secret

#### Azure App Service Names
- **Name**: `AZURE_APP_SERVICE_NAME_STAGING`
- **Value**: `veridoc-staging`
- **Type**: Secret

- **Name**: `AZURE_APP_SERVICE_NAME_PRODUCTION`
- **Value**: `veridoc-prod`
- **Type**: Secret

#### Azure Resource Group
- **Name**: `AZURE_RESOURCE_GROUP`
- **Value**: `veridoc-rg`
- **Type**: Secret

#### OpenAI Configuration
- **Name**: `OPENAI_API_KEY`
- **Value**: Your OpenAI API key (starts with `sk-...`)
- **Type**: Secret

- **Name**: `OPENAI_BASE_URL`
- **Value**: `https://api.openai.com/v1` (or your custom endpoint)
- **Type**: Secret

## Step 4: Test Deployment

### 4.1 Push Changes
```bash
git add .
git commit -m "Configure Azure App Service deployment"
git push origin main
```

### 4.2 Monitor Deployment
1. Go to your GitHub repository
2. Click on "Actions" tab
3. You should see the CI/CD pipeline running
4. Monitor the deployment progress

### 4.3 Access Your Application
- **Staging**: `https://veridoc-staging.azurewebsites.net`
- **Production**: `https://veridoc-prod.azurewebsites.net`

## Step 5: Environment-Specific Configuration

### 5.1 Staging Environment
```bash
# Set environment-specific settings for staging
az webapp config appsettings set \
  --name veridoc-staging \
  --resource-group veridoc-rg \
  --settings \
    ENVIRONMENT=staging \
    LOG_LEVEL=DEBUG
```

### 5.2 Production Environment
```bash
# Set environment-specific settings for production
az webapp config appsettings set \
  --name veridoc-prod \
  --resource-group veridoc-rg \
  --settings \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO
```

## Step 6: Custom Domain (Optional)

### 6.1 Add Custom Domain
```bash
# Add custom domain to staging
az webapp config hostname add \
  --webapp-name veridoc-staging \
  --resource-group veridoc-rg \
  --hostname staging.yourdomain.com

# Add custom domain to production
az webapp config hostname add \
  --webapp-name veridoc-prod \
  --resource-group veridoc-rg \
  --hostname app.yourdomain.com
```

### 6.2 Configure SSL
```bash
# Bind SSL certificate
az webapp config ssl bind \
  --certificate-thumbprint {thumbprint} \
  --ssl-type SNI \
  --name veridoc-prod \
  --resource-group veridoc-rg
```

## Troubleshooting

### Common Issues

#### 1. Deployment Fails
- Check GitHub Actions logs for detailed error messages
- Verify all secrets are correctly configured
- Ensure service principal has proper permissions

#### 2. Application Won't Start
- Check Azure App Service logs: `az webapp log tail --name veridoc-prod --resource-group veridoc-rg`
- Verify startup command in `startup.txt`
- Check environment variables are set correctly

#### 3. API Key Issues
- Verify `OPENAI_API_KEY` secret is set correctly
- Check API key permissions and quota
- Test API key locally first

#### 4. Port Configuration
- Ensure `WEBSITES_PORT=8000` is set in app settings
- Verify the application listens on the correct port

### Useful Commands

#### View App Service Logs
```bash
# Stream logs
az webapp log tail --name veridoc-prod --resource-group veridoc-rg

# Download logs
az webapp log download --name veridoc-prod --resource-group veridoc-rg
```

#### Restart App Service
```bash
az webapp restart --name veridoc-prod --resource-group veridoc-rg
```

#### Check App Settings
```bash
az webapp config appsettings list --name veridoc-prod --resource-group veridoc-rg
```

## Cost Optimization

### 1. Use Appropriate SKU
- **Development**: F1 (Free) or B1 (Basic)
- **Production**: S1 (Standard) or higher for better performance

### 2. Enable Auto-scaling
```bash
az monitor autoscale create \
  --resource-group veridoc-rg \
  --resource veridoc-prod \
  --resource-type Microsoft.Web/sites \
  --name veridoc-autoscale \
  --min-count 1 \
  --max-count 3 \
  --count 1
```

### 3. Monitor Usage
- Use Azure Monitor to track resource usage
- Set up alerts for cost thresholds
- Review and optimize resource allocation

## Security Best Practices

### 1. Network Security
- Use Azure Application Gateway for additional security
- Configure IP restrictions if needed
- Enable HTTPS-only traffic

### 2. Secrets Management
- Use Azure Key Vault for sensitive configuration
- Rotate service principal credentials regularly
- Never commit secrets to source code

### 3. Monitoring
- Enable Application Insights for monitoring
- Set up alerts for application errors
- Monitor authentication and authorization

## Next Steps

1. **Set up monitoring** with Application Insights
2. **Configure backup** for your App Service
3. **Set up staging/production environments** with proper branching strategy
4. **Implement blue-green deployment** for zero-downtime updates
5. **Add custom domain and SSL certificate**

---

For additional support, refer to:
- [Azure App Service Documentation](https://docs.microsoft.com/en-us/azure/app-service/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Azure CLI Documentation](https://docs.microsoft.com/en-us/cli/azure/)
