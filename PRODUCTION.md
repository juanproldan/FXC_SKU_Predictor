# SKU Predictor - Production Deployment Guide

This document provides instructions for deploying the SKU Predictor application to a production environment.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)
- A server or cloud instance with sufficient resources (recommended: 2+ CPU cores, 4+ GB RAM)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/juanproldan/FXC_SKU_Predictor.git
   cd FXC_SKU_Predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Set environment variables:
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your-secure-secret-key  # Change this to a secure random string
   ```

2. Create necessary directories:
   ```bash
   mkdir -p data/models logs
   ```

## Deployment Options

### Option 1: Using the Deployment Script

The simplest way to deploy the application is to use the provided deployment script:

```bash
python deploy.py --host 0.0.0.0 --port 5000
```

This script will:
- Install dependencies
- Set up the environment
- Start the application in production mode

### Option 2: Manual Deployment

If you prefer to deploy manually, follow these steps:

1. Start the application in production mode:
   ```bash
   python run.py web --production --host 0.0.0.0 --port 5000
   ```

### Option 3: Using a Process Manager (Recommended)

For production environments, it's recommended to use a process manager like Supervisor or systemd.

#### Using Supervisor

1. Install Supervisor:
   ```bash
   pip install supervisor
   ```

2. Create a Supervisor configuration file (`/etc/supervisor/conf.d/sku_predictor.conf`):
   ```ini
   [program:sku_predictor]
   command=python /path/to/FXC_SKU_Predictor/run.py web --production --host 0.0.0.0 --port 5000
   directory=/path/to/FXC_SKU_Predictor
   user=your_user
   autostart=true
   autorestart=true
   stopasgroup=true
   killasgroup=true
   stderr_logfile=/path/to/FXC_SKU_Predictor/logs/supervisor-err.log
   stdout_logfile=/path/to/FXC_SKU_Predictor/logs/supervisor-out.log
   environment=FLASK_ENV=production,SECRET_KEY="your-secure-secret-key"
   ```

3. Update Supervisor and start the application:
   ```bash
   supervisorctl reread
   supervisorctl update
   supervisorctl start sku_predictor
   ```

#### Using systemd

1. Create a systemd service file (`/etc/systemd/system/sku_predictor.service`):
   ```ini
   [Unit]
   Description=SKU Predictor Service
   After=network.target

   [Service]
   User=your_user
   WorkingDirectory=/path/to/FXC_SKU_Predictor
   ExecStart=/usr/bin/python /path/to/FXC_SKU_Predictor/run.py web --production --host 0.0.0.0 --port 5000
   Restart=always
   Environment=FLASK_ENV=production SECRET_KEY=your-secure-secret-key

   [Install]
   WantedBy=multi-user.target
   ```

2. Enable and start the service:
   ```bash
   systemctl enable sku_predictor
   systemctl start sku_predictor
   ```

## Using a Web Server

For production environments, it's recommended to use a web server like Nginx or Apache as a reverse proxy in front of the application.

### Nginx Configuration Example

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoring

It's important to monitor the application in production. You can use the built-in monitoring tools or integrate with external monitoring services.

### Built-in Monitoring

The application includes a monitoring command that checks the logs for errors:

```bash
python run.py monitor --hours 24 --notify --continuous --interval 3600
```

This will check the logs every hour and send notifications if errors are found.

### External Monitoring

You can also integrate with external monitoring services like Prometheus, Grafana, or New Relic.

## Backup

Regular backups of the feedback database are recommended:

```bash
python run.py backup
```

This will create a backup of the feedback database in the `data/backups` directory.

## Troubleshooting

If you encounter issues with the deployment, check the logs in the `logs` directory:

```bash
tail -f logs/app.log
```

For more detailed information, you can also check the production logs:

```bash
tail -f logs/app-YYYYMMDD.log
```

## Security Considerations

1. Always use HTTPS in production
2. Set a secure SECRET_KEY
3. Keep dependencies up to date
4. Implement proper access controls
5. Regularly backup the feedback database

## Updating the Application

To update the application:

1. Pull the latest changes:
   ```bash
   git pull origin master
   ```

2. Install any new dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Restart the application:
   ```bash
   # If using Supervisor
   supervisorctl restart sku_predictor
   
   # If using systemd
   systemctl restart sku_predictor
   
   # If running manually
   # First stop the current process, then start a new one
   python run.py web --production --host 0.0.0.0 --port 5000
   ```
