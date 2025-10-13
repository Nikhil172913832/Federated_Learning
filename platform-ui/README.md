# Platform Dashboard

Web dashboard for monitoring and controlling federated learning training.

## Features

- Start/stop training from UI
- Real-time system metrics (CPU, memory, disk)
- Container status monitoring
- Training log viewer
- Configuration editor
- MLflow integration

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

Access at http://localhost:8050

## Docker Deployment

The dashboard is included in the main platform deployment:

```bash
cd ../complete
docker compose -f compose-with-ui.yml up -d
```

## Configuration

Training parameters can be updated through the UI or by editing `complete/fl/config/default.yaml`.
# Start the UI
python app.py
```

### Access Points
- **Dashboard UI**: http://localhost:8050
- **MLflow UI**: http://localhost:5000
- **SuperLink API**: http://localhost:9093

## 🎨 UI Overview

### Main Dashboard
The dashboard is divided into several sections:

1. **Training Control Center**: Start/stop training and access configuration
2. **System Stats**: CPU, Memory, Disk usage, and container count
3. **Performance Graphs**: Real-time system metrics visualization
4. **Service Health**: SuperLink and MLflow status
5. **Container Status**: Detailed view of all running containers
6. **Training Logs**: Real-time log output from training

### Starting Training

1. Click **"Configure"** to set training parameters (optional)
2. Click **"Start Training"** to begin federated learning
3. Monitor progress in real-time through logs and metrics
4. Click **"Stop Training"** when complete

### Configuration

Click the **"Configure"** button to open the configuration modal where you can adjust:

- **Topology**:
  - Number of clients
  - Fraction of clients per round
  
- **Training**:
  - Learning rate
  - Local epochs
  - Server rounds
  
- **Data**:
  - Batch size
  - IID/Non-IID distribution

Changes are saved to `../complete/fl/config/default.yaml`

### Viewing Logs

Two ways to view logs:
1. **Dashboard View**: Last 20 log entries in the main dashboard
2. **Detailed View**: Click **"View Logs"** for full log history in a modal

## 🛠️ Technical Details

### Architecture
- **Frontend**: Dash (Plotly) with Bootstrap LUX theme
- **Backend**: Python with Flask
- **Monitoring**: Docker SDK, psutil
- **Config Management**: PyYAML

### Key Components

#### Real-time Updates
- Dashboard updates every 2 seconds
- Automatic log streaming from training process
- Live container status monitoring

#### Training Control
- Manages Docker Compose services
- Executes `flwr run` commands
- Captures and displays output in real-time

#### Configuration
- Reads/writes YAML configuration
- Validates input parameters
- Applies changes immediately

## 🎯 Usage Tips

1. **Before Training**: Review configuration settings
2. **During Training**: Monitor system resources and logs
3. **After Training**: Check MLflow for detailed metrics
4. **Container Issues**: View container status for debugging

## 🐛 Troubleshooting

### Training Won't Start
- Ensure Docker is running
- Check if ports 9092, 9093, 5000 are available
- Verify `flwr` CLI is installed

### No Containers Visible
- Click "Start Training" to launch containers
- Check Docker daemon status
- Review detailed logs for errors

### Configuration Not Saving
- Ensure write permissions on config file
- Check YAML syntax in manual edits
- Verify file path in `load_config()` function

## 📝 Development

To modify the UI:

1. Edit `app.py` for layout/callbacks
2. Add new metrics in `get_*_metrics()` functions
3. Update styles in `custom_style` dictionary
4. Test changes with `python app.py`

## 🤝 Contributing

Improvements welcome! Focus areas:
- Additional metrics visualization
- MLflow integration enhancements
- Advanced training controls
- Mobile responsiveness

## 📄 License

Same as the main Federated Learning project.

## 🔗 Related

- Main project: `../README.md`
- FL implementation: `../complete/fl/`
- Docker setup: `../complete/compose-with-ui.yml`
