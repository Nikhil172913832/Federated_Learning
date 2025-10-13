#!/usr/bin/env python3
"""
Federated Learning Platform UI
A web interface to monitor and control federated learning with Docker containers
"""

import dash
from dash import dcc, html, Input, Output, State, callback, no_update, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import requests
import docker
import psutil
import time
import json
import yaml
import subprocess
import os
from datetime import datetime
import threading
import queue
import socket
from collections import deque

# Initialize Dash app with a modern theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX, dbc.icons.FONT_AWESOME])
app.title = "Federated Learning Platform"

# Global variables for data storage
container_stats = {}
federated_metrics = []
client_status = {}
training_active = False
training_process = None
log_queue = deque(maxlen=100)
metrics_history = {'time': [], 'cpu': [], 'memory': [], 'clients': []}

# Docker client
try:
    docker_client = docker.from_env()
except Exception as e:
    print(f"Warning: Could not connect to Docker: {e}")
    docker_client = None

def get_container_info():
    """Get information about running containers"""
    if not docker_client:
        return {}
    
    containers = {}
    try:
        for container in docker_client.containers.list():
            if any(name in container.name for name in ['superlink', 'supernode', 'superexec']):
                containers[container.name] = {
                    'id': container.short_id,
                    'status': container.status,
                    'image': container.image.tags[0] if container.image.tags else 'unknown',
                    'created': container.attrs['Created'],
                    'ports': container.attrs['NetworkSettings']['Ports']
                }
    except Exception as e:
        print(f"Error getting container info: {e}")
    
    return containers

def get_system_stats():
    """Get system resource statistics"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'network_sent': psutil.net_io_counters().bytes_sent / (1024**3),  # GB
        'network_recv': psutil.net_io_counters().bytes_recv / (1024**3)   # GB
    }

def get_flower_metrics():
    """Get metrics from Flower SuperLink"""
    try:
        response = requests.get('http://superlink:9093/health', timeout=2)
        if response.status_code == 200:
            return {'status': 'healthy', 'response_time': response.elapsed.total_seconds()}
    except Exception as e:
        # Fallback: treat SuperLink as healthy if TCP connection succeeds
        try:
            start = time.time()
            with socket.create_connection(("superlink", 9093), timeout=2):
                return {'status': 'healthy', 'response_time': time.time() - start}
        except Exception as _:
            pass
    
    return {'status': 'unhealthy', 'response_time': 0}

def get_mlflow_metrics():
    """Get latest metrics from MLflow"""
    try:
        mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        response = requests.get(f'{mlflow_uri}/api/2.0/mlflow/experiments/search', timeout=5)
        if response.status_code == 200:
            experiments = response.json().get('experiments', [])
            if experiments:
                # Get runs from the most recent experiment
                exp_id = experiments[0]['experiment_id']
                runs_response = requests.get(
                    f'{mlflow_uri}/api/2.0/mlflow/runs/search',
                    json={'experiment_ids': [exp_id]},
                    timeout=5
                )
                if runs_response.status_code == 200:
                    runs = runs_response.json().get('runs', [])
                    return {
                        'available': True, 
                        'count': len(experiments),
                        'runs': len(runs),
                        'latest_metrics': runs[0].get('data', {}).get('metrics', []) if runs else []
                    }
                return {'available': True, 'count': len(experiments), 'runs': 0, 'latest_metrics': []}
    except Exception as e:
        print(f"Error getting MLflow metrics: {e}")
    return {'available': False, 'count': 0, 'runs': 0, 'latest_metrics': []}

def get_container_logs():
    """Get recent logs from FL containers"""
    if not docker_client:
        return []
    
    logs = []
    try:
        for container in docker_client.containers.list():
            if any(name in container.name for name in ['superexec-clientapp', 'superexec-serverapp']):
                try:
                    # Get last 50 lines of logs
                    container_logs = container.logs(tail=50, timestamps=True).decode('utf-8', errors='ignore')
                    for line in container_logs.split('\n'):
                        if line.strip():
                            logs.append(f"[{container.name}] {line}")
                except Exception as e:
                    print(f"Error getting logs from {container.name}: {e}")
    except Exception as e:
        print(f"Error getting container logs: {e}")
    
    return logs[-100:]  # Return last 100 log lines

def load_config():
    """Load FL configuration from YAML"""
    # Try mounted config path first (in Docker), then local path
    config_paths = [
        '/config/default.yaml',  # Docker mounted volume
        os.path.join(os.path.dirname(__file__), '../complete/fl/config/default.yaml')  # Local development
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config:  # Make sure we got valid YAML
                        return config
            except Exception as e:
                print(f"Error loading config from {config_path}: {e}")
                continue
    
    print("Error: Could not load config from any path")
    return {}

def save_config(config):
    """Save FL configuration to YAML"""
    # Try mounted config path first (in Docker), then local path
    config_paths = [
        '/config/default.yaml',  # Docker mounted volume
        os.path.join(os.path.dirname(__file__), '../complete/fl/config/default.yaml')  # Local development
    ]
    
    for config_path in config_paths:
        # Try to save to the first path that exists or can be created
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Config saved to: {config_path}")
            return True
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")
            continue
    
    print("Error: Could not save config to any path")
    return False

def start_training():
    """Start federated learning training"""
    global training_active, training_process
    try:
        # Change to complete directory
        complete_dir = os.path.join(os.path.dirname(__file__), '../complete')
        
        # Start containers if not running
        subprocess.run(['docker', 'compose', '-f', 'compose-with-ui.yml', 'up', '-d'], 
                      cwd=complete_dir, check=True)
        
        time.sleep(3)  # Wait for containers to be ready
        
        # Start FL training
        training_process = subprocess.Popen(
            ['flwr', 'run', 'fl', 'local-deployment', '--stream'],
            cwd=complete_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        training_active = True
        return True, "Training started successfully!"
    except Exception as e:
        return False, f"Error starting training: {str(e)}"

def stop_training():
    """Stop federated learning training"""
    global training_active, training_process
    try:
        if training_process:
            training_process.terminate()
            training_process.wait(timeout=5)
        
        complete_dir = os.path.join(os.path.dirname(__file__), '../complete')
        subprocess.run(['docker', 'compose', '-f', 'compose-with-ui.yml', 'down'], 
                      cwd=complete_dir, check=True)
        
        training_active = False
        return True, "Training stopped successfully!"
    except Exception as e:
        return False, f"Error stopping training: {str(e)}"


# Custom CSS styles
custom_style = {
    'header': {
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'padding': '2rem',
        'borderRadius': '10px',
        'marginBottom': '2rem',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
    },
    'card': {
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'marginBottom': '1.5rem'
    }
}

# Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1([
                    html.I(className="fas fa-network-wired me-3"),
                    "Federated Learning Platform"
                ], className="text-white mb-2"),
                html.P([
                    html.I(className="fas fa-chart-line me-2"),
                    "Real-time monitoring and control of distributed machine learning"
                ], className="text-white-50 mb-0", style={'fontSize': '1.1rem'})
            ], style=custom_style['header'])
        ])
    ]),
    
    # Training Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-play-circle me-2"),
                    html.Strong("Training Control Center")
                ], className="bg-primary text-white"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="training-status-indicator", className="mb-3")
                        ], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button([
                                html.I(className="fas fa-play me-2"),
                                "Start Training"
                            ], id="start-training-btn", color="success", size="lg", className="w-100 mb-2"),
                        ], width=6),
                        dbc.Col([
                            dbc.Button([
                                html.I(className="fas fa-stop me-2"),
                                "Stop Training"
                            ], id="stop-training-btn", color="danger", size="lg", className="w-100 mb-2", disabled=True),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button([
                                html.I(className="fas fa-cog me-2"),
                                "Configure"
                            ], id="config-btn", color="info", size="lg", className="w-100", outline=True),
                        ], width=6),
                        dbc.Col([
                            dbc.Button([
                                html.I(className="fas fa-file-alt me-2"),
                                "View Logs"
                            ], id="logs-btn", color="secondary", size="lg", className="w-100", outline=True),
                        ], width=6),
                    ]),
                ])
            ], style=custom_style['card'])
        ], md=12, lg=4),
        
        # System Status Cards
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-server fa-2x text-primary mb-2"),
                                html.H3(id="cpu-stat", children="0%", className="mb-1"),
                                html.P("CPU Usage", className="text-muted mb-0")
                            ], className="text-center")
                        ])
                    ], style=custom_style['card'])
                ], width=6, md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-memory fa-2x text-success mb-2"),
                                html.H3(id="memory-stat", children="0%", className="mb-1"),
                                html.P("Memory Usage", className="text-muted mb-0")
                            ], className="text-center")
                        ])
                    ], style=custom_style['card'])
                ], width=6, md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-hdd fa-2x text-warning mb-2"),
                                html.H3(id="disk-stat", children="0%", className="mb-1"),
                                html.P("Disk Usage", className="text-muted mb-0")
                            ], className="text-center")
                        ])
                    ], style=custom_style['card'])
                ], width=6, md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-docker fa-2x text-info mb-2"),
                                html.H3(id="container-count", children="0", className="mb-1"),
                                html.P("Containers", className="text-muted mb-0")
                            ], className="text-center")
                        ])
                    ], style=custom_style['card'])
                ], width=6, md=3),
            ])
        ], md=12, lg=8),
    ]),
    
    # Metrics and Status Row
    dbc.Row([
        # Real-time System Metrics
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-chart-area me-2"),
                    html.Strong("System Performance")
                ], className="bg-dark text-white"),
                dbc.CardBody([
                    dcc.Graph(id="system-metrics-graph", config={'displayModeBar': False})
                ])
            ], style=custom_style['card'])
        ], md=12, lg=8),
        
        # SuperLink and MLflow Status
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-heartbeat me-2"),
                    html.Strong("Service Health")
                ], className="bg-dark text-white"),
                dbc.CardBody([
                    html.Div(id="service-health")
                ])
            ], style=custom_style['card'])
        ], md=12, lg=4),
    ]),
    
    # Container Status and Logs
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fab fa-docker me-2"),
                    html.Strong("Container Status")
                ], className="bg-dark text-white"),
                dbc.CardBody([
                    html.Div(id="container-table")
                ])
            ], style=custom_style['card'])
        ], md=12, lg=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-list-alt me-2"),
                    html.Strong("Training Logs")
                ], className="bg-dark text-white"),
                dbc.CardBody([
                    html.Div(id="training-logs", 
                            style={
                                'height': '300px', 
                                'overflowY': 'scroll',
                                'backgroundColor': '#1e1e1e',
                                'padding': '10px',
                                'borderRadius': '5px',
                                'fontFamily': 'monospace',
                                'fontSize': '0.9rem',
                                'color': '#d4d4d4'
                            })
                ])
            ], style=custom_style['card'])
        ], md=12, lg=6),
    ]),
    
    # Configuration Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            html.I(className="fas fa-cog me-2"),
            "Training Configuration"
        ])),
        dbc.ModalBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Number of Clients"),
                    dbc.Input(id="config-num-clients", type="number", value=10, min=1, max=100),
                ], md=6),
                dbc.Col([
                    dbc.Label("Fraction of Clients"),
                    dbc.Input(id="config-fraction", type="number", value=0.5, min=0.1, max=1.0, step=0.1),
                ], md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Learning Rate"),
                    dbc.Input(id="config-lr", type="number", value=0.01, min=0.0001, max=1.0, step=0.001),
                ], md=6),
                dbc.Col([
                    dbc.Label("Local Epochs"),
                    dbc.Input(id="config-local-epochs", type="number", value=1, min=1, max=10),
                ], md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Server Rounds"),
                    dbc.Input(id="config-server-rounds", type="number", value=3, min=1, max=100),
                ], md=6),
                dbc.Col([
                    dbc.Label("Batch Size"),
                    dbc.Input(id="config-batch-size", type="number", value=32, min=1, max=256),
                ], md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("IID Data Distribution"),
                    dbc.RadioItems(
                        id="config-iid",
                        options=[
                            {"label": "IID", "value": True},
                            {"label": "Non-IID", "value": False},
                        ],
                        value=True,
                        inline=True,
                    ),
                ], md=12),
            ]),
            html.Div(id="config-save-status", className="mt-3")
        ]),
        dbc.ModalFooter([
            dbc.Button("Save Configuration", id="config-save-btn", color="primary"),
            dbc.Button("Close", id="config-close-btn", color="secondary", className="ms-2"),
        ]),
    ], id="config-modal", size="lg", is_open=False),
    
    # Logs Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            html.I(className="fas fa-file-alt me-2"),
            "Detailed Logs"
        ])),
        dbc.ModalBody([
            html.Div(id="detailed-logs", 
                    style={
                        'height': '500px', 
                        'overflowY': 'scroll',
                        'backgroundColor': '#1e1e1e',
                        'padding': '15px',
                        'borderRadius': '5px',
                        'fontFamily': 'monospace',
                        'fontSize': '0.85rem',
                        'color': '#d4d4d4',
                        'whiteSpace': 'pre-wrap'
                    })
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="logs-close-btn", color="secondary"),
        ]),
    ], id="logs-modal", size="xl", is_open=False),
    
    # Status Toast
    dbc.Toast(
        id="status-toast",
        header="Status",
        is_open=False,
        dismissable=True,
        duration=4000,
        style={"position": "fixed", "top": 66, "right": 10, "width": 350, "zIndex": 9999},
    ),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=2*1000,  # Update every 2 seconds
        n_intervals=0
    ),
    
    # Store for training state
    dcc.Store(id='training-state', data={'active': False}),
    
], fluid=True, style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'padding': '2rem'})



# Callbacks
@app.callback(
    [Output('cpu-stat', 'children'),
     Output('memory-stat', 'children'),
     Output('disk-stat', 'children'),
     Output('container-count', 'children'),
     Output('container-table', 'children'),
     Output('service-health', 'children'),
     Output('system-metrics-graph', 'figure'),
     Output('training-logs', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    """Update dashboard with real-time data"""
    
    # Get system stats
    sys_stats = get_system_stats()
    
    # Update metrics history
    metrics_history['time'].append(datetime.now())
    metrics_history['cpu'].append(sys_stats['cpu_percent'])
    metrics_history['memory'].append(sys_stats['memory_percent'])
    
    # Keep only last 50 data points
    if len(metrics_history['time']) > 50:
        for key in metrics_history:
            metrics_history[key] = metrics_history[key][-50:]
    
    # Get container info
    containers = get_container_info()
    
    metrics_history['clients'].append(len([c for c in containers if 'supernode' in c]))
    
    # CPU, Memory, Disk stats
    cpu_stat = f"{sys_stats['cpu_percent']:.1f}%"
    memory_stat = f"{sys_stats['memory_percent']:.1f}%"
    disk_stat = f"{sys_stats['disk_percent']:.1f}%"
    container_count = str(len(containers))
    
    # Container table
    if containers:
        table_rows = []
        for name, info in containers.items():
            status_badge = dbc.Badge(
                info['status'].upper(),
                color="success" if info['status'] == 'running' else "danger",
                className="me-2"
            )
            
            # Determine container type and icon
            if 'superlink' in name.lower():
                icon = html.I(className="fas fa-server text-primary me-2")
                type_label = "SuperLink"
            elif 'supernode' in name.lower():
                icon = html.I(className="fas fa-network-wired text-success me-2")
                type_label = "SuperNode"
            elif 'superexec' in name.lower():
                if 'serverapp' in name.lower():
                    icon = html.I(className="fas fa-cogs text-info me-2")
                    type_label = "ServerApp"
                else:
                    icon = html.I(className="fas fa-cube text-warning me-2")
                    type_label = "ClientApp"
            else:
                icon = html.I(className="fas fa-box text-secondary me-2")
                type_label = "Container"
            
            table_rows.append(
                dbc.ListGroupItem([
                    dbc.Row([
                        dbc.Col([icon, html.Strong(name[:30])], width=5),
                        dbc.Col([status_badge], width=2),
                        dbc.Col([html.Small(type_label, className="text-muted")], width=3),
                        dbc.Col([html.Small(info['id'], className="text-muted")], width=2)
                    ])
                ])
            )
        container_table_html = dbc.ListGroup(table_rows, flush=True)
    else:
        container_table_html = dbc.Alert([
            html.I(className="fas fa-info-circle me-2"),
            "No containers running. Start training to see active containers."
        ], color="info")
    
    # Service health
    flower_metrics = get_flower_metrics()
    mlflow_metrics = get_mlflow_metrics()
    
    service_health_items = [
        dbc.ListGroupItem([
            dbc.Row([
                dbc.Col([
                    html.I(className="fas fa-circle me-2", 
                          style={'color': '#28a745' if flower_metrics['status'] == 'healthy' else '#dc3545'}),
                    html.Strong("SuperLink")
                ], width=6),
                dbc.Col([
                    dbc.Badge(flower_metrics['status'].upper(), 
                             color="success" if flower_metrics['status'] == 'healthy' else "danger")
                ], width=6, className="text-end")
            ]),
            html.Small(f"Response: {flower_metrics['response_time']:.3f}s", className="text-muted")
        ]),
        dbc.ListGroupItem([
            dbc.Row([
                dbc.Col([
                    html.I(className="fas fa-circle me-2", 
                          style={'color': '#28a745' if mlflow_metrics['available'] else '#dc3545'}),
                    html.Strong("MLflow")
                ], width=6),
                dbc.Col([
                    dbc.Badge("ACTIVE" if mlflow_metrics['available'] else "INACTIVE", 
                             color="success" if mlflow_metrics['available'] else "secondary")
                ], width=6, className="text-end")
            ]),
            html.Small(f"Experiments: {mlflow_metrics['count']} | Runs: {mlflow_metrics.get('runs', 0)}", className="text-muted")
        ])
    ]
    
    service_health_html = dbc.ListGroup(service_health_items, flush=True)
    
    # System metrics graph
    fig = go.Figure()
    
    if len(metrics_history['time']) > 1:
        fig.add_trace(go.Scatter(
            x=metrics_history['time'],
            y=metrics_history['cpu'],
            mode='lines',
            name='CPU %',
            line=dict(color='#667eea', width=2),
            fill='tozeroy'
        ))
        
        fig.add_trace(go.Scatter(
            x=metrics_history['time'],
            y=metrics_history['memory'],
            mode='lines',
            name='Memory %',
            line=dict(color='#764ba2', width=2),
            fill='tozeroy'
        ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            title=""
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            title="Usage %",
            range=[0, 100]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=250
    )
    
    # Training logs - get from containers
    container_logs = get_container_logs()
    
    if training_active and training_process:
        try:
            # Read new output from training process
            while True:
                line = training_process.stdout.readline()
                if not line:
                    break
                log_queue.append(line.strip())
        except:
            pass
    
    # Combine process logs and container logs
    all_logs = list(log_queue) + container_logs
    
    if all_logs:
        log_lines = [html.Div(log, style={'marginBottom': '2px', 'fontFamily': 'monospace', 'fontSize': '0.85em'}) 
                    for log in all_logs[-30:]]
    else:
        log_lines = [html.Div("No training logs available. Start training to see logs.", 
                             className="text-muted")]
    
    return cpu_stat, memory_stat, disk_stat, container_count, container_table_html, service_health_html, fig, log_lines

@app.callback(
    [Output('training-state', 'data'),
     Output('status-toast', 'is_open'),
     Output('status-toast', 'children'),
     Output('status-toast', 'header'),
     Output('status-toast', 'icon'),
     Output('start-training-btn', 'disabled'),
     Output('stop-training-btn', 'disabled')],
    [Input('start-training-btn', 'n_clicks'),
     Input('stop-training-btn', 'n_clicks')],
    [State('training-state', 'data')],
    prevent_initial_call=True
)
def control_training(start_clicks, stop_clicks, training_state):
    """Control training start/stop"""
    global training_active
    
    if not ctx.triggered:
        return no_update, False, "", "", "", False, True
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-training-btn':
        success, message = start_training()
        if success:
            training_state['active'] = True
            return (training_state, True, message, "Success", "success", 
                   True, False)
        else:
            return (training_state, True, message, "Error", "danger", 
                   False, True)
    
    elif button_id == 'stop-training-btn':
        success, message = stop_training()
        if success:
            training_state['active'] = False
            return (training_state, True, message, "Success", "success", 
                   False, True)
        else:
            return (training_state, True, message, "Error", "danger", 
                   True, False)
    
    return no_update, False, "", "", "", False, True

@app.callback(
    Output('training-status-indicator', 'children'),
    [Input('training-state', 'data')]
)
def update_training_status(training_state):
    """Update training status indicator"""
    if training_state.get('active', False):
        return dbc.Alert([
            html.I(className="fas fa-spinner fa-spin me-2"),
            html.Strong("Training in Progress"),
            html.P("Federated learning is currently running...", className="mb-0 mt-2 small")
        ], color="success")
    else:
        return dbc.Alert([
            html.I(className="fas fa-pause-circle me-2"),
            html.Strong("Training Idle"),
            html.P("Ready to start federated learning training", className="mb-0 mt-2 small")
        ], color="secondary")

@app.callback(
    Output('config-modal', 'is_open'),
    [Input('config-btn', 'n_clicks'),
     Input('config-close-btn', 'n_clicks'),
     Input('config-save-btn', 'n_clicks')],
    [State('config-modal', 'is_open')],
    prevent_initial_call=True
)
def toggle_config_modal(config_clicks, close_clicks, save_clicks, is_open):
    """Toggle configuration modal"""
    return not is_open

@app.callback(
    [Output('config-num-clients', 'value'),
     Output('config-fraction', 'value'),
     Output('config-lr', 'value'),
     Output('config-local-epochs', 'value'),
     Output('config-server-rounds', 'value'),
     Output('config-batch-size', 'value'),
     Output('config-iid', 'value')],
    [Input('config-modal', 'is_open')]
)
def load_config_values(is_open):
    """Load configuration values when modal opens"""
    if is_open:
        config = load_config()
        return (
            config.get('topology', {}).get('num_clients', 10),
            config.get('topology', {}).get('fraction', 0.5),
            config.get('train', {}).get('lr', 0.01),
            config.get('train', {}).get('local_epochs', 1),
            config.get('train', {}).get('num_server_rounds', 3),
            config.get('data', {}).get('batch_size', 32),
            config.get('data', {}).get('iid', True)
        )
    return no_update, no_update, no_update, no_update, no_update, no_update, no_update

@app.callback(
    Output('config-save-status', 'children'),
    [Input('config-save-btn', 'n_clicks')],
    [State('config-num-clients', 'value'),
     State('config-fraction', 'value'),
     State('config-lr', 'value'),
     State('config-local-epochs', 'value'),
     State('config-server-rounds', 'value'),
     State('config-batch-size', 'value'),
     State('config-iid', 'value')],
    prevent_initial_call=True
)
def save_config_values(n_clicks, num_clients, fraction, lr, local_epochs, server_rounds, batch_size, iid):
    """Save configuration values"""
    config = load_config()
    
    # Check if config was loaded successfully
    if not config:
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Error: Could not load configuration file!"
        ], color="danger")
    
    # Ensure required keys exist
    if 'topology' not in config:
        config['topology'] = {}
    if 'train' not in config:
        config['train'] = {}
    if 'data' not in config:
        config['data'] = {}
    
    # Update configuration
    config['topology']['num_clients'] = num_clients
    config['topology']['fraction'] = fraction
    config['train']['lr'] = lr
    config['train']['local_epochs'] = local_epochs
    config['train']['num_server_rounds'] = server_rounds
    config['data']['batch_size'] = batch_size
    config['data']['iid'] = iid
    
    if save_config(config):
        return dbc.Alert([
            html.I(className="fas fa-check-circle me-2"),
            "Configuration saved successfully!"
        ], color="success")
    else:
        return dbc.Alert([
            html.I(className="fas fa-exclamation-circle me-2"),
            "Error saving configuration!"
        ], color="danger")

@app.callback(
    Output('logs-modal', 'is_open'),
    [Input('logs-btn', 'n_clicks'),
     Input('logs-close-btn', 'n_clicks')],
    [State('logs-modal', 'is_open')],
    prevent_initial_call=True
)
def toggle_logs_modal(logs_clicks, close_clicks, is_open):
    """Toggle logs modal"""
    return not is_open

@app.callback(
    Output('detailed-logs', 'children'),
    [Input('logs-modal', 'is_open'),
     Input('interval-component', 'n_intervals')]
)
def update_detailed_logs(is_open, n):
    """Update detailed logs in modal"""
    if not is_open:
        return no_update
    
    # Get container logs
    container_logs = get_container_logs()
    all_logs = list(log_queue) + container_logs
    
    if all_logs:
        return "\n".join(all_logs[-200:])  # Show last 200 lines
    else:
        return "No logs available. Start training to see detailed logs."


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Federated Learning Platform UI")
    print("=" * 60)
    print("📊 Dashboard: http://localhost:8050")
    print("🌐 SuperLink API: http://localhost:9093")
    print("📈 MLflow UI: http://localhost:5000")
    print("=" * 60)
    print("\n✨ Features:")
    print("  • Real-time system monitoring")
    print("  • Start/Stop FL training from UI")
    print("  • Configure training parameters")
    print("  • View training logs")
    print("  • Monitor container status")
    print("\n🔄 Starting server...\n")
    
    app.run(host='0.0.0.0', port=8050, debug=True)

