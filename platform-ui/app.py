#!/usr/bin/env python3
"""
Federated Learning Platform UI
A web interface to monitor and showcase federated learning with Docker containers
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import requests
import docker
import psutil
import time
import json
from datetime import datetime
import threading
import queue
import socket

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Federated Learning Platform"

# Global variables for data storage
container_stats = {}
federated_metrics = []
client_status = {}

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
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent
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

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🚀 Federated Learning Platform", className="text-center mb-4"),
            html.P("Real-time monitoring of federated learning with Docker container clients", 
                   className="text-center text-muted mb-4")
        ])
    ]),
    
    # System Overview Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📊 System Resources", className="card-title"),
                    html.Div(id="system-stats")
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🌐 SuperLink Status", className="card-title"),
                    html.Div(id="superlink-status")
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🐳 Active Containers", className="card-title"),
                    html.Div(id="container-count")
                ])
            ])
        ], width=4)
    ], className="mb-4"),
    
    # Container Status Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🐳 Container Status", className="card-title"),
                    html.Div(id="container-table")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Real-time Metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📈 Real-time Metrics", className="card-title"),
                    dcc.Graph(id="metrics-graph")
                ])
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📋 Client Status", className="card-title"),
                    html.Div(id="client-status")
                ])
            ])
        ], width=4)
    ], className="mb-4"),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🎮 Platform Controls", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("🔄 Refresh", id="refresh-btn", color="primary", className="me-2"),
                            dbc.Button("📊 View Logs", id="logs-btn", color="info", className="me-2"),
                            dbc.Button("🎯 Start FL", id="start-fl-btn", color="success", className="me-2"),
                        ])
                    ]),
                    html.Hr(),
                    html.Div(id="control-output")
                ])
            ])
        ], width=12)
    ]),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=2*1000,  # Update every 2 seconds
        n_intervals=0
    )
], fluid=True)

# Callbacks
@app.callback(
    [Output('system-stats', 'children'),
     Output('superlink-status', 'children'),
     Output('container-count', 'children'),
     Output('container-table', 'children'),
     Output('client-status', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    """Update dashboard with real-time data"""
    
    # Get system stats
    sys_stats = get_system_stats()
    system_stats_html = html.Div([
        html.P(f"CPU: {sys_stats['cpu_percent']:.1f}%"),
        html.P(f"Memory: {sys_stats['memory_percent']:.1f}%"),
        html.P(f"Disk: {sys_stats['disk_percent']:.1f}%")
    ])
    
    # Get SuperLink status
    flower_metrics = get_flower_metrics()
    superlink_status_html = html.Div([
        html.P(f"Status: {flower_metrics['status'].upper()}", 
               className="text-success" if flower_metrics['status'] == 'healthy' else "text-danger"),
        html.P(f"Response: {flower_metrics['response_time']:.3f}s")
    ])
    
    # Get container info
    containers = get_container_info()
    container_count_html = html.Div([
        html.H2(f"{len(containers)}", className="text-primary"),
        html.P("Active containers")
    ])
    
    # Container table
    if containers:
        table_rows = []
        for name, info in containers.items():
            status_color = "success" if info['status'] == 'running' else "danger"
            table_rows.append(
                dbc.Row([
                    dbc.Col(html.Strong(name), width=3),
                    dbc.Col(html.Span(info['status'], className=f"badge bg-{status_color}"), width=2),
                    dbc.Col(info['image'], width=4),
                    dbc.Col(info['id'], width=3)
                ], className="mb-2")
            )
        container_table_html = html.Div(table_rows)
    else:
        container_table_html = html.P("No containers found", className="text-muted")
    
    # Client status
    client_containers = {k: v for k, v in containers.items() if 'clientapp' in k}
    if client_containers:
        client_status_html = html.Div([
            html.H5(f"{len(client_containers)} Client Containers"),
            html.Ul([
                html.Li(f"{name}: {info['status']}", 
                       className="text-success" if info['status'] == 'running' else "text-danger")
                for name, info in client_containers.items()
            ])
        ])
    else:
        client_status_html = html.P("No client containers running", className="text-muted")
    
    return system_stats_html, superlink_status_html, container_count_html, container_table_html, client_status_html

@app.callback(
    Output('metrics-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_metrics_graph(n):
    """Update metrics graph"""
    
    # Get current time
    now = datetime.now()
    
    # Get system stats
    sys_stats = get_system_stats()
    
    # Create figure
    fig = go.Figure()
    
    # Add CPU usage
    fig.add_trace(go.Scatter(
        x=[now],
        y=[sys_stats['cpu_percent']],
        mode='lines+markers',
        name='CPU %',
        line=dict(color='blue')
    ))
    
    # Add memory usage
    fig.add_trace(go.Scatter(
        x=[now],
        y=[sys_stats['memory_percent']],
        mode='lines+markers',
        name='Memory %',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title="System Resource Usage",
        xaxis_title="Time",
        yaxis_title="Percentage",
        height=300
    )
    
    return fig

@app.callback(
    Output('control-output', 'children'),
    [Input('refresh-btn', 'n_clicks'),
     Input('logs-btn', 'n_clicks'),
     Input('start-fl-btn', 'n_clicks')]
)
def handle_controls(refresh_clicks, logs_clicks, start_clicks):
    """Handle control button clicks"""
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'refresh-btn':
        return dbc.Alert("Dashboard refreshed!", color="info", duration=2000)
    elif button_id == 'logs-btn':
        return dbc.Alert("Logs feature coming soon!", color="info", duration=2000)
    elif button_id == 'start-fl-btn':
        return dbc.Alert("Federated learning start feature coming soon!", color="success", duration=2000)
    
    return ""

if __name__ == '__main__':
    print("🚀 Starting Federated Learning Platform UI...")
    print("📊 Dashboard will be available at: http://localhost:8050")
    print("🌐 SuperLink API: http://localhost:9093")
    print("📈 MLflow UI: http://localhost:5000")
    app.run(host='0.0.0.0', port=8050, debug=True)
