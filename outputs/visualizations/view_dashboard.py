#!/usr/bin/env python3
"""
Simple HTTP server to view the Harvey Dashboard visualization.

Usage:
    python view_dashboard.py

Then open http://localhost:8080 in your browser.
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 8080

# Change to the visualizations directory
os.chdir(Path(__file__).parent)

Handler = http.server.SimpleHTTPRequestHandler

# Add CORS headers for local development
class CORSHandler(Handler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

print(f"Starting server at http://localhost:{PORT}")
print("Press Ctrl+C to stop")
print()

# Open browser
webbrowser.open(f'http://localhost:{PORT}/harvey_dashboard.html')

with socketserver.TCPServer(("", PORT), CORSHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
