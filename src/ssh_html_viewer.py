#!/usr/bin/env python3
"""
Simple HTML file viewer server for accessing HTML files via Tailscale
"""
import http.server
import os
from pathlib import Path
import socketserver
import subprocess
from urllib.parse import quote

PORT = 8080
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class HTMLFileHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to serve HTML files with directory listing"""

    def list_directory(self, path):
        """Override to create a custom directory listing focused on HTML files"""
        try:
            # If it's the root path, show custom HTML index
            if self.path == '/':
                return self.generate_html_index()

            # Otherwise, use default directory listing
            return super().list_directory(path)
        except OSError:
            self.send_error(404, "File not found")
            return None

    def generate_html_index(self):
        """Generate an index page showing all HTML files organized by directory"""
        import io

        # Find all HTML files
        html_files = []
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Skip hidden directories and common excludes
            dirs[:] = [d for d in dirs if not d.startswith(
                '.') and d not in ['node_modules', '__pycache__', 'venv', '.venv']]

            for file in files:
                if file.endswith('.html'):
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(PROJECT_ROOT)
                    html_files.append(rel_path)

        # Sort by directory then filename
        html_files.sort()

        # Group by directory
        grouped = {}
        for file_path in html_files:
            dir_name = str(file_path.parent)
            if dir_name == '.':
                dir_name = 'Root'
            if dir_name not in grouped:
                grouped[dir_name] = []
            grouped[dir_name].append(file_path)

        # Generate HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>HTML File Viewer - WildfirePrediction</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 32px;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .stats {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
            color: #555;
        }}
        .directory {{
            margin-bottom: 30px;
        }}
        .directory h2 {{
            color: #444;
            font-size: 18px;
            margin-bottom: 12px;
            padding: 10px;
            background: #e9ecef;
            border-radius: 5px;
            cursor: pointer;
            user-select: none;
        }}
        .directory h2:hover {{
            background: #dee2e6;
        }}
        .directory h2::before {{
            content: '📁 ';
        }}
        .file-list {{
            list-style: none;
            padding-left: 20px;
        }}
        .file-list li {{
            padding: 8px 10px;
            border-bottom: 1px solid #eee;
        }}
        .file-list li:hover {{
            background: #f8f9fa;
        }}
        .file-list a {{
            color: #0066cc;
            text-decoration: none;
            font-size: 14px;
            display: block;
        }}
        .file-list a::before {{
            content: '📄 ';
        }}
        .file-list a:hover {{
            text-decoration: underline;
        }}
        .filename {{
            font-weight: 500;
        }}
        .collapsed {{
            display: none;
        }}
        .search-box {{
            width: 100%;
            padding: 12px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .search-box:focus {{
            outline: none;
            border-color: #0066cc;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>HTML File Viewer</h1>
        <div class="subtitle">WildfirePrediction Project</div>

        <div class="stats">
            <strong>{len(html_files)}</strong> HTML files found across <strong>{len(grouped)}</strong> directories
        </div>

        <input type="text" class="search-box" id="searchBox" placeholder="Search for files or directories...">

        <div id="fileList">
"""

        for dir_name in sorted(grouped.keys()):
            files = grouped[dir_name]
            html_content += f"""
            <div class="directory" data-dir="{dir_name}">
                <h2 onclick="toggleDirectory(this)">{dir_name} ({len(files)} files)</h2>
                <ul class="file-list">
"""
            for file_path in files:
                url_path = quote(str(file_path))
                html_content += f'                    <li data-filename="{file_path.name}"><a href="/{url_path}" target="_blank"><span class="filename">{file_path.name}</span></a></li>\n'

            html_content += """                </ul>
            </div>
"""

        html_content += """        </div>
    </div>

    <script>
        function toggleDirectory(element) {
            const fileList = element.nextElementSibling;
            fileList.classList.toggle('collapsed');
        }

        // Search functionality
        const searchBox = document.getElementById('searchBox');
        searchBox.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            const directories = document.querySelectorAll('.directory');

            directories.forEach(dir => {
                const dirName = dir.getAttribute('data-dir').toLowerCase();
                const files = dir.querySelectorAll('.file-list li');
                let hasVisibleFiles = false;

                files.forEach(file => {
                    const filename = file.getAttribute('data-filename').toLowerCase();
                    if (filename.includes(searchTerm) || dirName.includes(searchTerm)) {
                        file.style.display = '';
                        hasVisibleFiles = true;
                    } else {
                        file.style.display = 'none';
                    }
                });

                dir.style.display = hasVisibleFiles || searchTerm === '' ? '' : 'none';

                // Expand directories with matches
                if (hasVisibleFiles && searchTerm !== '') {
                    dir.querySelector('.file-list').classList.remove('collapsed');
                }
            });
        });
    </script>
</body>
</html>
"""

        # Send response
        encoded = html_content.encode('utf-8', 'surrogateescape')
        f = io.BytesIO()
        f.write(encoded)
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        return f


def get_tailscale_ip():
    """Get the Tailscale IP address of this machine"""
    try:
        result = subprocess.run(
            ['tailscale', 'ip', '-4'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: try to get from status
    try:
        result = subprocess.run(
            ['tailscale', 'status', '--json'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            if 'Self' in data and 'TailscaleIPs' in data['Self']:
                ips = data['Self']['TailscaleIPs']
                if ips:
                    return ips[0]
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass

    return None


def main():
    # Change to project directory
    os.chdir(PROJECT_ROOT)

    # Get Tailscale IP
    tailscale_ip = get_tailscale_ip()

    print("=" * 60)
    print("HTML File Viewer Server")
    print("=" * 60)
    print(f"Serving from: {PROJECT_ROOT}")
    print(f"Port: {PORT}")
    print()

    if tailscale_ip:
        url = f"http://{tailscale_ip}:{PORT}"
        print(f"Tailscale URL: {url}")
        print()
        print("   Copy and paste this URL in your browser!")
    else:
        print("Could not detect Tailscale IP")
        print(f"   Local URL: http://localhost:{PORT}")
        print()
        print("   To find your Tailscale IP, run: tailscale ip -4")

    print()
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    # Create and start server
    with socketserver.TCPServer(("0.0.0.0", PORT), HTMLFileHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down server...")


if __name__ == "__main__":
    main()
