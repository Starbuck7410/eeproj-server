
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, session
import threading
import time
from users import USERS
import secrets
import shared_state as state
import config
from processing import process_stream
import socket



# Stop the annoying log messages
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


app = Flask(__name__)
app.secret_key = secrets.token_hex(32) # Required for session management



def generate():
    while True:
        time.sleep(0.01)
        with state.lock:
            if state.output_frame is None:
                continue
            frame_to_send = state.output_frame
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Simple verification logic
        if username in USERS and USERS[username] == password:
            session['user'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')

@app.route("/")
def index():
    if 'user' in session:
        return render_template('index.html')
    return redirect(url_for('login'))


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/garbage")
def garbage():
    with state.lock:
        local_x, local_y = state.garbage_loc.get_coords()
        local_valid = state.garbage_loc.get_valid()
    target = {
        "x": float(local_x / 1000),
        "y": float(local_y / 1000),
        "valid": local_valid,
    }
    return jsonify(target)

@app.route("/api/basket")
def basket():
    with state.lock:
        local_x, local_y = state.basket_loc.get_coords()
        local_valid = state.basket_loc.get_valid()
    target = {
        "x": float(local_x / 1000),
        "y": float(local_y / 1000),
        "valid": local_valid,
    }
    return jsonify(target)

@app.route("/api/frame_stats")
def frame_stats():
    with state.lock:
        local_fps = state.fps
        local_res = state.resolution
        local_online = state.online
        local_name = state.rpi_name
        local_g_id = state.garbage_id

    stats = {
        "fps": local_fps,
            # Note this ⨯ isnt the letter x, but the math symbol ⨯.
        "resolution": f"{local_res[0]} ⨯ {local_res[1]}" if len(local_res) > 1 else "0 ⨯ 0", 
        "online": local_online,
        "device_name": local_name,
        "garbage_id": local_g_id
    }
    return jsonify(stats)


@app.route("/api/set_battery", methods=["POST"])
def set_battery():
    data = request.get_json(silent=True) or {}

    battery = data.get("battery")
    if battery is None:
        return jsonify({"error": "battery field required"}), 400

    with state.lock:
        state.battery = battery

    return jsonify({
        "success": True,
        "battery": battery
    })


@app.route("/api/get_battery")
def get_battery():
    with state.lock:
        battery = state.battery

    return jsonify({
        "battery": battery
    })

if __name__ == "__main__":
    process_thread = threading.Thread(target=process_stream, daemon=True)
    process_thread.start()
    hostname = socket.gethostname()
    print("Login at:")
    print(f"http://127.0.0.1:{config.PORT}")
    print(f"http://{socket.gethostbyname(hostname)}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, threaded=True, debug=False)
    

