import os
import django
from flask import Flask, request
from flask_cors import CORS, cross_origin
from kwsapp.views import process_audio_bng, process_audio_man, process_audio_miz

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'KWSSystem.settings')
django.setup()
app = Flask(__name__)
CORS(app)

ROOT='./'
os.makedirs(ROOT+'rec',exist_ok=True)

@app.route('/flask_process_audio_bng', methods=['POST'])
@cross_origin()
def flask_process_audio_bng():
    return process_audio_bng(request)

@app.route('/flask_process_audio_man', methods=['POST'])
@cross_origin()
def flask_process_audio_man():
    return process_audio_man(request)

@app.route('/flask_process_audio_miz', methods=['POST'])
@cross_origin()
def flask_process_audio_miz():
    return process_audio_miz(request)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
