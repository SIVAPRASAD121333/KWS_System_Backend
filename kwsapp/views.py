# kwsapp/views.py
import wave
from io import BytesIO
import sys
import os
from django.shortcuts import render
from django.http import JsonResponse
import subprocess
from flask import jsonify
from django.views.decorators.csrf import csrf_exempt
import base64
# import io
from pydub import AudioSegment

def display_page(request):
    return render(request, 'KWS-NE.htm')

def display_page_bng(request):
    return render(request, 'index_bng.html')

def display_page_man(request):
    return render(request, 'index_man.html')

def display_page_miz(request):
    return render(request, 'index_miz.html')


ROOT='./'
os.makedirs(ROOT+'rec',exist_ok=True)

SYSTEM_PATH=ROOT+'KWS_V_bng/'
sys.path.append(SYSTEM_PATH)
import KWS_denseNet_bng as KWS_bng
SYSTEM_PATH=ROOT+'KWS_V_man/'
sys.path.append(SYSTEM_PATH)
import KWS_denseNet_man as KWS_man
SYSTEM_PATH=ROOT+'KWS_V_miz/'
sys.path.append(SYSTEM_PATH)
import KWS_denseNet_miz as KWS_miz

@csrf_exempt
def upload_audio(request):
    if request.method == 'POST':
        try:
            if request.method == 'POST' and 'wavBase64' in request.POST:
                # Get the base64-encoded WAV data
                wav_base64 = request.POST['wavBase64']

                # Decode base64 and convert to bytes
                wav_bytes = base64.b64decode(wav_base64)

                # Create a BytesIO object to read the bytes
                wav_buffer = BytesIO(wav_bytes)

                # Specify the file path to save the WAV file
                save_path = 'rec/recorded_audio.wav'  # Update with desired path and filename

                # Save the WAV file
                with wave.open(save_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Set number of channels (1 for mono, 2 for stereo)
                    wav_file.setsampwidth(2)   # Set sample width in bytes (2 for 16-bit)
                    wav_file.setframerate(44100)  # Set the frame rate (e.g., 44100 Hz for CD quality)
                    # wav_file.setframerate(16000)  # Set the frame rate (e.g., 44100 Hz for CD quality)
                    wav_file.writeframes(wav_buffer.read())

                return JsonResponse({'status': 'success', 'message': 'Audio file saved successfully'})
            

            # Return a JSON response indicating success
            return JsonResponse({'status': 'success', 'message': 'Audio processed and saved'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@csrf_exempt
def process_audio_bng(request):
    if request.method == 'POST' and request.files.get('audio_data'):
        audio_file = request.files['audio_data']
        #size = audio_file.size
        input_data = audio_file.read()
        output_name = audio_file.name + ".wav"

        # Specify the path where you want to save the uploaded file on the server
        save_path = ROOT+"rec/"+output_name
        save_path=save_path.replace(':', '-')

        # Write the file content to the specified save path
        with open(save_path, 'wb') as output_file:
            output_file.write(input_data)
        try:  
            output_file_path = save_path+'.wav'
            audio = AudioSegment.from_file(save_path)
            # audio = AudioSegment.from_file(BytesIO(input_data), format="wav")
            # sample_width = 4  # 32 bits is 4 bytes
            # sample_rate = 8000
            sample_width = 2  # 32 bits is 4 bytes
            sample_rate = 16000
            converted_audio = audio.set_sample_width(sample_width).set_frame_rate(sample_rate)

            converted_audio.export(output_file_path, format="wav")


            result_str,result_conf = KWS_bng.main(output_file_path)
            print("Output of the script:\n", result_str)
            return jsonify({'status': 'success', 'result': result_str, 'conf': result_conf}),200

        except subprocess.CalledProcessError as e: 
            print("Error occurred:", e)
            result_str = None
            return jsonify({'status': 'Audio processing error', 'message': str( e)}),400


        
    else:
        return jsonify({'status': 'error', 'message': 'Invalid request'}),400

@csrf_exempt
def process_audio_man(request):
    if request.method == 'POST' and request.files.get('audio_data'):
        audio_file = request.files['audio_data']
        #size = audio_file.size
        input_data = audio_file.read()
        output_name = audio_file.name + ".wav"

        # Specify the path where you want to save the uploaded file on the server
        save_path = ROOT+f"rec/{output_name}"
        save_path=save_path.replace(':', '-')

        # Write the file content to the specified save path
        with open(save_path, 'wb') as output_file:
            output_file.write(input_data)
        try:  
            output_file_path = save_path+'.wav'
            audio = AudioSegment.from_file(save_path)
            # sample_width = 4  # 32 bits is 4 bytes
            # sample_rate = 8000
            sample_width = 2  # 32 bits is 4 bytes
            sample_rate = 16000
            converted_audio = audio.set_sample_width(sample_width).set_frame_rate(sample_rate)

            converted_audio.export(output_file_path, format="wav")


            result_str,result_conf = KWS_man.main(output_file_path)
            print("Output of the script:\n", result_str)
            return jsonify({'status': 'success', 'result': result_str, 'conf': result_conf}),200

        except subprocess.CalledProcessError as e: 
            print("Error occurred:", e)
            result_str = None
            return jsonify({'status': 'Audio processing error', 'message': str( e)}),400


        
    else:
        return jsonify({'status': 'error', 'message': 'Invalid request'}),400

@csrf_exempt
def process_audio_miz(request):
    if request.method == 'POST' and request.files.get('audio_data'):
        audio_file = request.files['audio_data']
        #size = audio_file.size
        input_data = audio_file.read()
        output_name = audio_file.name + ".wav"

        # Specify the path where you want to save the uploaded file on the server
        save_path = ROOT+f"rec/{output_name}"
        save_path=save_path.replace(':', '-')

        # Write the file content to the specified save path
        with open(save_path, 'wb') as output_file:
            output_file.write(input_data)
        try:  
            output_file_path = save_path+'.wav'
            audio = AudioSegment.from_file(save_path)
            # sample_width = 4  # 32 bits is 4 bytes
            # sample_rate = 8000
            sample_width = 2  # 32 bits is 4 bytes
            sample_rate = 16000
            converted_audio = audio.set_sample_width(sample_width).set_frame_rate(sample_rate)

            converted_audio.export(output_file_path, format="wav")


            result_str,result_conf = KWS_miz.main(output_file_path)
            print("Output of the script:\n", result_str)
            return jsonify({'status': 'success', 'result': result_str, 'conf': result_conf}),200

        except subprocess.CalledProcessError as e: 
            print("Error occurred:", e)
            result_str = None
            return jsonify({'status': 'Audio processing error', 'message': str( e)}),400


        
    else:
        return jsonify({'status': 'error', 'message': 'Invalid request'}),400
