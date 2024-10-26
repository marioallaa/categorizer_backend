from flask import Flask, json, request, send_file, render_template, jsonify
import os
import uuid
import datetime 
from flask_cors import CORS
import pandas as pd
from werkzeug.utils import secure_filename
from main_auth import require_authentication
from ml_utils import categorize_transactions, process_log_file, transform_csv, reshape_dataframe


import firebase_admin
from firebase_admin import credentials, auth



app = Flask(__name__)

service_account_key_path = 'firebase.json'

try:
    default_app = firebase_admin.get_app()
except ValueError:
    # Initialize Firebase Admin SDK
    cred = credentials.Certificate(service_account_key_path)
    firebase_admin.initialize_app(cred)


CORS(app)

@app.route('/')
def index():
    return "running"


@app.route('/categorize/<unique_id>/<method>', methods=['POST'])
@require_authentication
def categorize_endpoint(unique_id, method):
    # Parse the JSON body to retrieve parameters
    data = request.get_json()
    
    # Extract parameters from the JSON body
    description = data.get('description')
    suggestedCategory = data.get('suggestedCategory')
    customize = data.get('customize')
    
    directory = os.path.join('bank_stm', unique_id)
    if not os.path.exists(directory):
        return jsonify({'error': f'Request ID not found'}), 404
    



    if method == 'money_in':
        file_path = os.path.join(directory, 'money_in.csv')
        output_filepath = os.path.join(directory, 'cat_money_in.csv')
        categorize_transactions(file_path, description, output_filepath, suggestedCategory, customize)
        return send_file(output_filepath, as_attachment=True, download_name='cat_money_in.csv')

    elif method == 'money_out':
        file_path = os.path.join(directory, 'money_out.csv')
        output_filepath = os.path.join(directory, 'cat_money_out.csv')
        categorize_transactions(file_path, description, output_filepath, suggestedCategory, customize)
        return send_file(output_filepath, as_attachment=True, download_name='cat_money_out.csv')
    


    else:
        return jsonify({'error': f"Request ID found, but method not accepted."}), 404



@app.route('/transform', methods=['POST'])
@require_authentication
def transform_file():
    if 'file' not in request.files or 'transform_request' not in request.form:
        return jsonify({'error': 'No file part or transformation request'}), 400
    
    file = request.files['file']
    transform_request = request.form.get('transform_request')
    print(transform_request)
    
    if file.filename == '':
         return jsonify({'error': f"No File Selected"}), 400
    
    try:
        transform_request = json.loads(transform_request)  # Safely convert JSON string to dictionary
        if not isinstance(transform_request, dict):
            raise ValueError("Invalid transformation request format")
    except Exception as e:
        return jsonify({'error': f"Error in transformation request: {e}"}), 400
    
    if file and file.filename.endswith('.csv'):
        # Create a unique directory for this request
        unique_id = str(uuid.uuid4())

        

        request_dir = os.path.join('bank_stm', unique_id)
        os.makedirs(request_dir, exist_ok=True)
        

        # Save the input file
        input_filename = 'input.csv'
        input_filepath = os.path.join(request_dir, input_filename)
        file.save(input_filepath)

        # Process the file and save the outputs
        output_files = transform_csv(input_filepath, transform_request, output_dir=request_dir)
        
        response = {'status': 'success', 'files': {}, 'errors': []}

        # Collect output files
        if 'money_in' in output_files:
            response['files']['money_in'] = os.path.join(request_dir, os.path.basename(output_files['money_in']))
        if 'money_out' in output_files:
            response['files']['money_out'] = os.path.join(request_dir, os.path.basename(output_files['money_out']))
        if 'log' in output_files:
            response['files']['log'] = os.path.join(request_dir, os.path.basename(output_files['log']))

        # Check if log file exists and extract errors if any
        if 'log' in output_files:
            log_filepath = output_files['log']
            errors = process_log_file(log_filepath)
            response['errors'] = errors
            
        response['id'] = unique_id
        
        return jsonify(response)
    
    return jsonify({'error': 'Invalid file format'}), 400


@app.route('/files/<unique_id>', methods=['GET'])
@require_authentication
def get_files(unique_id,):
    request_dir = os.path.join('bank_stm', unique_id,)
    
    if not os.path.exists(request_dir):
        return jsonify({'error': 'Request ID not found'}), 404
    
    files = {}
    for filename in os.listdir(request_dir):
        file_path = os.path.join(request_dir, filename)
        if os.path.isfile(file_path):
            files[filename] = os.path.join(request_dir, filename)
    
    if not files:
        return jsonify({'error': 'No files found for this request'}), 404
    
    return jsonify(files)


@app.route('/files/<unique_id>/<filename>', methods=['GET'])
@require_authentication
def get_file(unique_id, filename):
    request_dir = os.path.join('bank_stm', unique_id)
    file_path = os.path.join(request_dir, filename)
    
    if not os.path.exists(request_dir):
        return jsonify({'error': 'Request ID not found'}), 404
    
    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path, as_attachment=True, download_name=filename)


@app.route('/files/<unique_id>/<filename>', methods=['POST'])
@require_authentication
def upload_extra_file(unique_id, filename):
    # Check if the request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # If the user does not select a file, the browser may submit an empty part without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    print(file.filename)
    
    # Ensure the uploaded file is a CSV
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file type, only CSV allowed'}), 400

    # Secure the filename to prevent directory traversal attacks
    secure_name = secure_filename(filename)
    
    request_dir = os.path.join('bank_stm', unique_id)
    
    # Create the directory if it does not exist
    if not os.path.exists(request_dir):
        os.makedirs(request_dir)

    file_path = os.path.join(request_dir, secure_name)
    
    # Save the file to the specified directory
    file.save(file_path)
    
    return jsonify({'message': 'File uploaded successfully'}), 201





@app.route('/upload/excel', methods=['POST'])
# @require_authentication
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        return jsonify({'error': 'The file is not an Excel file'}), 400

    json_data = request.form if request.form else request.get_json()
    client_name = json_data.get('clientName')
    file_type = json_data.get('type')

    if not client_name or not file_type:
        return jsonify({'error': 'Missing required information in JSON payload'}), 400

    # Generate a unique ID
    unique_id = str(uuid.uuid4())[:6]

    # Create directory for this client
    client_dir = os.path.join('workbooks', f"{client_name}_{unique_id}")
    os.makedirs(client_dir, exist_ok=True)

    date = datetime.datetime.now()

    # Generate the filename
    date_str = date.strftime("%d_%m_%Y")
    filename = f"{file_type}_{date_str}.xlsx"
    file_path = os.path.join(client_dir, secure_filename(filename))

    # Save the file
    file.save(file_path)

    try:
        # Load the workbook and get sheet names
        workbook = pd.ExcelFile(file_path)
        sheet_names = workbook.sheet_names

        # Generate download URLs for each sheet
        sheets_info = []
        for sheet_name in sheet_names:
            sheet_url = f"/workbook/{client_name}_{unique_id}/{secure_filename(filename)}/sheet/{sheet_name}"
            sheets_info.append({
                'sheet_name': sheet_name,
                'download_url': sheet_url
            })

        # Generate the download URL for the entire workbook
        download_url = f"/workbooks/{client_name}_{unique_id}/{secure_filename(filename)}"

        return jsonify({
            'message': 'File uploaded and saved successfully',
            'download_url': download_url,
            'sheets': sheets_info
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/workbooks/<unique_id>/<filename>', methods=['GET'])
# @require_authentication
def get_workbook(unique_id, filename):
    request_dir = os.path.join('workbooks', unique_id)
    file_path = os.path.join(request_dir, filename)
    
    if not os.path.exists(request_dir):
        return jsonify({'error': 'Request ID not found'}), 404
    
    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path, as_attachment=True, download_name=filename)


@app.route('/workbook/<unique_id>/<filename>/sheet/<sheet_name>', methods=['GET'])
# @require_authentication
def get_sheet(unique_id, filename, sheet_name):
    file_path = os.path.join('workbooks', unique_id, filename)

    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404

    try:
        workbook = pd.ExcelFile(file_path)
        
        if sheet_name not in workbook.sheet_names:
            return jsonify({'error': f'Sheet "{sheet_name}" not found in the workbook.'}), 404
        
        sheet_df = pd.read_excel(workbook, sheet_name=sheet_name)

        # Save the sheet temporarily as a separate file to send it
        temp_file_path = os.path.join('workbooks', f"{sheet_name}.xlsx")
        sheet_df.to_excel(temp_file_path, index=False)

        return send_file(temp_file_path, as_attachment=True)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists('bank_stm'):
        os.makedirs('bank_stm')
    app.run(port=5000, debug=True, )
