from functools import wraps
from flask import request, jsonify
from firebase_admin import auth

def require_authentication(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get the Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authorization header is missing'}), 401
        
        # Extract the token from the header (assuming format: "Bearer <token>")
        token = auth_header.split(' ')[1] if ' ' in auth_header else auth_header
        if not token:
            return jsonify({'error': 'Token is missing from Authorization header'}), 401
        
        try:
            # Verify the ID token
            decoded_token = auth.verify_id_token(token)
            request.user = decoded_token  # Attach user info to request object
        except Exception as e:
            return jsonify({'fire error': str(e)}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function
