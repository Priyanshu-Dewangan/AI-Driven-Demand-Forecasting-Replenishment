"""
Flask Application with HTML Dashboard for LSTM Inventory Forecasting
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import traceback

# Import LSTM model functions
from lstm_model import (
    train_lstm_models, generate_forecasts, load_trained_models,
    calculate_statistics, calculate_reorder_recommendations,
    process_uploaded_file, save_uploaded_file, export_to_json,
    DATASETS_DIR, EXPORTS_DIR, MODELS_DIR
)

app = Flask(__name__)
app.secret_key = 'lstm-inventory-dashboard-secret-key'
CORS(app)

# Global state
models_cache = {}
metadata_cache = None
current_data = None
current_forecasts = {}
current_stats = {}
current_config = {
    'forecastDays': 7,
    'reorderThreshold': 50,
    'leadTime': 3,
    'safetyFactor': 1.5,
    'lstmLayers': 5,
    'trainingEpochs': 20
}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html',
                         data_loaded=current_data is not None,
                         api_status='connected',
                         stats=current_stats,
                         config=current_config)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    global current_data, models_cache, metadata_cache, current_forecasts, current_stats
    
    try:
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect('/')
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect('/')
        
        if not file.filename.lower().endswith('.csv'):
            flash('Please upload a CSV file', 'error')
            return redirect('/')
        
        # Save file
        file_path, filename = save_uploaded_file(file)
        
        # Process file
        df, error = process_uploaded_file(file_path)
        if error:
            flash(f'Error processing file: {error}', 'error')
            return redirect('/')
        
        current_data = df
        
        # Calculate statistics
        current_stats = calculate_statistics(df)
        
        # Try to load existing models
        models_cache, metadata_cache = load_trained_models()
        
        if models_cache:
            # Generate forecasts with existing models
            current_forecasts = generate_forecasts(df, models_cache, current_config['forecastDays'])
            flash(f'File uploaded successfully! Loaded {len(df)} records. Using existing trained models.', 'success')
        else:
            flash(f'File uploaded successfully! Loaded {len(df)} records. Train models to generate forecasts.', 'info')
        
        return redirect('/')
    
    except Exception as e:
        flash(f'Upload error: {str(e)}', 'error')
        return redirect('/')

@app.route('/train', methods=['POST'])
def train_models():
    """Train LSTM models"""
    global current_data, models_cache, metadata_cache, current_forecasts
    
    try:
        if current_data is None:
            return jsonify({'success': False, 'error': 'No data loaded. Please upload a CSV file first.'})
        
        force_retrain = request.json.get('force_retrain', False) if request.is_json else False
        
        # Train models
        models_cache, metadata_cache = train_lstm_models(current_data, force_retrain=force_retrain)
        
        if models_cache:
            # Generate forecasts
            current_forecasts = generate_forecasts(current_data, models_cache, current_config['forecastDays'])
            
            return jsonify({
                'success': True,
                'message': f'Models trained successfully for {len(models_cache)} products',
                'num_products': len(models_cache)
            })
        else:
            return jsonify({'success': False, 'error': 'Model training failed'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/forecast', methods=['GET', 'POST'])
def get_forecasts():
    """Get forecasts for all products"""
    global current_forecasts
    
    try:
        if request.method == 'POST':
            # Regenerate forecasts
            if current_data is not None and models_cache:
                forecast_days = request.json.get('forecast_days', current_config['forecastDays']) if request.is_json else current_config['forecastDays']
                current_forecasts = generate_forecasts(current_data, models_cache, forecast_days)
        
        if not current_forecasts:
            return jsonify({'success': False, 'error': 'No forecasts available. Train models first.'})
        
        # Format forecasts for frontend
        formatted_forecasts = {}
        for product_id, forecast_data in current_forecasts.items():
            formatted_forecasts[product_id] = {
                'forecast': forecast_data.get('forecast_details', []),
                'historical': [],  # Could add historical data here
                'avgSales': forecast_data.get('avg_sales', 0),
                'currentInventory': forecast_data.get('current_inventory', 0),
                'accuracy': forecast_data.get('accuracy', 0),
                'trend': forecast_data.get('trend', 'up')
            }
        
        return jsonify({
            'success': True,
            'forecasts': formatted_forecasts,
            'generated_at': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reorder-recommendations', methods=['GET', 'POST'])
def get_reorder_recommendations():
    """Get reorder recommendations"""
    global current_forecasts, current_config
    
    try:
        if request.method == 'POST':
            # Update config
            config = request.json if request.is_json else {}
            current_config.update(config)
        
        if not current_forecasts:
            return jsonify({'success': False, 'error': 'No forecasts available. Train models first.'})
        
        # Calculate recommendations
        recommendations = calculate_reorder_recommendations(current_forecasts, current_config)
        
        # Format for frontend
        formatted_recommendations = []
        for rec in recommendations:
            formatted_recommendations.append({
                'product': {
                    'id': rec['product_id'],
                    'name': rec['product_id'],
                    'category': 'Unknown',
                    'avgSales': rec['avg_sales']
                },
                'currentInventory': rec['current_inventory'],
                'reorderPoint': rec['reorder_point'],
                'avgSales': rec['avg_sales'],
                'daysUntilStockout': rec['days_until_stockout'],
                'urgency': rec['urgency'],
                'needsReorder': rec['needs_reorder'],
                'recommendedOrder': rec['recommended_order']
            })
        
        return jsonify({
            'success': True,
            'recommendations': formatted_recommendations,
            'config': current_config
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get current statistics"""
    global current_stats, current_data
    
    stats = current_stats.copy() if current_stats else {}
    
    # Add additional stats
    stats['data_loaded'] = current_data is not None
    stats['data_rows'] = len(current_data) if current_data is not None else 0
    stats['models_trained'] = len(models_cache) if models_cache else 0
    stats['forecasts_generated'] = len(current_forecasts) if current_forecasts else 0
    
    return jsonify({
        'success': True,
        'statistics': stats
    })

@app.route('/products', methods=['GET'])
def get_products():
    """Get list of products"""
    global current_data
    
    try:
        if current_data is None:
            return jsonify({'success': False, 'error': 'No data loaded'})
        
        # Find product column
        product_col = None
        for col in current_data.columns:
            if 'product' in col.lower():
                product_col = col
                break
        
        if product_col is None:
            product_col = current_data.columns[0]
        
        products = []
        unique_products = current_data[product_col].unique()[:50]  # Limit to 50
        
        for product_id in unique_products:
            products.append({
                'id': str(product_id),
                'name': str(product_id),
                'category': 'Unknown'
            })
        
        return jsonify({
            'success': True,
            'products': products
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update configuration"""
    global current_config
    
    if request.method == 'POST':
        config = request.json if request.is_json else {}
        current_config.update(config)
        return jsonify({'success': True, 'config': current_config})
    
    return jsonify({'success': True, 'config': current_config})

@app.route('/sample-data', methods=['POST'])
def load_sample_data():
    """Load sample data"""
    global current_data, current_stats
    
    try:
        # Create sample data
        sample_data = []
        products = ['P001', 'P002', 'P003', 'P004']
        categories = ['Electronics', 'Audio', 'Wearables', 'Home']
        
        for i in range(30):  # 30 days of data
            date = (datetime.now() - timedelta(days=29-i)).strftime('%Y-%m-%d')
            for j, product in enumerate(products):
                inventory = max(0, 200 - (i * 10) + np.random.randint(-20, 20))
                sales = np.random.randint(5, 15)
                price = [1299.99, 199.99, 349.99, 599.99][j]
                
                sample_data.append({
                    'Date': date,
                    'Product ID': product,
                    'Product Name': ['Premium Laptop', 'Wireless Headphones', 'Smart Watch', 'Coffee Machine'][j],
                    'Category': categories[j],
                    'Inventory Level': inventory,
                    'Units Sold': sales,
                    'Price': price
                })
        
        current_data = pd.DataFrame(sample_data)
        current_stats = calculate_statistics(current_data)
        
        # Try to load existing models
        models_cache, metadata_cache = load_trained_models()
        
        if models_cache:
            current_forecasts = generate_forecasts(current_data, models_cache, current_config['forecastDays'])
        
        return jsonify({
            'success': True,
            'message': 'Sample data loaded successfully',
            'rows': len(current_data)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export', methods=['GET'])
def export_data():
    """Export data as JSON"""
    global current_data, current_forecasts, current_stats, current_config
    
    try:
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'statistics': current_stats,
            'config': current_config,
            'forecasts': current_forecasts
        }
        
        filename = f"inventory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_path = export_to_json(export_data, filename)
        
        return send_file(export_path, as_attachment=True, download_name=filename)
    
    except Exception as e:
        flash(f'Export error: {str(e)}', 'error')
        return redirect('/')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'models_loaded': len(models_cache) if models_cache else 0,
        'data_loaded': current_data is not None,
        'forecasts_available': len(current_forecasts) if current_forecasts else 0
    })

@app.route('/api/forecast/<product_id>', methods=['GET'])
def get_product_forecast(product_id):
    """Get forecast for specific product"""
    global current_forecasts
    
    try:
        if not current_forecasts or product_id not in current_forecasts:
            return jsonify({'success': False, 'error': 'Forecast not found for this product'})
        
        forecast_data = current_forecasts[product_id]
        
        return jsonify({
            'success': True,
            'product_id': product_id,
            'forecast': forecast_data.get('forecast_details', []),
            'current_inventory': forecast_data.get('current_inventory', 0),
            'avg_sales': forecast_data.get('avg_sales', 0),
            'accuracy': forecast_data.get('accuracy', 0),
            'trend': forecast_data.get('trend', 'up')
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error='Server error'), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('exports', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    print("\n🚀 Starting LSTM Inventory Dashboard...")
    print("=" * 80)
    print("📊 Dashboard available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /                   - Main dashboard")
    print("  POST /upload             - Upload CSV file")
    print("  POST /train              - Train LSTM models")
    print("  GET  /forecast           - Get forecasts")
    print("  GET  /reorder-recommendations - Get reorder recommendations")
    print("  GET  /statistics         - Get statistics")
    print("  POST /sample-data        - Load sample data")
    print("  GET  /export             - Export data as JSON")
    print("  GET  /health             - Health check")
    print("\n📁 Directory structure:")
    print("  templates/    - HTML templates")
    print("  static/       - CSS, JS, images")
    print("  uploads/      - Uploaded CSV files")
    print("  exports/      - Exported JSON files")
    print("  saved_models/ - Trained LSTM models")
    print("=" * 80)
    
    # Try to load existing models
    models_cache, metadata_cache = load_trained_models()
    
    app.run(debug=True, port=5000, host='0.0.0.0')