from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash, session
import os
import pandas as pd
import mysql.connector
from sklearn.ensemble import RandomForestClassifier
from static.stock import run_for_stock, predict_future, download_data, add_features, add_pattern_features, add_horizon_features
from static.boomCrash import BoomCrashModel
from werkzeug.security import generate_password_hash, check_password_hash
import secrets

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = secrets.token_hex(32)  # Generate a secure random key

# Database configuration
db_config = {
    'host': '127.0.0.1',
    'user': 'root',  # Default XAMPP MySQL username
    'database': 'KeynesAI'
}

def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        print("Successfully connected to the database")
        return conn
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            # Create users table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL
                )
            ''')
            conn.commit()
            cursor.close()
            conn.close()
            print("Database table created successfully")
        except mysql.connector.Error as err:
            print(f"Error creating table: {err}")

# Initialize database on startup
init_db()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                flash('Login successful!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Invalid email or password', 'error')
        except Exception as e:
            flash(f'Database error: {str(e)}', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        
        print(f"Attempting to register user: {email}")
        
        try:
            conn = get_db_connection()
            if not conn:
                flash('Database connection failed', 'error')
                return redirect(url_for('register'))
            
            cursor = conn.cursor()
            print("Executing INSERT query...")
            cursor.execute('INSERT INTO users (email, password) VALUES (%s, %s)',
                         (email, hashed_password))
            conn.commit()
            print("User successfully registered in database")
            cursor.close()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            print(f"Database error during registration: {err}")
            if err.errno == 1062:  # Duplicate entry error
                flash('Email already registered', 'error')
            else:
                flash(f'Database error: {str(err)}', 'error')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))


from static.stock_tree import StockTree

# Initialize the stock tree (put this near the top of your file)
stock_tree = StockTree()

@app.route('/sectors.html')
def sectors():
    category_path = request.args.get('category', '').split('/')
    if category_path == ['']:
        category_path = []
    category_info = stock_tree.get_category_info(category_path)
    stocks = stock_tree.get_stocks_in_category(category_path)
    return render_template(
        'sectors.html',
        category_info=category_info,
        current_path=category_path,
        stocks=stocks
    )

@app.route('/')
def home():
    return render_template('index.HTML')

@app.route('/about_us.html')
def about():  
    return render_template('about_us.html')

@app.route('/predictions.html')
def predictions():
    ticker = request.args.get('ticker', 'AAPL') 

    try:
        df = download_data(ticker)
        df = add_features(df)
        df = add_pattern_features(df)
        df = add_horizon_features(df, [2, 5, 60, 250, 1000])
        df = df.dropna()

        predictors = [f"Close_Ratio_{h}" for h in [2, 5, 60, 250, 1000]] + [f"Trend_{h}" for h in [2, 5, 60, 250, 1000]]
        model = RandomForestClassifier(n_estimators=150, min_samples_split=50, random_state=1)
        model.fit(df[predictors], df["Target"])  

        predictions_df = predict_future(df, model, predictors, ticker)

        if predictions_df is not None:
            table_html = predictions_df.head(20).to_html(classes="prediction-table", border=0)
        else:
            table_html = "<p style='color:red;'>No predictions returned.</p>"

    except Exception as e:
        table_html = f"<p style='color:red;'>Error generating predictions for {ticker}: {e}</p>"

    return render_template('predictions.html', predictions_table=table_html, selected_ticker=ticker)

@app.route('/portfolio.html')
def portfolio():
    portfolio_data = [
        {"name": "AAPL", "shares": 10, "buy_price": 150.00, "current_price": 165.32},
        {"name": "IBM", "shares": 5, "buy_price": 130.00, "current_price": 127.45},
        {"name": "MSFT", "shares": 8, "buy_price": 290.00, "current_price": 312.25},
        {"name": "S&P500", "shares": 20, "buy_price": 4000.00, "current_price": 4180.22},
    ]

    total_value = 0
    total_cost = 0
    top_stock = None
    best_gain = float('-inf')

    for stock in portfolio_data:
        stock["value"] = stock["shares"] * stock["current_price"]
        stock["cost"] = stock["shares"] * stock["buy_price"]
        stock["change_percent"] = round(((stock["current_price"] - stock["buy_price"]) / stock["buy_price"]) * 100, 2)
        total_value += stock["value"]
        total_cost += stock["cost"]
        if stock["change_percent"] > best_gain:
            best_gain = stock["change_percent"]
            top_stock = stock["name"]

    gain_loss = round(total_value - total_cost, 2)

    return render_template("portfolio.html",
                           portfolio=portfolio_data,
                           total_value=round(total_value, 2),
                           gain_loss=gain_loss,
                           top_stock=top_stock)

@app.route('/trending_stocks.html')
def trending_stocks():
    return render_template('trending_stocks.html')

@app.route('/nick.html')
def nick():
    return render_template('nick.html')

@app.route('/sam.html')
def sam():
    return render_template('sam.html')

@app.route('/wilson.html')
def wilson():
    return render_template('wilson.html')

@app.route('/mostafa.html')
def mostafa():
    return render_template('mostafa.html')

@app.route('/style.css')
def style():
    return send_from_directory('static', 'style.css')

if __name__ == '__main__':
    app.run(debug=True)
