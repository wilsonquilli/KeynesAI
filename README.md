KeynesAI – Intelligent Stock Market Prediction Platform
KeynesAI is a stock forecasting platform that leverages machine learning to achieve prediction accuracy around 65%. This advanced, web-based system supports users in analyzing the stock market, generating price forecasts, and managing their investment portfolios. The platform is built using Flask and integrates machine learning techniques for robust performance.

Key Features:

  1. Stock Insights:
    - Live stock data monitoring
    - Detection of patterns in stock behavior
    - Integration of technical indicators
    - Predictive models powered by machine learning
    - Stock data collected using the YFinance API
    - Trained a model using the RandomForestClassifier from scikit-learn

  2. User Authentication:
    - Secure sign-up and login features
    - Encrypted password storage with hashing
    - Session tracking for secure access
    - Built using a MySQL database via XAMPP

  3. Portfolio Tracker:
    - Monitor and manage multiple stock holdings
    - Dynamic portfolio valuation
    - Performance analysis tools
    - Track gains and losses in real-time

  4. Sector Breakdown:
    - Organized stock categories using a tree data structure
    - Navigate stocks by sector
    - Filter stocks based on their classification

  5. Prediction Module:
    - Stock data retrieved from YFinance
    - Forecasts generated using machine learning algorithms
    - Predictions across various time ranges
    - Pattern recognition based on technical data
    - Utilizes a Random Forest classifier model

Tech Stack: 
  - Backend: Python (Flask)
  - Database: MySQL
  - Machine Learning: scikit-learn
  - Data Processing: pandas
  - Frontend: HTML/CSS, JavaScript

Requirements:
  - Python 3.x
  - MySQL Server
  - XAMPP (for local testing and development)

Setup Instructions:
1. Clone the Repository
  - git clone https://github.com/yourusername/KeynesAI.git
  - cd KeynesAI

2. Install Dependencies
  - pip install -r requirements.txt

3. Database Setup
  - Start MySQL via XAMPP

Create the database KeynesAI and run:
  - CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
    );
The application will auto-create necessary tables on the first launch.

4. Update Database Credentials
  - In app.py, adjust db_config if your MySQL login details differ.

Running the App:
  - python app.py
  - Visit your app in the browser: http://localhost:5000

Project Structure:
  KeynesAI/
  ├── app.py              #Core backend application
  ├── static/             #Backend logic and assets
  │   ├── stock.py        #Stock analysis functionality
  │   ├── stock_tree.py   #Hierarchical stock categorization
  │   └── boomCrash.py    #Market fluctuation data models
  ├── templates/          #Frontend HTML templates
  └── requirements.txt    #Required Python packages

Contributors:
Mostafa Amer – Developed the ML model, sourced YFinance data, implemented stock sector categorization with tree structures, built the user authentication system with MySQL, and designed several UI components.
Nicholas Shvelidze – Developed boomCrash.py and chart.py modules to enhance the training set, boosting model accuracy to 65%.
Wilson Quilli – Handled the frontend development; also assisted with backend integration of the ML model through Python (Flask).

License:
This project is released under the MIT License – see the LICENSE file for details.

Contact:
For questions or support, please reach out via email: wilo240105@gmail.com

README written by: Wilson Quilli
