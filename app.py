import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import json
import re
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Document processing libraries
import PyPDF2
import docx
from PIL import Image
import pytesseract
import openpyxl
from io import BytesIO
import zipfile
import tempfile
import os

# Statistical libraries
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression

MAX_FILE_SIZE_MB = 100  # 100MB absolute limit
WARNING_FILE_SIZE_MB = 50  # 50MB warning threshold
MAX_ROWS_FOR_ANALYSIS = 50000  # Limit rows for performance
MIN_FILE_SIZE_BYTES = 10  # Minimum file size

# Configuration
st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DataAnalystAgent:
    def __init__(self):
        self.together_api_key = None
        self.model_name = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        self.data = None
        self.data_summary = None
        self.conversation_history = []
        self.analysis_context = {}
        
    def set_api_key(self, api_key):
        """Set the Together.ai API key"""
        self.together_api_key = api_key
        
    def call_llama_model(self, prompt, max_tokens=2048, temperature=0.7):
        """Call the Llama model via Together.ai API"""
        if not self.together_api_key:
            return "Please set your Together.ai API key first."
            
        headers = {
            "Authorization": f"Bearer {self.together_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert data analyst. Provide detailed, accurate analysis and insights. Always be specific about findings and recommendations. When discussing data, reference actual values and patterns you observe."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
        
        try:
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return "Error: No response from model"
                
        except requests.exceptions.RequestException as e:
            return f"API Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def extract_text_from_pdf(self, pdf_file):
        """Enhanced PDF extraction with better error handling and structure preservation"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            extracted_data = {
                'text': "",
                'metadata': {},
                'page_count': len(pdf_reader.pages),
                'structure': []
            }
            
            # Extract metadata
            if pdf_reader.metadata:
                extracted_data['metadata'] = {
                    'title': pdf_reader.metadata.get('/Title', 'Unknown'),
                    'author': pdf_reader.metadata.get('/Author', 'Unknown'),
                    'creation_date': pdf_reader.metadata.get('/CreationDate', 'Unknown')
                }
            
            # Extract text with page structure
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                extracted_data['text'] += f"\n--- Page {i+1} ---\n{page_text}\n"
                extracted_data['structure'].append({
                    'page': i+1,
                    'text_length': len(page_text),
                    'has_content': bool(page_text.strip())
                })
            
            # Try alternative extraction if text is minimal
            if len(extracted_data['text'].strip()) < 100:
                try:
                    import pdfplumber
                    with pdfplumber.open(pdf_file) as pdf:
                        extracted_data['text'] = ""
                        for page in pdf.pages:
                            extracted_data['text'] += page.extract_text() or ""
                except ImportError:
                    pass
            
            return extracted_data
        except Exception as e:
            return f"Error extracting PDF: {str(e)}"
    
    def extract_text_from_docx(self, docx_file):
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"Error extracting DOCX: {str(e)}"
    
    def extract_text_from_image(self, image_file):
        """Enhanced OCR with preprocessing and confidence scoring"""
        try:
            image = Image.open(image_file)
            
            # Image preprocessing for better OCR
            import cv2
            import numpy as np
            
            # Convert PIL to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocessing steps
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Noise removal
            denoised = cv2.medianBlur(gray, 5)
            
            # Thresholding
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL
            processed_image = Image.fromarray(thresh)
            
            # OCR with confidence data
            ocr_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
            
            # Extract text with confidence filtering
            confident_text = []
            for i, conf in enumerate(ocr_data['conf']):
                if int(conf) > 30:  # Filter low confidence text
                    text = ocr_data['text'][i].strip()
                    if text:
                        confident_text.append(text)
            
            extracted_text = ' '.join(confident_text)
            
            # Fallback to basic OCR if enhanced method fails
            if not extracted_text.strip():
                extracted_text = pytesseract.image_to_string(image)
            
            return {
                'text': extracted_text,
                'confidence_stats': {
                    'avg_confidence': np.mean([c for c in ocr_data['conf'] if c > 0]),
                    'high_confidence_words': len([c for c in ocr_data['conf'] if c > 80]),
                    'total_words': len([t for t in ocr_data['text'] if t.strip()])
                },
                'image_info': {
                    'size': image.size,
                    'mode': image.mode,
                    'format': image.format
                }
            }
        except Exception as e:
            return f"Error extracting text from image: {str(e)}"
    

    def validate_file_size(self, uploaded_file):
        """Validate file size and provide warnings/errors"""
        try:
            file_size = len(uploaded_file.getvalue())
            file_size_mb = file_size / (1024 * 1024)
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            if file_size < MIN_FILE_SIZE_BYTES:
                return False, "File appears to be empty or corrupted"
            
            if file_size_mb > MAX_FILE_SIZE_MB:
                return False, f"File size ({file_size_mb:.1f}MB) exceeds maximum limit of {MAX_FILE_SIZE_MB}MB. Please use a smaller file."
            
            # Warning for large files
            if file_size_mb > WARNING_FILE_SIZE_MB:
                return True, f"Warning: Large file ({file_size_mb:.1f}MB) detected. Processing may be slow."
            
            return True, ""
            
        except Exception as e:
            return False, f"Error checking file size: {str(e)}"
    
    def format_file_size(self, size_bytes):
        """Format file size in appropriate units"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    
    def process_uploaded_file(self, uploaded_file):
        """Process different types of uploaded files with size validation"""
        
        # First, validate file size
        is_valid, message = self.validate_file_size(uploaded_file)
        
        if not is_valid:
            return None, message
        
        if message:  # Warning message
            st.warning(message)
        
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        
                        # Check row count and sample if too large
                        if len(df) > MAX_ROWS_FOR_ANALYSIS:
                            st.warning(f"Dataset has {len(df):,} rows. Using first {MAX_ROWS_FOR_ANALYSIS:,} rows for analysis to ensure performance.")
                            df = df.head(MAX_ROWS_FOR_ANALYSIS)
                        
                        self.data = df
                        return df, "csv"
                    except UnicodeDecodeError:
                        continue
                return None, "Error: Unable to decode CSV file with any supported encoding"
                
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
                
                # Check row count and sample if too large
                if len(df) > MAX_ROWS_FOR_ANALYSIS:
                    st.warning(f"Dataset has {len(df):,} rows. Using first {MAX_ROWS_FOR_ANALYSIS:,} rows for analysis.")
                    df = df.head(MAX_ROWS_FOR_ANALYSIS)
                
                self.data = df
                return df, "excel"
                
            elif file_extension == 'txt':
                content = uploaded_file.read().decode('utf-8')
                
                # Limit text content for analysis
                if len(content) > 50000:  # ~50k characters
                    st.warning("Large text file detected. Using first 50,000 characters for analysis.")
                    content = content[:50000] + "..."
                
                return content, "text"
                
            elif file_extension == 'pdf':
                text = self.extract_text_from_pdf(uploaded_file)
                return text, "pdf"
                
            elif file_extension == 'docx':
                text = self.extract_text_from_docx(uploaded_file)
                return text, "docx"
                
            elif file_extension in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
                text = self.extract_text_from_image(uploaded_file)
                return text, "image"
                
            else:
                return None, f"Unsupported file type: {file_extension}"
                
        except Exception as e:
            return None, f"Error processing file: {str(e)}"
    
    def generate_data_summary(self, data):
        """Generate comprehensive data summary"""
        if data is None:
            return "No data available"
            
        summary = {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "numeric_columns": list(data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(data.select_dtypes(include=['object']).columns),
            "memory_usage": data.memory_usage(deep=True).sum(),
        }
        
        # Statistical summary for numeric columns
        if summary["numeric_columns"]:
            summary["numeric_stats"] = data[summary["numeric_columns"]].describe().to_dict()
        
        # Categorical summary
        if summary["categorical_columns"]:
            summary["categorical_stats"] = {}
            for col in summary["categorical_columns"]:
                summary["categorical_stats"][col] = {
                    "unique_count": data[col].nunique(),
                    "top_values": data[col].value_counts().head(5).to_dict()
                }
        
        self.data_summary = summary
        return summary
    
    def detect_data_patterns(self, data):
        """Detect interesting patterns in the data"""
        patterns = []
        
        try:
            # Check for correlations in numeric data
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                
                if high_corr:
                    patterns.append(f"High correlations found: {high_corr}")
            
            # Check for outliers
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = data[(data[col] < Q1 - 1.5*IQR) | (data[col] > Q3 + 1.5*IQR)]
                if len(outliers) > 0:
                    patterns.append(f"Column '{col}' has {len(outliers)} outliers")
            
            # Check for data distribution
            for col in numeric_cols:
                skewness = data[col].skew()
                if abs(skewness) > 1:
                    patterns.append(f"Column '{col}' is highly skewed (skewness: {skewness:.2f})")
            
            # Check for missing data patterns
            missing_cols = [col for col in data.columns if data[col].isnull().sum() > 0]
            if missing_cols:
                patterns.append(f"Missing data in columns: {missing_cols}")
            
        except Exception as e:
            patterns.append(f"Error detecting patterns: {str(e)}")
        
        return patterns
    
    def perform_statistical_analysis(self, data, numeric_cols):
        """Perform statistical analysis on numeric columns"""
        results = {}
        try:
            for col in numeric_cols:
                col_data = data[col].dropna()
                results[col] = {
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis(),
                    'normality_test': stats.normaltest(col_data)[1] if len(col_data) > 8 else None
                }
        except Exception as e:
            results['error'] = str(e)
        return results

    def perform_clustering_analysis(self, data, numeric_cols):
        """Perform clustering analysis"""
        if len(numeric_cols) < 2:
            return {"error": "Need at least 2 numeric columns for clustering"}
        
        try:
            # Prepare data
            cluster_data = data[numeric_cols].dropna()
            if len(cluster_data) < 10:
                return {"error": "Not enough data points for clustering"}
            
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            
            return {
                'n_clusters': 3,
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'inertia': kmeans.inertia_,
                'cluster_counts': pd.Series(clusters).value_counts().to_dict()
            }
        except Exception as e:
            return {"error": str(e)}

    def perform_regression_analysis(self, data, numeric_cols):
        """Perform regression analysis"""
        if len(numeric_cols) < 2:
            return {"error": "Need at least 2 numeric columns for regression"}
        
        try:
            # Use first column as target, rest as features
            target_col = numeric_cols[0]
            feature_cols = numeric_cols[1:]
            
            # Prepare data
            reg_data = data[numeric_cols].dropna()
            if len(reg_data) < 10:
                return {"error": "Not enough data points for regression"}
            
            X = reg_data[feature_cols]
            y = reg_data[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Fit model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            return {
                'target_column': target_col,
                'feature_columns': feature_cols,
                'mse': mse,
                'r2_score': model.score(X_test, y_test),
                'coefficients': dict(zip(feature_cols, model.coef_)),
                'intercept': model.intercept_
            }
        except Exception as e:
            return {"error": str(e)}

    def analyze_feature_importance(self, data, numeric_cols):
        """Analyze feature importance using Random Forest"""
        if len(numeric_cols) < 2:
            return {"error": "Need at least 2 numeric columns"}
        
        try:
            target_col = numeric_cols[0]
            feature_cols = numeric_cols[1:]
            
            reg_data = data[numeric_cols].dropna()
            if len(reg_data) < 10:
                return {"error": "Not enough data points"}
            
            X = reg_data[feature_cols]
            y = reg_data[target_col]
            
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            
            importance_dict = dict(zip(feature_cols, rf.feature_importances_))
            
            return {
                'feature_importance': importance_dict,
                'most_important': max(importance_dict, key=lambda k: importance_dict[k])

            }
        except Exception as e:
            return {"error": str(e)}

    def assess_data_quality(self, data):
        """Assess data quality"""
        try:
            total_cells = data.shape[0] * data.shape[1]
            missing_cells = data.isnull().sum().sum()
            
            quality_score = ((total_cells - missing_cells) / total_cells) * 100
            
            # Check for duplicates
            duplicate_rows = data.duplicated().sum()
            
            return {
                'completeness_score': quality_score,
                'missing_values_count': missing_cells,
                'duplicate_rows': duplicate_rows,
                'data_types_consistent': True,  # Simplified check
                'quality_rating': 'Good' if quality_score > 90 else 'Fair' if quality_score > 70 else 'Poor'
            }
        except Exception as e:
            return {"error": str(e)}

    def build_enhanced_context(self, question_type):
        """Build enhanced context for questions"""
        context = ""
        
        if self.data is not None:
            context += f"Dataset Info:\n"
            context += f"- Shape: {self.data.shape}\n"
            context += f"- Columns: {list(self.data.columns)}\n"
            
            if hasattr(self, 'data_summary') and self.data_summary:
                context += f"- Numeric columns: {self.data_summary.get('numeric_columns', [])}\n"
                context += f"- Missing values: {sum(self.data_summary.get('missing_values', {}).values())}\n"
            
            # Add sample data
            context += f"\nSample data:\n{self.data.head(3).to_string()}\n"
        
        return context

    def get_followup_context(self, question):
        """Get context for follow-up questions"""
        if len(self.conversation_history) > 0:
            last_conversation = self.conversation_history[-1]
            return f"\nPrevious question: {last_conversation['question']}\nPrevious answer: {last_conversation['response'][:200]}...\n"
        return ""

    def process_excel_with_fallback(self, uploaded_file, processing_log):
        """Process Excel with fallback strategies"""
        try:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            processing_log['processing_steps'].append("Successfully parsed Excel file")
            df = self.clean_dataframe(df, processing_log)
            return df, "excel"
        except Exception as e:
            try:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file, engine='xlrd')
                processing_log['processing_steps'].append("Successfully parsed with xlrd engine")
                df = self.clean_dataframe(df, processing_log)
                return df, "excel"
            except Exception as e2:
                processing_log['errors'].append(f"Excel parsing failed: {str(e)}, {str(e2)}")
                return None, None

    def process_pdf_with_fallback(self, uploaded_file, processing_log):
        """Process PDF with fallback strategies"""
        try:
            text_data = self.extract_text_from_pdf(uploaded_file)
            if isinstance(text_data, dict):
                processing_log['processing_steps'].append("Successfully extracted PDF text")
                return text_data['text'], "pdf"
            else:
                processing_log['errors'].append(f"PDF extraction failed: {text_data}")
                return None, None
        except Exception as e:
            processing_log['errors'].append(f"PDF processing error: {str(e)}")
            return None, None

    def process_other_files(self, uploaded_file, processing_log):
        """Process other file types"""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_extension == 'txt':
                content = uploaded_file.read().decode('utf-8')
                processing_log['processing_steps'].append("Successfully read text file")
                return content, "text"
            elif file_extension == 'docx':
                text = self.extract_text_from_docx(uploaded_file)
                processing_log['processing_steps'].append("Successfully extracted DOCX text")
                return text, "docx"
            elif file_extension in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
                text_data = self.extract_text_from_image(uploaded_file)
                processing_log['processing_steps'].append("Successfully performed OCR on image")
                return text_data, "image"
            else:
                processing_log['errors'].append(f"Unsupported file type: {file_extension}")
                return None, None
        except Exception as e:
            processing_log['errors'].append(f"Error processing {file_extension}: {str(e)}")
            return None, None
    
    def create_visualizations(self, data, chart_types=None):
        """Create appropriate visualizations based on data and selected chart types"""
        if data is None:
            return None
            
        figures = []
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            # If no chart types specified, create all
            if chart_types is None:
                chart_types = ["Distribution Plots", "Correlation Heatmap", "Scatter Plots", "Box Plots", "Bar Charts"]
            
            # Distribution plots for numeric columns
            if "Distribution Plots" in chart_types and len(numeric_cols) > 0:
                for col in numeric_cols[:2]:  # Limit to first 2 columns
                    fig = px.histogram(data, x=col, title=f'Distribution of {col}')
                    figures.append(fig)
            
            # Box plots for numeric columns
            if "Box Plots" in chart_types and len(numeric_cols) > 0:
                fig = px.box(data, y=numeric_cols[0], title=f'Box Plot of {numeric_cols[0]}')
                figures.append(fig)
            
            # Correlation heatmap
            if "Correlation Heatmap" in chart_types and len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                fig = px.imshow(corr_matrix, 
                              title='Correlation Heatmap',
                              labels=dict(color="Correlation"))
                figures.append(fig)
            
            # Scatter plots
            if "Scatter Plots" in chart_types and len(numeric_cols) >= 2:
                fig = px.scatter(data, x=numeric_cols[0], y=numeric_cols[1], 
                               title=f'{numeric_cols[0]} vs {numeric_cols[1]}')
                figures.append(fig)
            
            # Bar charts for categorical data
            if "Bar Charts" in chart_types and len(categorical_cols) > 0:
                for col in categorical_cols[:1]:  # Limit to first column
                    value_counts = data[col].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f'Top 10 Values in {col}')
                    figures.append(fig)
            
            # Time series plot if datetime column exists
            datetime_cols = data.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0 and len(numeric_cols) > 0:
                fig = px.line(data, x=datetime_cols[0], y=numeric_cols[0],
                            title=f'{numeric_cols[0]} over time')
                figures.append(fig)
                
        except Exception as e:
            st.error(f"Error creating visualizations: {str(e)}")
        
        return figures
    
    def perform_advanced_analysis(self, data, analysis_type="comprehensive"):
        """Enhanced advanced analysis with more robust methods"""
        if data is None:
            return "No data available for analysis"
        
        results = {
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'shape': data.shape,
                'memory_usage': data.memory_usage(deep=True).sum(),
                'completeness': (1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            }
        }
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            if analysis_type in ["comprehensive", "statistical"]:
                results["statistical_analysis"] = self.perform_statistical_analysis(data, numeric_cols)
            
            if analysis_type in ["comprehensive", "clustering"]:
                results["clustering_analysis"] = self.perform_clustering_analysis(data, numeric_cols)
            
            if analysis_type in ["comprehensive", "regression"]:
                results["regression_analysis"] = self.perform_regression_analysis(data, numeric_cols)
            
            if analysis_type in ["comprehensive", "anomaly"]:
                results["anomaly_detection"] = self.detect_anomalies(data, numeric_cols)
            
            # Feature importance analysis
            if len(numeric_cols) > 1:
                results["feature_importance"] = self.analyze_feature_importance(data, numeric_cols)
            
            # Data quality assessment
            results["data_quality"] = self.assess_data_quality(data)
            
        except Exception as e:
            results["error"] = f"Error in advanced analysis: {str(e)}"
            results["partial_results"] = True
        
        # Store analysis context
        self.analysis_context['last_analysis'] = results
        self.analysis_context['last_analysis_type'] = analysis_type
        
        return results

    def detect_anomalies(self, data, numeric_cols):
        """Enhanced anomaly detection using multiple methods"""
        anomalies = {}
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) < 10:
                continue
            
            # Statistical outliers (IQR method)
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            statistical_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            
            # Z-score outliers
            z_scores = np.abs(stats.zscore(col_data))
            z_outliers = data[z_scores > 3]
            
            # Isolation Forest for multivariate outliers
            try:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
                iso_outliers = data[outlier_labels == -1]
                
                anomalies[col] = {
                    'statistical_outliers_count': len(statistical_outliers),
                    'z_score_outliers_count': len(z_outliers),
                    'isolation_forest_outliers_count': len(iso_outliers),
                    'outlier_percentage': (len(statistical_outliers) / len(data)) * 100,
                    'severity': 'high' if len(statistical_outliers) > len(data) * 0.05 else 'low'
                }
            except ImportError:
                anomalies[col] = {
                    'statistical_outliers_count': len(statistical_outliers),
                    'z_score_outliers_count': len(z_outliers),
                    'outlier_percentage': (len(statistical_outliers) / len(data)) * 100
                }
        
        return anomalies
    
    def maintain_conversation_context(self, question, response):
        """Enhanced context management for multi-turn conversations"""
        # Store conversation with more context
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'response': response,
            'data_state': {
                'columns': list(self.data.columns) if self.data is not None else None,
                'shape': self.data.shape if self.data is not None else None,
                'last_analysis': self.analysis_context.get('last_analysis_type', None)
            },
            'question_type': self.classify_question_type(question)
        }
        
        self.conversation_history.append(conversation_entry)
        
        # Maintain context window (keep last 10 exchanges)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def classify_question_type(self, question):
        """Classify the type of question for better context handling"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['correlation', 'relationship', 'associated']):
            return 'correlation'
        elif any(word in question_lower for word in ['trend', 'pattern', 'over time', 'change']):
            return 'trend'
        elif any(word in question_lower for word in ['outlier', 'anomaly', 'unusual', 'strange']):
            return 'anomaly'
        elif any(word in question_lower for word in ['missing', 'null', 'empty', 'incomplete']):
            return 'data_quality'
        elif any(word in question_lower for word in ['predict', 'forecast', 'future', 'estimate']):
            return 'prediction'
        elif any(word in question_lower for word in ['summary', 'overview', 'describe', 'explain']):
            return 'summary'
        elif any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'
        else:
            return 'general'

    def answer_question(self, question, context_data=None):
        """Enhanced Q&A with better context and follow-up handling"""
        question_type = self.classify_question_type(question)
        
        # Build enhanced context
        context = self.build_enhanced_context(question_type)
        
        # Check for follow-up questions
        if self.is_followup_question(question):
            context += self.get_followup_context(question)
        
        # Add relevant previous answers if applicable
        relevant_history = self.get_relevant_conversation_history(question_type)
        if relevant_history:
            context += f"\nRelevant previous discussion:\n{relevant_history}\n"
        
        # Construct enhanced prompt
        prompt = f"""
        {context}
        
        Current Question: {question}
        Question Type: {question_type}
        
        Instructions:
        - If this is a follow-up question, reference previous discussion appropriately
        - Provide specific data-driven answers with actual values
        - If the question requires calculations, show the work
        - If the question is unclear, ask for clarification
        - Suggest related analyses that might be helpful
        
        Answer:"""
        
        response = self.call_llama_model(prompt, max_tokens=2048)
        
        # Store with enhanced context
        self.maintain_conversation_context(question, response)
        
        # Update analysis context
        self.analysis_context['last_question_type'] = question_type
        self.analysis_context['last_question_time'] = datetime.now()
        
        return response

    def is_followup_question(self, question):
        """Detect if current question is a follow-up"""
        followup_indicators = [
            'what about', 'how about', 'and what', 'can you also',
            'additionally', 'furthermore', 'moreover', 'also',
            'that', 'this', 'those', 'these', 'it', 'they'
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in followup_indicators)

    def get_relevant_conversation_history(self, question_type):
        """Get relevant previous conversations based on question type"""
        relevant_conversations = []
        
        for entry in self.conversation_history[-5:]:  # Last 5 conversations
            if entry.get('question_type') == question_type:
                relevant_conversations.append(f"Q: {entry['question']}\nA: {entry['response'][:200]}...")
        
        return "\n".join(relevant_conversations)
    
    def robust_file_processing(self, uploaded_file):
        """More robust file processing with better error handling"""
        processing_log = {
            'start_time': datetime.now(),
            'file_name': uploaded_file.name,
            'file_size': len(uploaded_file.getvalue()),
            'processing_steps': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # Size validation with detailed feedback
            is_valid, message = self.validate_file_size(uploaded_file)
            processing_log['processing_steps'].append(f"Size validation: {message}")
            
            if not is_valid:
                processing_log['errors'].append(message)
                return None, None, processing_log
            
            # File type detection
            file_extension = uploaded_file.name.split('.')[-1].lower()
            processing_log['file_type'] = file_extension
            
            # Process based on file type with fallback methods
            if file_extension == 'csv':
                data, file_type = self.process_csv_with_fallback(uploaded_file, processing_log)
            elif file_extension in ['xlsx', 'xls']:
                data, file_type = self.process_excel_with_fallback(uploaded_file, processing_log)
            elif file_extension == 'pdf':
                data, file_type = self.process_pdf_with_fallback(uploaded_file, processing_log)
            else:
                data, file_type = self.process_other_files(uploaded_file, processing_log)
            
            processing_log['end_time'] = datetime.now()
            processing_log['processing_duration'] = (processing_log['end_time'] - processing_log['start_time']).total_seconds()
            
            return data, file_type, processing_log
            
        except Exception as e:
            processing_log['errors'].append(f"Unexpected error: {str(e)}")
            return None, None, processing_log

    def process_csv_with_fallback(self, uploaded_file, processing_log):
        """Process CSV with multiple fallback strategies"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        separators = [',', ';', '\t', '|']
        
        for encoding in encodings:
            for separator in separators:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding, sep=separator)
                    
                    # Validate the DataFrame
                    if df.shape[1] > 1 and df.shape[0] > 0:
                        processing_log['processing_steps'].append(f"Successfully parsed with encoding: {encoding}, separator: '{separator}'")
                        
                        # Data cleaning
                        df = self.clean_dataframe(df, processing_log)
                        
                        return df, "csv"
                except Exception as e:
                    continue
        
        processing_log['errors'].append("Could not parse CSV with any supported encoding/separator combination")
        return None, None

    def clean_dataframe(self, df, processing_log):
        """Clean DataFrame with logging"""
        original_shape = df.shape
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Handle column names
        df.columns = df.columns.astype(str)
        df.columns = [col.strip() for col in df.columns]
        
        # Log cleaning actions
        if df.shape != original_shape:
            processing_log['processing_steps'].append(f"Cleaned data: {original_shape} -> {df.shape}")
        
        return df
    
    def generate_insights(self, data):
        """Generate AI-powered insights about the data"""
        if data is None:
            return "No data available for insights"
        
        summary = self.generate_data_summary(data)
        
        # Check if summary generation failed
        if isinstance(summary, str):  # This handles the 'No data available' case
            return summary
        
        patterns = self.detect_data_patterns(data)
        
        # Create a comprehensive prompt
        prompt = f"""
        Analyze this dataset and provide actionable insights:
        
        Dataset Overview:
        - Shape: {summary['shape']}
        - Columns: {summary['columns']}
        - Numeric columns: {summary['numeric_columns']}
        - Categorical columns: {summary['categorical_columns']}
        
        Statistical Summary:
        {summary.get('numeric_stats', 'No numeric data')}
        
        Detected Patterns:
        {chr(10).join(patterns) if patterns else 'No specific patterns detected'}
        
        Sample Data:
        {data.head(3).to_string()}
        
        Please provide:
        1. Key findings and trends
        2. Potential data quality issues
        3. Recommendations for further analysis
        4. Business implications (if applicable)
        5. Suggested visualizations
        
        Focus on actionable insights that would be valuable for decision-making.
        """
        
        insights = self.call_llama_model(prompt, max_tokens=1500)
        return insights
    
def create_enhanced_chat_interface():
    """Create a more sophisticated chat interface"""
    st.subheader("💬 Intelligent Q&A Assistant")
    
    # Chat configuration
    col1, col2 = st.columns([3, 1])
    with col1:
        conversation_mode = st.selectbox(
            "Conversation Mode",
            ["Normal", "Deep Analysis", "Quick Answers", "Guided Discovery"]
        )
    with col2:
        if st.button("🔄 Reset Context"):
            st.session_state.agent.conversation_history = []
            st.success("Context reset!")
    
    # Smart question suggestions based on data
    if st.session_state.uploaded_data is not None:
        smart_suggestions = generate_smart_suggestions(st.session_state.uploaded_data)
        if smart_suggestions:
            st.info("💡 Smart suggestions based on your data:")
            suggestion_cols = st.columns(min(len(smart_suggestions), 3))
            for i, suggestion in enumerate(smart_suggestions[:3]):
                with suggestion_cols[i]:
                    if st.button(suggestion, key=f"smart_suggest_{i}"):
                        return suggestion
    
    return None

def generate_smart_suggestions(data):
    """Generate intelligent question suggestions based on data characteristics"""
    suggestions = []
    
    if hasattr(data, 'columns'):
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # Correlation suggestions
        if len(numeric_cols) > 1:
            suggestions.append(f"What's the correlation between {numeric_cols[0]} and {numeric_cols[1]}?")
        
        # Missing data suggestions
        missing_cols = [col for col in data.columns if data[col].isnull().sum() > 0]
        if missing_cols:
            suggestions.append(f"How should I handle missing values in {missing_cols[0]}?")
        
        # Distribution suggestions
        if len(numeric_cols) > 0:
            suggestions.append(f"What does the distribution of {numeric_cols[0]} look like?")
        
        # Categorical analysis
        if len(categorical_cols) > 0:
            suggestions.append(f"What are the most common values in {categorical_cols[0]}?")
        
        return suggestions

def main():
    st.title("🤖 Intelligent Data Analyst Agent")
    st.sidebar.title("Configuration")
    
    # Initialize the agent
    if 'agent' not in st.session_state:
        st.session_state.agent = DataAnalystAgent()
        st.session_state.uploaded_data = None
        st.session_state.file_type = None
        st.session_state.analysis_results = None
    
    # API Key Input
    api_key = st.sidebar.text_input("Enter Together.ai API Key", type="password")
    if api_key:
        st.session_state.agent.set_api_key(api_key)
        st.sidebar.success("API Key Set!")
    else:
        st.sidebar.warning("Please enter your Together.ai API key to continue")
        st.info("📌 **Instructions:**\n1. Get your free API key from [Together.ai](https://www.together.ai/)\n2. Enter it in the sidebar\n3. Upload your data file\n4. Start analyzing!")
        return
    
    # File Upload with custom styling to hide default text
    st.sidebar.header("📁 File Upload")
    
    # Custom CSS to hide the default file size text
    st.markdown("""
    <style>
    .uploadedFile {
        display: none;
    }
    div[data-testid="stFileUploaderDropzone"] > div > div > small {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file (Max 50MB)",
        type=['csv', 'xlsx', 'xls', 'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload files up to 50MB (recommended) or 100MB (maximum)"
    )

    if uploaded_file is not None:
        # Show file info with proper size formatting
        file_size_bytes = len(uploaded_file.getvalue())
        file_size_formatted = st.session_state.agent.format_file_size(file_size_bytes)
        st.sidebar.info(f"File: {uploaded_file.name}\nSize: {file_size_formatted}")
        
        with st.spinner("Processing file..."):
            data, file_type = st.session_state.agent.process_uploaded_file(uploaded_file)
            
            if isinstance(data, str) and (data.startswith("Error") or data.startswith("File size")):
                st.error(data)
                return
            
            st.session_state.uploaded_data = data
            st.session_state.file_type = file_type
            
            st.sidebar.success(f"File uploaded successfully! Type: {file_type}")
    
    # Main content area
    if st.session_state.uploaded_data is not None:
        data = st.session_state.uploaded_data
        file_type = st.session_state.file_type
        
        # Check if data is a DataFrame (structured data) or string (text data)
        is_dataframe = isinstance(data, pd.DataFrame)
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🔍 Advanced Analysis", "💡 AI Insights", "❓ Q&A"])
        
        with tab1:
            st.header("Data Overview")
            
            if is_dataframe and file_type in ['csv', 'excel']:
                # Display basic info with corrected memory usage formatting
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", data.shape[0])
                with col2:
                    st.metric("Columns", data.shape[1])
                with col3:
                    memory_bytes = data.memory_usage(deep=True).sum()
                    memory_formatted = st.session_state.agent.format_file_size(memory_bytes)
                    st.metric("Memory Usage", memory_formatted)
                
                # Display data types
                st.subheader("Column Information")
                col_info = pd.DataFrame({
                    'Column': data.columns,
                    'Data Type': data.dtypes,
                    'Non-Null Count': data.count(),
                    'Null Count': data.isnull().sum(),
                    'Null %': (data.isnull().sum() / len(data) * 100).round(2)
                })
                st.dataframe(col_info)
                
                # Display sample data
                st.subheader("Sample Data")
                st.dataframe(data.head(10))
                
                # Statistical summary
                try:
                    numeric_cols = data.select_dtypes(include='number').columns
                    if len(numeric_cols) > 0:
                        st.subheader("Statistical Summary")
                        st.dataframe(data[numeric_cols].describe())
                except Exception as e:
                    st.warning(f"Could not generate statistical summary: {e}")
                
                # Missing values heatmap
                if data.isnull().sum().sum() > 0:
                    st.subheader("Missing Values Pattern")
                    try:
                        fig = px.imshow(data.isnull().T, title="Missing Values Heatmap")
                        st.plotly_chart(fig)
                    except Exception as e:
                        st.warning(f"Could not generate missing values heatmap: {e}")
                        
            elif isinstance(data, str):
                # For text-based files
                st.subheader("Document Content")
                st.text_area("Content", data, height=400)
                
                # Basic text statistics
                word_count = len(data.split())
                char_count = len(data)
                line_count = len(data.split('\n'))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words", word_count)
                with col2:
                    st.metric("Characters", char_count)
                with col3:
                    st.metric("Lines", line_count)
            else:
                st.error("Unsupported data format for overview display.")
        
        with tab2:
            st.header("📈 Data Visualizations")
            
            if is_dataframe and file_type in ['csv', 'excel']:
                # Chart type selection with maximum 3 selections
                chart_options = st.multiselect(
                    "Select chart types to generate (Maximum 3):",
                    ["Distribution Plots", "Correlation Heatmap", "Scatter Plots", "Box Plots", "Bar Charts"],
                    default=["Distribution Plots", "Correlation Heatmap"],
                    max_selections=3,
                    help="Choose up to 3 chart types for optimal performance"
                )
                
                if len(chart_options) > 3:
                    st.error("Please select maximum 3 chart types for optimal performance.")
                elif chart_options and st.button("Generate Visualizations"):
                    with st.spinner("Creating visualizations..."):
                        figures = st.session_state.agent.create_visualizations(data, chart_options)
                        
                        if figures:
                            for i, fig in enumerate(figures):
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No visualizations could be generated for this data.")
                
                # Custom plotting section
                st.subheader("Custom Visualization")
                try:
                    numeric_cols = data.select_dtypes(include='number').columns
                    categorical_cols = data.select_dtypes(include=['object']).columns
                    
                    if len(numeric_cols) > 0 or len(categorical_cols) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            plot_type = st.selectbox("Plot Type", ["Scatter", "Line", "Bar", "Histogram", "Box"])
                            
                        with col2:
                            if plot_type in ["Scatter", "Line"]:
                                x_col = st.selectbox("X-axis", list(data.columns))
                                y_col = st.selectbox("Y-axis", list(data.columns))
                            else:
                                x_col = st.selectbox("Column", list(data.columns))
                                y_col = None
                        
                        if st.button("Create Custom Plot"):
                            try:
                                fig = None
                                
                                if plot_type == "Scatter" and x_col and y_col:
                                    fig = px.scatter(data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                                elif plot_type == "Line" and x_col and y_col:
                                    fig = px.line(data, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                                elif plot_type == "Bar" and x_col:
                                    fig = px.bar(data, x=x_col, title=f"Bar chart of {x_col}")
                                elif plot_type == "Histogram" and x_col:
                                    fig = px.histogram(data, x=x_col, title=f"Distribution of {x_col}")
                                elif plot_type == "Box" and x_col:
                                    fig = px.box(data, y=x_col, title=f"Box plot of {x_col}")
                                
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error("Could not create plot with the selected parameters.")
                            except Exception as e:
                                st.error(f"Error creating plot: {str(e)}")
                    else:
                        st.info("No suitable columns found for visualization.")
                except Exception as e:
                    st.error(f"Error setting up visualization options: {e}")
            else:
                st.info("Visualizations are available for structured data (CSV, Excel) files.")
        
        with tab3:
            st.header("🔍 Advanced Analysis")
            
            if is_dataframe and file_type in ['csv', 'excel']:
                analysis_type = st.selectbox(
                    "Select Analysis Type",
                    ["Comprehensive", "Statistical Testing", "Clustering", "Regression Analysis"]
                )
                
                if st.button("Run Advanced Analysis"):
                    with st.spinner("Performing advanced analysis..."):
                        analysis_type_map = {
                            "Comprehensive": "comprehensive",
                            "Statistical Testing": "statistical",
                            "Clustering": "clustering",
                            "Regression Analysis": "regression"
                        }
                        
                        results = st.session_state.agent.perform_advanced_analysis(
                            data, analysis_type_map[analysis_type]
                        )
                        
                        st.session_state.analysis_results = results
                        
                        # Display results
                        if isinstance(results, dict):
                            for key, value in results.items():
                                st.subheader(key.replace("_", " ").title())
                                if isinstance(value, dict):
                                    st.json(value)
                                else:
                                    st.write(value)
                        else:
                            st.write(results)
                
                # Data patterns detection
                st.subheader("Data Patterns")
                if st.button("Detect Patterns"):
                    patterns = st.session_state.agent.detect_data_patterns(data)
                    if patterns:
                        for pattern in patterns:
                            st.write(f"• {pattern}")
                    else:
                        st.info("No significant patterns detected.")
            else:
                st.info("Advanced analysis is available for structured data (CSV, Excel) files.")
        
        with tab4:
            st.header("💡 AI-Powered Insights")
            
            if st.button("Generate AI Insights"):
                with st.spinner("Analyzing data and generating insights..."):
                    if is_dataframe and file_type in ['csv', 'excel']:
                        insights = st.session_state.agent.generate_insights(data)
                    else:
                        # For text documents, create a different prompt
                        prompt = f"""
                        Analyze this document and provide insights:
                        
                        Document Type: {file_type}
                        Content Length: {len(str(data))} characters
                        
                        Sample Content:
                        {str(data)[:1000]}...
                        
                        Please provide:
                        1. Key topics and themes
                        2. Content structure analysis
                        3. Important findings or information
                        4. Recommendations for further analysis
                        """
                        insights = st.session_state.agent.call_llama_model(prompt)
                    
                    st.markdown(insights)
            
            # Additional insight options
            st.subheader("Specific Insight Categories")
            insight_options = st.multiselect(
                "Select specific insights to generate:",
                ["Data Quality Assessment", "Trend Analysis", "Anomaly Detection", "Predictive Insights", "Business Recommendations"]
            )
            
            if insight_options and st.button("Generate Specific Insights"):
                for insight_type in insight_options:
                    with st.spinner(f"Generating {insight_type}..."):
                        if is_dataframe and file_type in ['csv', 'excel']:
                            prompt = f"""
                            Perform {insight_type} on this dataset:
                            
                            Dataset Info:
                            - Shape: {data.shape}
                            - Columns: {list(data.columns)}
                            - Sample: {data.head(3).to_string()}
                            
                            Provide detailed analysis focusing specifically on {insight_type}.
                            """
                        else:
                            prompt = f"""
                            Perform {insight_type} on this document:
                            
                            Content: {str(data)[:800]}...
                            
                            Focus specifically on {insight_type} aspects.
                            """
                        
                        insight = st.session_state.agent.call_llama_model(prompt)
                        st.subheader(insight_type)
                        st.markdown(insight)
        
        with tab5:
            st.header("❓ Q&A - Ask Anything About Your Data")
            
            # Chat interface
            st.subheader("Ask Questions About Your Data")
            
            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            if st.session_state.chat_history:
                st.subheader("Conversation History")
                for i, (question, answer) in enumerate(st.session_state.chat_history):
                    with st.expander(f"Q{i+1}: {question[:50]}..."):
                        st.write(f"**Question:** {question}")
                        st.write(f"**Answer:** {answer}")
            
            # Question input
            user_question = st.text_area(
                "Ask a question about your data:",
                placeholder="e.g., What are the main trends in this data? What columns have the most missing values? Can you explain the correlation between X and Y?",
                height=100
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("Ask Question", type="primary")
            with col2:
                clear_history = st.button("Clear History")
            
            if clear_history:
                st.session_state.chat_history = []
                st.session_state.agent.conversation_history = []
                st.rerun()
            
            if ask_button and user_question:
                with st.spinner("Analyzing and generating answer..."):
                    try:
                        # Get answer from the agent
                        answer = st.session_state.agent.answer_question(user_question)
                        
                        # Add to chat history
                        st.session_state.chat_history.append((user_question, answer))
                        
                        # Display the latest answer
                        st.subheader("Answer:")
                        st.markdown(answer)
                        
                        # Auto-scroll to show the new answer
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
            
            # Suggested questions based on data type
            st.subheader("💡 Suggested Questions")
            
            if file_type in ['csv', 'excel']:
                suggested_questions = [
                    "What are the key statistics for this dataset?",
                    "Which columns have missing values and how much?",
                    "What are the strongest correlations in the data?",
                    "Are there any outliers in the numeric columns?",
                    "What patterns do you see in this data?",
                    "Can you summarize the main characteristics of each column?",
                    "What insights can you derive from this dataset?",
                    "Are there any data quality issues I should be aware of?",
                    "What would be the best way to visualize this data?",
                    "Can you identify any trends or seasonality in the data?"
                ]
            else:
                suggested_questions = [
                    "What are the main topics discussed in this document?",
                    "Can you summarize the key points?",
                    "What is the overall tone or sentiment?",
                    "Are there any important dates or numbers mentioned?",
                    "What are the most frequently mentioned terms?",
                    "Can you extract the main conclusions or recommendations?",
                    "What questions does this document answer?",
                    "Are there any action items or next steps mentioned?",
                    "What is the structure of this document?",
                    "Can you identify any patterns in the content?"
                ]
            
            # Display suggested questions as clickable buttons
            st.write("Click on any suggested question:")
            for i, question in enumerate(suggested_questions[:6]):  # Show first 6
                if st.button(question, key=f"suggested_{i}"):
                    with st.spinner("Generating answer..."):
                        try:
                            answer = st.session_state.agent.answer_question(question)
                            st.session_state.chat_history.append((question, answer))
                            st.subheader("Answer:")
                            st.markdown(answer)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    
    else:
        # Welcome screen when no data is uploaded
        st.header("🚀 Welcome to the Intelligent Data Analyst Agent")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### What can this agent do?
            
            🔍 **Multi-format Support**: Upload CSV, Excel, PDF, Word documents, images, and text files
            
            📊 **Smart Analysis**: Automatic data profiling, pattern detection, and statistical analysis
            
            📈 **Intelligent Visualizations**: Auto-generated charts and graphs based on your data
            
            🤖 **AI-Powered Insights**: Get actionable insights and recommendations using advanced AI
            
            💬 **Interactive Q&A**: Ask questions about your data in natural language
            
            🔬 **Advanced Analytics**: Clustering, regression analysis, and statistical testing
            
            ### Getting Started:
            1. Enter your Together.ai API key in the sidebar
            2. Upload your data file
            3. Explore the different analysis tabs
            4. Ask questions about your data
            
            ### Supported File Types:
            - **Structured Data**: CSV, Excel (XLSX, XLS)
            - **Documents**: PDF, Word (DOCX), Text (TXT)
            - **Images**: PNG, JPG, JPEG, BMP, TIFF (with OCR)

            ### File Size Recommendations:
            - **Optimal**: Under 10MB for best performance
            - **Acceptable**: 10-50MB (may be slower)
            - **Maximum**: 100MB (may timeout or cause memory issues)
            - **Large datasets**: Consider sampling or preprocessing before upload

            ### Performance Tips:
            - CSV files generally process faster than Excel
            - Remove unnecessary columns before upload
            - Consider data sampling for very large datasets

            """)
        
        with col2:
            st.image("https://via.placeholder.com/300x400/0066cc/ffffff?text=Data+Analyst+Agent", 
                    caption="Your AI Data Analyst")
        
        # Show example questions
        st.subheader("🎯 Example Questions You Can Ask:")
        example_questions = [
            "What are the main trends in my sales data?",
            "Which features are most important for prediction?",
            "Are there any anomalies or outliers in the data?",
            "What's the correlation between different variables?",
            "Can you segment my customers based on their behavior?",
            "What insights can help improve business performance?",
            "How clean is my data and what issues should I address?",
            "What would be the best visualization for this data?"
        ]
        
        for i, question in enumerate(example_questions, 1):
            st.write(f"{i}. {question}")

# Additional helper functions that might be useful

def export_analysis_report(agent, data, file_type):
    """Generate a comprehensive analysis report"""
    report = f"""
    # Data Analysis Report
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ## Dataset Overview
    - File Type: {file_type}
    - Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}
    - Memory Usage: {data.memory_usage(deep=True).sum() / 1024:.1f KB if hasattr(data, 'memory_usage') else 'N/A'}
    
    ## Key Insights
    {agent.generate_insights(data) if hasattr(agent, 'generate_insights') else 'No insights available'}
    
    ## Conversation Summary
    """
    
    if agent.conversation_history:
        for i, (q, a) in enumerate(agent.conversation_history, 1):
            report += f"\n### Question {i}: {q}\n**Answer:** {a}\n"
    
    return report

def handle_file_upload_errors(uploaded_file, agent):
    """Handle common file upload errors"""
    try:
        data, file_type = agent.process_uploaded_file(uploaded_file)
        return data, file_type, None
    except UnicodeDecodeError:
        return None, None, "File encoding error. Try saving your file with UTF-8 encoding."
    except pd.errors.EmptyDataError:
        return None, None, "The uploaded file appears to be empty."
    except pd.errors.ParserError:
        return None, None, "Error parsing the file. Please check if it's a valid CSV/Excel file."
    except Exception as e:
        return None, None, f"Unexpected error: {str(e)}"

# Add to the main function before the existing tabs
def add_sidebar_features():
    # Footer copyright in the left sidebar bottom
    st.markdown(
        """
        <style>
        .alamin-footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 18rem;
            padding: 0.5em 0;
            color: gray;
            background: transparent;
            text-align: left;
            font-size: 0.9em;
            z-index: 9999;
        }
        </style>
        <div class='alamin-footer'>
            © 2025 Al Amin. Made with coffee and code ☕
        </div>
        """,
        unsafe_allow_html=True
    )
    """Add additional features to the sidebar"""
    st.sidebar.header("📊 Quick Stats")
    
    if st.session_state.uploaded_data is not None:
        data = st.session_state.uploaded_data
        file_type = st.session_state.file_type
        
        if file_type in ['csv', 'excel']:
            st.sidebar.metric("Rows", data.shape[0])
            st.sidebar.metric("Columns", data.shape[1])
            st.sidebar.metric("Missing Values", data.isnull().sum().sum())
            
            # Data type breakdown
            st.sidebar.subheader("Data Types")
            type_counts = data.dtypes.value_counts()
            for dtype, count in type_counts.items():
                st.sidebar.write(f"• {dtype}: {count}")
        
        else:
            # For text files
            content_length = len(str(data))
            word_count = len(str(data).split())
            st.sidebar.metric("Characters", content_length)
            st.sidebar.metric("Words", word_count)
    
    # Export functionality
    st.sidebar.header("📥 Export")
    if st.sidebar.button("Export Analysis Report"):
        if st.session_state.uploaded_data is not None:
            report = export_analysis_report(
                st.session_state.agent, 
                st.session_state.uploaded_data, 
                st.session_state.file_type
            )
            st.sidebar.download_button(
                label="Download Report",
                data=report,
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

# Add this call in the main function after API key setup
# add_sidebar_features()

if __name__ == "__main__":
    main()