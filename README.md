# 🤖 Intelligent Data Analyst Agent

An AI-powered data analysis application that can analyze various file formats, generate insights, create visualizations, and answer questions about your data using advanced language models.

## 🚀 Features

- **Multi-Format Support**: CSV, Excel, PDF, Word, Text files, and Images (with OCR)
- **Smart Analysis**: Automatic data profiling, pattern detection, and statistical analysis
- **Interactive Visualizations**: Auto-generated charts, correlation heatmaps, and custom plots
- **AI-Powered Insights**: Natural language insights and business recommendations
- **Q&A System**: Ask questions about your data in plain English

## 🛠️ Installation

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd data-analyst-agent
   pip install -r requirements.txt
   ```

2. **Get API Key**
   - Sign up at [Together.ai](https://www.together.ai/)
   - Generate API key from dashboard

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

## 📦 Key Dependencies

```txt
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.15.0
scikit-learn>=1.3.0
PyPDF2>=3.0.0
python-docx>=0.8.11
pytesseract>=0.3.10
requests>=2.31.0
```

## 🎯 Usage

1. **Launch**: Run the Streamlit app
2. **API Key**: Enter your Together.ai API key
3. **Upload**: Choose your data file
4. **Analyze**: Explore through 5 main tabs:
   - 📊 **Data Overview**: Basic statistics and data structure
   - 📈 **Visualizations**: Interactive charts and plots
   - 🔍 **Advanced Analysis**: Statistical testing, clustering, regression
   - 💡 **AI Insights**: Automated insights and recommendations
   - ❓ **Q&A**: Natural language questions about your data

## 💬 Example Questions

- "What are the main trends in this data?"
- "Which columns have missing values?"
- "Can you explain the correlation patterns?"
- "Are there any outliers or anomalies?"
- "What insights can improve performance?"

## 🔧 Features by Tab

### Data Overview
- Dataset structure and statistics
- Data types and missing values
- Sample data preview

### Visualizations
- Auto-generated charts based on data type
- Custom plot creation
- Correlation heatmaps and distribution plots

### Advanced Analysis
- Statistical testing and pattern detection
- K-means clustering
- Regression analysis

### AI Insights
- Automated business insights
- Trend and anomaly detection
- Actionable recommendations

### Q&A System
- Natural language query processing
- Conversation history
- Context-aware responses

## 🐛 Troubleshooting

- **File Upload Issues**: Ensure UTF-8 encoding for CSV files
- **API Errors**: Verify Together.ai API key is valid
- **Performance**: Consider smaller files for initial testing
- **OCR**: Install Tesseract for image text extraction

## 📊 Supported Analysis

- Descriptive statistics and data profiling
- Correlation analysis and pattern detection
- Outlier identification and missing data analysis
- Clustering and segmentation
- Predictive modeling and trend analysis
- Natural language insights generation

Built with Streamlit and powered by Together.ai's language models.