# Sentiment Analysis Frontend

A modern React TypeScript application for sentiment analysis using LSTM and ModernBERT Verbalizer models.

## 🎯 Features

- **Beautiful Material-UI Interface**: Professional design with responsive layout
- **Dual Model Support**: LSTM and Verbalizer models with consensus predictions
- **Interactive Gauges**: Visual sentiment confidence scores with color-coded indicators
- **Real-time API Integration**: Live sentiment analysis with error handling
- **Form Validation**: Comprehensive input validation and user feedback
- **Example Text Buttons**: Quick testing with pre-defined examples
- **API Health Monitoring**: Real-time API status indicators

## 🏗️ Architecture

### Frontend Stack

- **React 18** with TypeScript
- **Material-UI (MUI) v5** for components and theming
- **React Hook Form** for form management
- **Axios** for API communication
- **MUI X Charts** for gauge visualizations

### AWS Deployment

- **S3 Bucket**: Static website hosting
- **CloudFront**: Global CDN with caching
- **CDK**: Infrastructure as Code deployment
- **Docker**: Containerized build process

## 🚀 Quick Start

### Local Development

1. **Install Dependencies**

   ```bash
   cd deployment/frontend
   npm install
   ```

2. **Set Environment Variables**

   ```bash
   # Create .env.local file
   REACT_APP_API_URL=https://your-api-gateway-url.amazonaws.com/prod
   REACT_APP_API_KEY=your-api-key-here
   ```

3. **Start Development Server**

   ```bash
   npm start
   ```

4. **Open Browser**
   Navigate to `http://localhost:3000`

### Production Deployment

Use the automated deployment script:

```bash
cd ../../scripts
python deploy_with_frontend.py --environment prod --email your@email.com
```

## 📁 Project Structure

```
deployment/frontend/
├── public/
│   ├── index.html          # HTML template
│   └── manifest.json       # PWA manifest
├── src/
│   ├── components/         # React components
│   │   ├── SentimentForm.tsx    # Input form component
│   │   └── SentimentGauge.tsx   # Gauge visualization
│   ├── services/          # API services
│   │   └── sentimentApi.ts      # API client
│   ├── types/             # TypeScript definitions
│   │   └── sentiment.ts         # Type definitions
│   ├── App.tsx            # Main application component
│   ├── index.tsx          # Application entry point
│   └── index.css          # Global styles
├── package.json           # Dependencies and scripts
└── README.md             # This file
```

## 🎨 Components

### SentimentForm

- Multi-line text input with character counter
- Model selection (LSTM, Verbalizer, Both)
- Form validation and error handling
- Example text buttons for quick testing
- Loading states and user feedback

### SentimentGauge

- Circular gauge visualization
- Color-coded sentiment indicators
- Confidence percentage display
- Sentiment emoji icons
- Processing time metrics
- Error state handling

### App

- Main application layout
- API health monitoring
- Consensus analysis display
- Responsive Material-UI theme
- Real-time status indicators

## 🔧 Configuration

### Environment Variables

| Variable             | Description                | Default               |
| -------------------- | -------------------------- | --------------------- |
| `REACT_APP_API_URL`  | API Gateway base URL       | Local development URL |
| `REACT_APP_API_KEY`  | API key for authentication | Development key       |
| `NODE_ENV`           | Build environment          | `production`          |
| `GENERATE_SOURCEMAP` | Include source maps        | `false`               |

### Build Configuration

The application is configured for:

- **Production builds** with optimizations
- **TypeScript** strict mode
- **ESLint** code quality checks
- **Responsive design** for all devices
- **PWA capabilities** (future enhancement)

## 🚀 Deployment

### CDK Deployment

The frontend is deployed using AWS CDK with:

1. **S3 Bucket** for static hosting
2. **CloudFront Distribution** for global CDN
3. **Origin Access Identity** for security
4. **Cache Policies** optimized for SPA
5. **Error Pages** for client-side routing

### Build Process

During deployment:

1. Install Node.js dependencies
2. Set environment variables
3. Build React application
4. Upload to S3 bucket
5. Invalidate CloudFront cache
6. Configure proper headers

### Security Features

- **HTTPS Enforcement** via CloudFront
- **Security Headers** for XSS protection
- **CORS Configuration** for API access
- **Private S3 Bucket** with OAI access
- **API Key Authentication** for backend

## 🎯 Usage

### Basic Sentiment Analysis

1. **Enter Text**: Type or paste text in the input field
2. **Select Model**: Choose LSTM, Verbalizer, or Both
3. **Click Analyze**: Submit for sentiment analysis
4. **View Results**: See gauge visualizations and consensus

### Understanding Results

- **Positive Sentiment**: Green gauge with 😊 emoji
- **Negative Sentiment**: Red gauge with 😞 emoji
- **Neutral Sentiment**: Orange gauge with 😐 emoji
- **Confidence Score**: Percentage indicating model certainty
- **Processing Time**: Milliseconds for each model
- **Consensus**: Combined prediction from both models

### Example Texts

The application includes example texts for quick testing:

- Positive: "This movie was absolutely amazing!"
- Negative: "I'm really disappointed with this product."
- Neutral: "The weather is okay today, nothing special."

## 🔍 API Integration

### Endpoints

- **Health Check**: `GET /health`
- **Sentiment Analysis**: `POST /predict`

### Request Format

```json
{
  "text": "Your text here",
  "model": "both" // "lstm", "verbalizer", or "both"
}
```

### Response Format

```json
{
  "text": "Your text here",
  "predictions": {
    "lstm": {
      "prediction": "positive",
      "confidence": 0.95,
      "processing_time_ms": 150
    },
    "verbalizer": {
      "prediction": "positive",
      "confidence": 0.92,
      "processing_time_ms": 200
    }
  },
  "consensus": {
    "prediction": "positive",
    "avg_confidence": 0.935,
    "agreement": true,
    "models_count": 2
  },
  "total_processing_time_ms": 350,
  "lambda_request_id": "abc123",
  "version": "1.0.0"
}
```

## 🛠️ Development

### Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run test suite
- `npm run eject` - Eject from Create React App

### Code Quality

- **TypeScript** for type safety
- **ESLint** for code quality
- **Prettier** for code formatting
- **React Hooks** for state management
- **Material-UI** for consistent design

### Testing

```bash
npm test
```

Runs the test suite with:

- Component testing
- API service testing
- Type checking
- Accessibility testing

## 📊 Monitoring

### CloudWatch Metrics

The deployment includes monitoring for:

- **CloudFront Requests**: Request count and errors
- **S3 Operations**: Upload and download metrics
- **API Gateway**: Backend API performance
- **Lambda Functions**: Execution metrics

### Performance Optimization

- **Code Splitting**: Automatic chunk splitting
- **Tree Shaking**: Remove unused code
- **Compression**: Gzip compression enabled
- **Caching**: Optimized cache headers
- **CDN**: Global content delivery

## 🔧 Troubleshooting

### Common Issues

1. **API Connection Failed**

   - Check API URL in environment variables
   - Verify API key is correct
   - Ensure CORS is configured

2. **Build Failures**

   - Clear node_modules and reinstall
   - Check Node.js version compatibility
   - Verify all dependencies are installed

3. **Deployment Issues**
   - Ensure AWS credentials are configured
   - Check CDK bootstrap status
   - Verify S3 bucket permissions

### Debug Mode

Enable debug logging:

```bash
REACT_APP_DEBUG=true npm start
```

## 📝 License

This project is part of the Sentiment Analysis application suite.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:

- Check the troubleshooting section
- Review CloudWatch logs
- Contact the development team
