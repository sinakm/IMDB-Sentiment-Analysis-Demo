import React, { useState, useEffect } from "react";
import {
  Container,
  Grid,
  Typography,
  Box,
  Chip,
  Paper,
  Divider,
} from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import {
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Psychology as PsychologyIcon,
} from "@mui/icons-material";

import SentimentForm from "./components/SentimentForm";
import SentimentGauge from "./components/SentimentGauge";
import SentimentApiService from "./services/sentimentApi";
import {
  SentimentFormData,
  SentimentResponse,
  SentimentConsensus,
} from "./types/sentiment";

// Create Material-UI theme
const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#1976d2",
    },
    secondary: {
      main: "#dc004e",
    },
    background: {
      default: "#f5f5f5",
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 500,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
  },
});

function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<SentimentResponse | null>(null);
  const [apiHealthy, setApiHealthy] = useState<boolean | null>(null);

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      await SentimentApiService.checkHealth();
      setApiHealthy(true);
    } catch (error) {
      console.error("API health check failed:", error);
      setApiHealthy(false);
    }
  };

  const handleSentimentAnalysis = async (formData: SentimentFormData) => {
    setLoading(true);
    setError(null);

    try {
      const result = await SentimentApiService.analyzeSentiment({
        text: formData.text,
        model: formData.model,
      });

      setResponse(result);
    } catch (err: any) {
      setError(err.message || "Failed to analyze sentiment");
      setResponse(null);
    } finally {
      setLoading(false);
    }
  };

  const getConsensusColor = (consensus: SentimentConsensus) => {
    if (consensus.prediction === "positive") return "success";
    if (consensus.prediction === "negative") return "error";
    if (consensus.prediction === "neutral") return "warning";
    return "default";
  };

  const getConsensusIcon = (consensus: SentimentConsensus) => {
    if (consensus.prediction === "positive") return "😊";
    if (consensus.prediction === "negative") return "😞";
    if (consensus.prediction === "neutral") return "😐";
    return "❓";
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Header */}
        <Box sx={{ textAlign: "center", mb: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            🤖 AI Sentiment Analysis
          </Typography>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            Powered by LSTM and ModernBERT Verbalizer Models
          </Typography>

          {/* API Status */}
          <Box sx={{ mt: 2 }}>
            {apiHealthy === true && (
              <Chip
                icon={<TrendingUpIcon />}
                label="API Online"
                color="success"
                variant="outlined"
              />
            )}
            {apiHealthy === false && (
              <Chip
                icon={<TrendingUpIcon />}
                label="API Offline"
                color="error"
                variant="outlined"
              />
            )}
            {apiHealthy === null && (
              <Chip
                icon={<TrendingUpIcon />}
                label="Checking API..."
                color="default"
                variant="outlined"
              />
            )}
          </Box>
        </Box>

        {/* Main Content */}
        <Grid container spacing={3}>
          {/* Form Section */}
          <Grid item xs={12}>
            <SentimentForm
              onSubmit={handleSentimentAnalysis}
              loading={loading}
              error={error}
            />
          </Grid>

          {/* Results Section */}
          {response && (
            <>
              {/* Individual Model Results */}
              <Grid item xs={12} md={6}>
                <SentimentGauge
                  title="🧠 LSTM Model"
                  prediction={response.predictions.lstm}
                  loading={loading}
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <SentimentGauge
                  title="🤖 Verbalizer Model"
                  prediction={response.predictions.verbalizer}
                  loading={loading}
                />
              </Grid>

              {/* Consensus Results */}
              <Grid item xs={12}>
                <Paper elevation={2} sx={{ p: 3 }}>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      mb: 2,
                    }}
                  >
                    <PsychologyIcon sx={{ mr: 1, color: "primary.main" }} />
                    <Typography variant="h6">Consensus Analysis</Typography>
                  </Box>

                  <Grid container spacing={3} alignItems="center">
                    <Grid item xs={12} md={4}>
                      <Box sx={{ textAlign: "center" }}>
                        <Typography variant="h2" sx={{ mb: 1 }}>
                          {getConsensusIcon(response.consensus)}
                        </Typography>
                        <Chip
                          label={response.consensus.prediction.toUpperCase()}
                          color={getConsensusColor(response.consensus)}
                          sx={{
                            fontWeight: "bold",
                            fontSize: "1rem",
                            px: 2,
                            py: 1,
                          }}
                        />
                      </Box>
                    </Grid>

                    <Grid item xs={12} md={8}>
                      <Box>
                        <Typography variant="body1" gutterBottom>
                          <strong>Average Confidence:</strong>{" "}
                          {(response.consensus.avg_confidence * 100).toFixed(1)}
                          %
                        </Typography>
                        <Typography variant="body1" gutterBottom>
                          <strong>Models Agreement:</strong>{" "}
                          {response.consensus.agreement ? "✅ Yes" : "❌ No"}
                        </Typography>
                        <Typography variant="body1" gutterBottom>
                          <strong>Models Used:</strong>{" "}
                          {response.consensus.models_count} of{" "}
                          {response.models_available}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          <strong>Processing Time:</strong>{" "}
                          {response.total_processing_time_ms}ms
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>

                  <Divider sx={{ my: 2 }} />

                  {/* Analysis Details */}
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Analysis Details:
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Text:</strong> "{response.text}"
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Request ID:</strong> {response.lambda_request_id}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>API Version:</strong> {response.version}
                    </Typography>
                  </Box>
                </Paper>
              </Grid>
            </>
          )}

          {/* Instructions */}
          {!response && !loading && (
            <Grid item xs={12}>
              <Paper elevation={1} sx={{ p: 3, textAlign: "center" }}>
                <SpeedIcon
                  sx={{ fontSize: 48, color: "primary.main", mb: 2 }}
                />
                <Typography variant="h6" gutterBottom>
                  Ready to Analyze Sentiment
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Enter some text above and select a model to get started. Our
                  AI models will analyze the sentiment and provide confidence
                  scores with beautiful gauge visualizations.
                </Typography>
              </Paper>
            </Grid>
          )}
        </Grid>

        {/* Footer */}
        <Box sx={{ textAlign: "center", mt: 6, py: 3 }}>
          <Typography variant="body2" color="text.secondary">
            Built with React, Material-UI, and AWS Lambda
          </Typography>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
