import React from "react";
import { Box, Typography, Card, CardContent, Chip } from "@mui/material";
import { Gauge } from "@mui/x-charts/Gauge";
import { SentimentPrediction } from "../types/sentiment";

interface SentimentGaugeProps {
  title: string;
  prediction: SentimentPrediction | undefined;
  loading?: boolean;
}

const SentimentGauge: React.FC<SentimentGaugeProps> = ({
  title,
  prediction,
  loading = false,
}) => {
  // Determine gauge color based on sentiment
  const getGaugeColor = (sentiment: string): string => {
    switch (sentiment) {
      case "positive":
        return "#4caf50"; // Green
      case "negative":
        return "#f44336"; // Red
      case "neutral":
        return "#ff9800"; // Orange
      case "error":
        return "#9e9e9e"; // Gray
      default:
        return "#2196f3"; // Blue
    }
  };

  // Get sentiment icon/emoji
  const getSentimentIcon = (sentiment: string): string => {
    switch (sentiment) {
      case "positive":
        return "😊";
      case "negative":
        return "😞";
      case "neutral":
        return "😐";
      case "error":
        return "❌";
      default:
        return "🤔";
    }
  };

  // Convert confidence to percentage (0-100)
  const confidencePercentage = prediction
    ? Math.round(prediction.confidence * 100)
    : 0;
  const gaugeColor = prediction
    ? getGaugeColor(prediction.prediction)
    : "#e0e0e0";
  const sentimentIcon = prediction
    ? getSentimentIcon(prediction.prediction)
    : "⏳";

  return (
    <Card
      sx={{
        minHeight: 300,
        display: "flex",
        flexDirection: "column",
        opacity: loading ? 0.6 : 1,
        transition: "opacity 0.3s ease",
      }}
    >
      <CardContent
        sx={{
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        {/* Title */}
        <Typography variant="h6" component="h3" gutterBottom align="center">
          {title}
        </Typography>

        {/* Loading state */}
        {loading && (
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              mt: 2,
            }}
          >
            <Typography variant="body2" color="text.secondary">
              Analyzing...
            </Typography>
          </Box>
        )}

        {/* Gauge */}
        {!loading && (
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              flexGrow: 1,
            }}
          >
            <Box sx={{ position: "relative", mb: 2 }}>
              <Gauge
                width={200}
                height={200}
                value={confidencePercentage}
                startAngle={-110}
                endAngle={110}
                sx={{
                  [`& .MuiGauge-valueText`]: {
                    fontSize: 20,
                    fontWeight: "bold",
                    fill: gaugeColor,
                  },
                  [`& .MuiGauge-valueArc`]: {
                    fill: gaugeColor,
                  },
                  [`& .MuiGauge-referenceArc`]: {
                    fill: "#e0e0e0",
                  },
                }}
                text={({ value }) => `${value}%`}
              />

              {/* Sentiment icon overlay - positioned well below the percentage */}
              <Box
                sx={{
                  position: "absolute",
                  top: "85%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                  fontSize: "2rem",
                }}
              >
                {sentimentIcon}
              </Box>
            </Box>

            {/* Prediction details */}
            {prediction && (
              <Box sx={{ textAlign: "center", width: "100%" }}>
                <Chip
                  label={prediction.prediction.toUpperCase()}
                  color={
                    prediction.prediction === "positive"
                      ? "success"
                      : prediction.prediction === "negative"
                      ? "error"
                      : prediction.prediction === "neutral"
                      ? "warning"
                      : "default"
                  }
                  sx={{ mb: 1, fontWeight: "bold" }}
                />

                <Typography variant="body2" color="text.secondary">
                  Confidence: {(prediction.confidence * 100).toFixed(1)}%
                </Typography>

                <Typography variant="caption" color="text.secondary">
                  Processing: {prediction.processing_time_ms}ms
                </Typography>

                {/* Error message */}
                {prediction.error && (
                  <Typography variant="body2" color="error" sx={{ mt: 1 }}>
                    {prediction.error}
                  </Typography>
                )}
              </Box>
            )}

            {/* No prediction state */}
            {!prediction && !loading && (
              <Box sx={{ textAlign: "center", mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Enter text to analyze sentiment
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default SentimentGauge;
