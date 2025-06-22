import React from "react";
import {
  Box,
  TextField,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Button,
  Paper,
  Typography,
  Alert,
} from "@mui/material";
import {
  Send as SendIcon,
  Psychology as PsychologyIcon,
} from "@mui/icons-material";
import { useForm, Controller } from "react-hook-form";
import { SentimentFormData } from "../types/sentiment";

interface SentimentFormProps {
  onSubmit: (data: SentimentFormData) => void;
  loading?: boolean;
  error?: string | null;
}

const SentimentForm: React.FC<SentimentFormProps> = ({
  onSubmit,
  loading = false,
  error = null,
}) => {
  const {
    control,
    handleSubmit,
    formState: { errors },
    watch,
    reset,
  } = useForm<SentimentFormData>({
    defaultValues: {
      text: "",
      model: "both",
    },
  });

  const watchedText = watch("text");

  const handleFormSubmit = (data: SentimentFormData) => {
    onSubmit(data);
  };

  const handleClear = () => {
    reset();
  };

  const exampleTexts = [
    "This movie was absolutely amazing! I loved every minute of it.",
    "I'm really disappointed with this product. It doesn't work as expected.",
    "The weather is okay today, nothing special.",
    "I hate waiting in long lines at the store.",
    "The customer service was excellent and very helpful.",
  ];

  const handleExampleClick = (text: string) => {
    reset({ text, model: watch("model") });
  };

  return (
    <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
      <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
        <PsychologyIcon sx={{ mr: 1, color: "primary.main" }} />
        <Typography variant="h5" component="h2">
          Sentiment Analysis
        </Typography>
      </Box>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Enter text to analyze its sentiment using our LSTM and Verbalizer
        models.
      </Typography>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <form onSubmit={handleSubmit(handleFormSubmit)}>
        {/* Text Input */}
        <Controller
          name="text"
          control={control}
          rules={{
            required: "Please enter some text to analyze",
            minLength: {
              value: 5,
              message: "Text must be at least 5 characters long",
            },
            maxLength: {
              value: 1000,
              message: "Text must be less than 1000 characters",
            },
          }}
          render={({ field }) => (
            <TextField
              {...field}
              label="Text to Analyze"
              multiline
              rows={4}
              fullWidth
              variant="outlined"
              placeholder="Enter your text here..."
              error={!!errors.text}
              helperText={
                errors.text?.message || `${watchedText.length}/1000 characters`
              }
              disabled={loading}
              sx={{ mb: 3 }}
            />
          )}
        />

        {/* Model Selection */}
        <Controller
          name="model"
          control={control}
          render={({ field }) => (
            <FormControl component="fieldset" sx={{ mb: 3 }}>
              <FormLabel component="legend">Model Selection</FormLabel>
              <RadioGroup {...field} row aria-label="model selection">
                <FormControlLabel
                  value="lstm"
                  control={<Radio />}
                  label="LSTM Only"
                  disabled={loading}
                />
                <FormControlLabel
                  value="verbalizer"
                  control={<Radio />}
                  label="Verbalizer Only"
                  disabled={loading}
                />
                <FormControlLabel
                  value="both"
                  control={<Radio />}
                  label="Both Models"
                  disabled={loading}
                />
              </RadioGroup>
            </FormControl>
          )}
        />

        {/* Action Buttons */}
        <Box sx={{ display: "flex", gap: 2, mb: 3 }}>
          <Button
            type="submit"
            variant="contained"
            startIcon={<SendIcon />}
            disabled={loading || !watchedText.trim()}
            sx={{ minWidth: 120 }}
          >
            {loading ? "Analyzing..." : "Analyze"}
          </Button>

          <Button variant="outlined" onClick={handleClear} disabled={loading}>
            Clear
          </Button>
        </Box>
      </form>

      {/* Example Texts */}
      <Box>
        <Typography variant="subtitle2" gutterBottom>
          Try these examples:
        </Typography>
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
          {exampleTexts.map((text, index) => (
            <Button
              key={index}
              variant="text"
              size="small"
              onClick={() => handleExampleClick(text)}
              disabled={loading}
              sx={{
                textTransform: "none",
                fontSize: "0.75rem",
                maxWidth: 200,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              "{text.substring(0, 30)}..."
            </Button>
          ))}
        </Box>
      </Box>
    </Paper>
  );
};

export default SentimentForm;
