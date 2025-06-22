export interface SentimentPrediction {
  prediction: "positive" | "negative" | "neutral" | "error";
  confidence: number;
  probability: number;
  processing_time_ms: number;
  error?: string;
}

export interface SentimentConsensus {
  prediction: "positive" | "negative" | "neutral" | "error";
  agreement: boolean;
  avg_confidence: number;
  models_count: number;
}

export interface SentimentResponse {
  text: string;
  predictions: {
    lstm?: SentimentPrediction;
    verbalizer?: SentimentPrediction;
  };
  consensus: SentimentConsensus;
  total_processing_time_ms: number;
  models_available: number;
  models: string[];
  lambda_request_id: string;
  version: string;
  model_preference: "lstm" | "verbalizer" | "both";
}

export interface SentimentRequest {
  text: string;
  model: "lstm" | "verbalizer" | "both";
}

export type ModelType = "lstm" | "verbalizer" | "both";

export interface SentimentFormData {
  text: string;
  model: ModelType;
}
