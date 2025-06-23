import axios from "axios";
import { SentimentRequest, SentimentResponse } from "../types/sentiment";

// API configuration - loaded from build-time environment variables
const getApiConfig = () => {
  return {
    baseUrl: process.env.REACT_APP_API_URL || "https://api.placeholder.com/",
    apiKey: process.env.REACT_APP_API_KEY_ID || "placeholder-key-id",
  };
};

const config = getApiConfig();
const API_BASE_URL = config.baseUrl;
const API_KEY = config.apiKey;

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    "Content-Type": "application/json",
    "x-api-key": API_KEY,
  },
});

// Add request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log("API Request:", {
      url: config.url,
      method: config.method,
      data: config.data,
    });
    return config;
  },
  (error) => {
    console.error("API Request Error:", error);
    return Promise.reject(error);
  }
);

// Add response interceptor for logging and error handling
apiClient.interceptors.response.use(
  (response) => {
    console.log("API Response:", {
      status: response.status,
      data: response.data,
    });
    return response;
  },
  (error) => {
    console.error("API Response Error:", {
      status: error.response?.status,
      data: error.response?.data,
      message: error.message,
    });
    return Promise.reject(error);
  }
);

export class SentimentApiService {
  /**
   * Check if the API is healthy
   */
  static async checkHealth(): Promise<{ status: string; message: string }> {
    try {
      const response = await apiClient.get("/health");
      return response.data;
    } catch (error) {
      console.error("Health check failed:", error);
      throw new Error("API health check failed");
    }
  }

  /**
   * Analyze sentiment of text
   */
  static async analyzeSentiment(
    request: SentimentRequest
  ): Promise<SentimentResponse> {
    try {
      const response = await apiClient.post<SentimentResponse>(
        "/predict",
        request
      );
      return response.data;
    } catch (error: any) {
      console.error("Sentiment analysis failed:", error);

      // Handle different types of errors
      if (error.response?.status === 403) {
        throw new Error("API key is invalid or missing");
      } else if (error.response?.status === 429) {
        throw new Error("Too many requests. Please try again later.");
      } else if (error.response?.status >= 500) {
        throw new Error("Server error. Please try again later.");
      } else if (error.code === "ECONNABORTED") {
        throw new Error("Request timeout. Please try again.");
      } else {
        throw new Error(
          error.response?.data?.message || "Failed to analyze sentiment"
        );
      }
    }
  }

  /**
   * Get API configuration info
   */
  static getApiInfo() {
    return {
      baseUrl: API_BASE_URL,
      hasApiKey: !!API_KEY,
      timeout: apiClient.defaults.timeout,
    };
  }
}

export default SentimentApiService;
