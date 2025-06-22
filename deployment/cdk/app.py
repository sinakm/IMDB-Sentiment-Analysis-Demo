#!/usr/bin/env python3
"""
AWS CDK app for deploying the sentiment analysis API.
"""

import aws_cdk as cdk
from sentiment_stack import SentimentAnalysisStack

app = cdk.App()

# Create the sentiment analysis stack
SentimentAnalysisStack(
    app, 
    "SentimentAnalysisStack",
    description="Sentiment Analysis API with LSTM, ModernBERT, and Verbalizer models",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region="us-east-1"  # As requested by user
    )
)

app.synth()
