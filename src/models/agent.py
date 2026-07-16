#!/usr/bin/env python3
"""
Agent Wrapper
=============
Wraps OpenAI API calls for the 3 BehAv-PO agents (Conservative, Mainstream, Sensitive).
Handles batch inference, YES/NO parsing, and model management.
"""

import json
import os
import sys
import time

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Agent:
    """A single BehAv-PO agent backed by a fine-tuned GPT-4o-mini model."""

    def __init__(self, cluster_name, model_id=None):
        """
        Args:
            cluster_name: "Conservative", "Mainstream", or "Sensitive"
            model_id: OpenAI model ID (e.g. "ft:gpt-4o-mini:..."). If None, uses base model.
        """
        self.cluster_name = cluster_name
        self.model_id = model_id or config.BASE_MODEL
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def predict(self, tweet, temperature=0.0):
        """
        Predict YES/NO for a single tweet.

        Returns:
            dict with keys: label ("YES"/"NO"), raw_response (str)
        """
        # Clean tweet text for API compatibility
        tweet_clean = tweet.replace("\x00", "").strip()
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Tweet: {tweet_clean}"},
                ],
                temperature=temperature,
                max_tokens=3,
            )
            raw = response.choices[0].message.content.strip().upper()
        except Exception:
            raw = "NO"  # fallback on API error
        label = "YES" if "YES" in raw else "NO"
        return {"label": label, "raw_response": raw}

    def predict_batch(self, tweets, temperature=0.0, delay=0.0):
        """
        Predict YES/NO for a list of tweets.

        Args:
            tweets: list of tweet strings
            temperature: sampling temperature (0.0 for deterministic)
            delay: seconds between API calls (for rate limiting)

        Returns:
            list of dicts with keys: label, raw_response
        """
        results = []
        for i, tweet in enumerate(tweets):
            result = self.predict(tweet, temperature=temperature)
            results.append(result)
            if delay > 0 and i < len(tweets) - 1:
                time.sleep(delay)
        return results

    def sample_multiple(self, tweet, k=8, temperature=1.0):
        """
        Sample K completions for a single tweet (for GRPO rejection sampling).

        Returns:
            list of K dicts with keys: label, raw_response
        """
        results = []
        for _ in range(k):
            result = self.predict(tweet, temperature=temperature)
            results.append(result)
        return results


class MultiAgentSystem:
    """Manages the 3-agent system and handles team voting."""

    def __init__(self, model_ids=None):
        """
        Args:
            model_ids: dict mapping cluster name -> model ID.
                       If None, all agents use the base model.
        """
        model_ids = model_ids or {}
        self.agents = {
            name: Agent(name, model_ids.get(name))
            for name in config.CLUSTER_NAMES
        }

    def predict(self, tweet, temperature=0.0):
        """
        Run all 3 agents on a tweet and compute majority vote.

        Returns:
            dict with keys:
                agent_predictions: {cluster: label}
                team_vote: "YES"/"NO"
                unanimous: bool
        """
        predictions = {}
        for name, agent in self.agents.items():
            result = agent.predict(tweet, temperature=temperature)
            predictions[name] = result["label"]

        yes_count = sum(1 for v in predictions.values() if v == "YES")
        team_vote = "YES" if yes_count >= 2 else "NO"
        unanimous = len(set(predictions.values())) == 1

        return {
            "agent_predictions": predictions,
            "team_vote": team_vote,
            "unanimous": unanimous,
        }

    def predict_batch(self, tweets, temperature=0.0, delay=0.05):
        """
        Run all 3 agents on a list of tweets.

        Returns:
            list of prediction dicts (same format as predict())
        """
        results = []
        for i, tweet in enumerate(tweets):
            result = self.predict(tweet, temperature=temperature)
            results.append(result)
            if delay > 0 and i < len(tweets) - 1:
                time.sleep(delay)
        return results

    def save_model_ids(self, path):
        """Save model IDs to JSON for reproducibility."""
        ids = {name: agent.model_id for name, agent in self.agents.items()}
        with open(path, "w") as f:
            json.dump(ids, f, indent=2)

    @classmethod
    def load(cls, path):
        """Load multi-agent system from saved model IDs."""
        with open(path) as f:
            model_ids = json.load(f)
        return cls(model_ids=model_ids)
