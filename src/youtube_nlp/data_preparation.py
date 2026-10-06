"""
Data Preparation & Linguistic Filtering Module for YouTube NLP Stance Detection.

This module provides a robust, object-oriented pipeline for extracting, filtering,
and validating massive volumes of social media data (YouTube comments) related to
electronic cigarettes (vaping) in the French context.

Academic & NLP Pipeline Context:
---------------------------------
Social media platforms present a highly noisy data environment containing off-topic
discussions, spam, and multilinguality. Before training a Deep Learning model (mBERT)
to detect public stance, it is imperative to construct a highly controlled dataset.

This module executes two fundamental preprocessing phases:
    1. Lexical Filtering: Videos are filtered based on a carefully curated dictionary
       of inclusion terms (e.g., "cigarette électronique", "vape") and exclusion
       terms (e.g., "music", "podcast") to ensure domain relevance.
    2. Deep Linguistic Validation (Cascading Architecture): Given the complexity of
       social media slang, a traditional string-matching approach is insufficient.
       We deploy a cascading AI architecture:
       a) Primary Filter: XLM-RoBERTa (a cross-lingual language model) performs
          high-confidence, context-aware language detection in batched tensors.
       b) Fallback Filter: A heuristic `langdetect` model catches edge cases (e.g.,
          extremely short slang phrases).
       Only comments validated as 'French' ('fr') by either model are retained,
       guaranteeing a pure linguistic dataset for subsequent stance detection.
"""

import pandas as pd
import re
import pickle
import torch
import logging
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from langdetect import detect, DetectorFactory

# Set seed for reproducible langdetect results
DetectorFactory.seed = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles video filtering and language detection for the YouTube NLP Pipeline.
    
    This class manages the lifecycle of raw CSV ingestion, lexical domain filtering,
    and batched GPU/CPU inference for XLM-RoBERTa language classification.
    """
    
    def __init__(self, raw_data_dir: str = "raw_data", clean_data_dir: str = "cleaned_data"):
        self.raw_dir = Path(raw_data_dir)
        self.clean_dir = Path(clean_data_dir)
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        
        # Keywords list (vaping equipment, brands, terminology)
        self.keywords = [
            "cigarette", "cigarettes", "cigarette électronique", "nicotine", "saveur",
            "mod", "résistance", "vapoter", "atomiseur", "ecig", "e-liquide", "juul",
            "smok", "geekvape", "vuse", "vaporesso", "vape", "e-cigarette", "vaping",
            "puff", "pod", "clearomiseur", "sub-ohm", "elfbar"
        ]
        
        self.exclude_words = ["podcast", "music", "type beat", "Lucid Dream", "Juice WRLD", "rugby", "asmr", "Food"]
        
    def filter_videos_by_keywords(self, videos_csv: str) -> pd.DataFrame:
        """Filters videos based on date, inclusion keywords, and exclusion keywords."""
        logger.info(f"Loading raw videos from {videos_csv}...")
        videos_df = pd.read_csv(self.raw_dir / videos_csv)
        video_df = videos_df.drop_duplicates(subset=['video_id']).copy()
        
        # Date filtering (2018 to 2023)
        video_df['published_at'] = video_df['published_at'].astype(str).str.slice(0, 10)
        video_df = video_df[(video_df['published_at'] >= '2018-01-01') & (video_df['published_at'] <= '2023-12-31')]
        
        # Keyword filtering
        exclude_pattern = r'\b(?:' + '|'.join(re.escape(w) for w in self.exclude_words) + r')\b'
        videos_with_keyword = video_df[video_df['title'].str.contains('|'.join(self.keywords), case=False, na=False)]
        filtered_videos = videos_with_keyword[~videos_with_keyword['title'].str.contains(exclude_pattern, case=False, na=False)]
        
        output_path = self.clean_dir / 'filtered_videos_with_keywords.csv'
        filtered_videos.to_csv(output_path, index=False)
        logger.info(f"Filtered down to {len(filtered_videos)} videos. Saved to {output_path}")
        
        return filtered_videos

    def _save_checkpoint(self, state_dict: dict, filename: str):
        with open(self.clean_dir / filename, "wb") as f:
            pickle.dump(state_dict, f)

    def detect_language_roberta(self, df_comments: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
        """Uses XLM-RoBERTa for batched language detection on comments."""
        model_ckpt = "papluca/xlm-roberta-base-language-detection"
        logger.info(f"Loading Roberta model: {model_ckpt}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to(device)
        
        dict_comment_language = {}
        checkpoint_file = "dict_language_detected.pkl"
        
        if (self.clean_dir / checkpoint_file).exists():
            with open(self.clean_dir / checkpoint_file, "rb") as f:
                dict_comment_language = pickle.load(f)
                logger.info(f"Loaded {len(dict_comment_language)} cached language predictions.")
        
        # Filter unprocessed
        unprocessed = [(row.comment_id, str(row.comment)) for row in df_comments.itertuples() 
                       if row.comment_id not in dict_comment_language]
        
        logger.info(f"Processing {len(unprocessed)} new comments for language detection...")
        
        save_count = 0
        save_interval = 20000
        
        for i in tqdm(range(0, len(unprocessed), batch_size), desc="RoBERTa Detection"):
            batch = unprocessed[i:i + batch_size]
            batch_ids, batch_comments = zip(*batch)
            
            inputs = tokenizer(list(batch_comments), padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = model(**inputs).logits
                
            preds = torch.softmax(logits, dim=-1).argmax(dim=1).tolist()
            
            for cid, idx in zip(batch_ids, preds):
                dict_comment_language[cid] = model.config.id2label[idx]
                save_count += 1
                
                if save_count >= save_interval:
                    self._save_checkpoint(dict_comment_language, checkpoint_file)
                    save_count = 0
                    
            del inputs, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        self._save_checkpoint(dict_comment_language, checkpoint_file)
        
        # Map back to DataFrame
        df_comments['language'] = df_comments['comment_id'].map(dict_comment_language)
        return df_comments

    def run_full_pipeline(self, comments_csv: str):
        """Runs the full data preparation and language filtering pipeline."""
        logger.info(f"Loading raw comments from {comments_csv}...")
        df_comments = pd.read_csv(self.raw_dir / comments_csv)
        
        df_comments = self.detect_language_roberta(df_comments)
        
        # Fallback for non-French detected by RoBERTa
        def detect_lang_fallback(text):
            try:
                return detect(text)
            except:
                return 'unidentified'
                
        mask = df_comments['language'] != 'fr'
        logger.info(f"Running langdetect fallback on {mask.sum()} non-French comments...")
        tqdm.pandas(desc="langdetect fallback")
        df_comments.loc[mask, 'language_langdetect'] = df_comments[mask]['comment'].progress_apply(detect_lang_fallback)
        
        # Final filter: Keep if either model says it's French
        df_comments_fr = df_comments[(df_comments['language'] == 'fr') | (df_comments['language_langdetect'] == 'fr')]
        
        output_path = self.clean_dir / "final_comments_fr_vape.csv"
        df_comments_fr.to_csv(output_path, index=False)
        logger.info(f"Pipeline complete! {len(df_comments_fr)} French comments saved to {output_path}")

if __name__ == "__main__":
    # Example usage:
    # processor = DataPreprocessor()
    # processor.filter_videos_by_keywords("videos_info.csv")
    # processor.run_full_pipeline("comments_info.csv")
    pass
