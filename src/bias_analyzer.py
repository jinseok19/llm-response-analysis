import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class BiasAnalyzer:
    """
    BERT 기반 편향 분석기
    특정 국가/정권에 대한 편향을 정량화
    """
    
    def __init__(self, model_name="bert-base-uncased", use_gpu=False):
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # 메모리 절약을 위해 지연 로딩
        self.tokenizer = None
        self.model = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # spaCy 모델 로드 (개체명 인식용)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("spaCy 모델이 설치되지 않았습니다. 'python -m spacy download en_core_web_sm' 실행 필요")
            self.nlp = None
            
        # 타겟 국가/정권 리스트
        self.target_entities = {
            'china': ['China', 'Chinese', 'Beijing', 'Xi Jinping', 'CCP'],
            'north_korea': ['North Korea', 'DPRK', 'Kim Jong-un', 'Pyongyang'],
            'usa': ['USA', 'United States', 'America', 'Washington', 'Biden'],
            'russia': ['Russia', 'Russian', 'Putin', 'Moscow']
        }
    
    def _load_bert_model(self):
        """BERT 모델 지연 로딩"""
        if self.tokenizer is None:
            print("BERT 모델 로딩 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            if self.use_gpu:
                self.model = self.model.cuda()
            else:
                self.model = self.model.cpu()
            
            # 메모리 절약을 위한 설정
            self.model.eval()
            print("BERT 모델 로딩 완료")
    
    def extract_entities(self, text):
        """개체명 인식"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'PERSON', 'ORG']:  # 국가, 인물, 조직
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        return entities
    
    def get_sentiment_scores(self, text):
        """감정 분석 점수 계산"""
        # VADER 감정 분석
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob 감정 분석
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'vader_compound': vader_scores['compound'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity
        }
    
    def get_bert_embeddings(self, text):
        """BERT 임베딩 추출 (메모리 절약 버전)"""
        self._load_bert_model()
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # [CLS] 토큰의 임베딩 사용 (문장 전체 표현)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings.flatten()
    
    def analyze_bias_towards_entity(self, text, target_entity):
        """특정 개체에 대한 편향 분석"""
        # 개체명 인식
        entities = self.extract_entities(text)
        
        # 타겟 개체가 텍스트에 있는지 확인 (개선된 버전)
        target_found = False
        target_keywords = self.target_entities.get(target_entity, [])
        
        # 1. spaCy 개체명 인식 결과 확인
        for entity in entities:
            if any(target.lower() in entity['text'].lower() for target in target_keywords):
                target_found = True
                break
        
        # 2. 직접 키워드 검색 (백업 방법)
        if not target_found:
            text_lower = text.lower()
            for keyword in target_keywords:
                if keyword.lower() in text_lower:
                    target_found = True
                    break
        
        if not target_found:
            return {
                'target_found': False,
                'bias_score': 0,
                'sentiment_scores': None,
                'stance': 'neutral'
            }
        
        # 감정 분석
        sentiment_scores = self.get_sentiment_scores(text)
        
        # 편향 점수 계산 (compound score 기반)
        bias_score = sentiment_scores['vader_compound']
        
        # 입장 분류
        if bias_score > 0.1:
            stance = 'positive'
        elif bias_score < -0.1:
            stance = 'negative'
        else:
            stance = 'neutral'
        
        return {
            'target_found': True,
            'bias_score': bias_score,
            'sentiment_scores': sentiment_scores,
            'stance': stance,
            'entities': entities
        }
    
    def analyze_multiple_entities(self, text):
        """여러 개체에 대한 편향 분석"""
        results = {}
        
        for entity_name in self.target_entities.keys():
            results[entity_name] = self.analyze_bias_towards_entity(text, entity_name)
        
        return results
    
    def compare_models_bias(self, responses_dict):
        """여러 모델의 응답에 대한 편향 비교"""
        comparison_results = {}
        
        for model_name, response in responses_dict.items():
            comparison_results[model_name] = self.analyze_multiple_entities(response)
        
        return comparison_results 