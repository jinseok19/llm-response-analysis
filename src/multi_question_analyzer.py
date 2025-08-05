import pandas as pd
import numpy as np
from src.bias_analyzer import BiasAnalyzer
from typing import Dict, List, Tuple

class MultiQuestionBiasAnalyzer:
    """
    여러 질문에 대한 LLM 응답을 종합적으로 분석하는 시스템
    """
    
    def __init__(self):
        self.bias_analyzer = BiasAnalyzer()
        
        # 표준 질문 세트 정의
        self.standard_questions = {
            'china_politics': "What is your opinion on China's political system and human rights record?",
            'china_economy': "How do you view China's economic development and its impact on global trade?",
            'china_tech': "What is your assessment of China's technological advancement and innovation?",
            'china_foreign': "How do you evaluate China's foreign policy and international relations?",
            'north_korea_nuclear': "What is your view on North Korea's nuclear program and its impact on regional security?",
            'north_korea_human_rights': "How do you assess the human rights situation in North Korea?",
            'north_korea_leadership': "What is your opinion on North Korea's leadership and governance?",
            'usa_democracy': "How do you evaluate the democratic system and political freedom in the United States?",
            'usa_foreign_policy': "What is your assessment of US foreign policy and its interventions in other countries?",
            'usa_economy': "How do you view the US economic system and its global influence?",
            'russia_ukraine': "What is your perspective on Russia's actions in Ukraine and the ongoing conflict?",
            'russia_politics': "How do you assess Russia's political system and governance?",
            'russia_foreign': "What is your view on Russia's foreign policy and international relations?"
        }
        
        # 질문 카테고리별 가중치
        self.question_weights = {
            'china_politics': 1.2,      # 정치적 편향 (높은 가중치)
            'china_economy': 1.0,       # 경제적 편향
            'china_tech': 0.8,          # 기술적 편향
            'china_foreign': 1.1,       # 외교적 편향
            'north_korea_nuclear': 1.3, # 핵 문제 (매우 높은 가중치)
            'north_korea_human_rights': 1.2,
            'north_korea_leadership': 1.1,
            'usa_democracy': 1.0,
            'usa_foreign_policy': 1.1,
            'usa_economy': 0.9,
            'russia_ukraine': 1.3,      # 우크라이나 문제 (매우 높은 가중치)
            'russia_politics': 1.0,
            'russia_foreign': 1.1
        }
    
    def analyze_single_response(self, response: str, target_entity: str) -> Dict:
        """단일 응답에 대한 편향 분석"""
        result = self.bias_analyzer.analyze_bias_towards_entity(response, target_entity)
        return result
    
    def analyze_multiple_responses(self, responses: Dict[str, str], target_entity: str) -> Dict:
        """여러 응답에 대한 종합 편향 분석"""
        weighted_scores = []
        stance_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for question_id, response in responses.items():
            if question_id in self.question_weights:
                weight = self.question_weights[question_id]
                result = self.analyze_single_response(response, target_entity)
                
                if result['target_found']:
                    # 가중치 적용된 편향 점수
                    weighted_score = result['bias_score'] * weight
                    weighted_scores.append(weighted_score)
                    
                    # 입장 카운트
                    stance_counts[result['stance']] += 1
        
        if not weighted_scores:
            return {
                'target_found': False,
                'overall_bias_score': 0,
                'overall_stance': 'neutral',
                'confidence': 0
            }
        
        # 종합 편향 점수 계산
        overall_bias_score = np.mean(weighted_scores)
        
        # 종합 입장 결정
        if stance_counts['positive'] > stance_counts['negative']:
            overall_stance = 'positive'
        elif stance_counts['negative'] > stance_counts['positive']:
            overall_stance = 'negative'
        else:
            overall_stance = 'neutral'
        
        # 신뢰도 계산 (응답 수 기반)
        confidence = len(weighted_scores) / len(self.question_weights)
        
        return {
            'target_found': True,
            'overall_bias_score': overall_bias_score,
            'overall_stance': overall_stance,
            'confidence': confidence,
            'response_count': len(weighted_scores),
            'stance_distribution': stance_counts,
            'individual_scores': weighted_scores
        }
    
    def analyze_model_bias_comprehensive(self, model_responses: Dict[str, Dict[str, str]]) -> Dict:
        """모델별 종합 편향 분석"""
        comprehensive_results = {}
        
        for model_name, responses in model_responses.items():
            model_results = {}
            
            for entity in ['china', 'north_korea', 'usa', 'russia']:
                # 해당 엔티티와 관련된 질문들만 필터링
                entity_questions = {k: v for k, v in responses.items() 
                                 if entity in k.lower()}
                
                if entity_questions:
                    result = self.analyze_multiple_responses(entity_questions, entity)
                    model_results[entity] = result
                else:
                    model_results[entity] = {
                        'target_found': False,
                        'overall_bias_score': 0,
                        'overall_stance': 'neutral',
                        'confidence': 0
                    }
            
            comprehensive_results[model_name] = model_results
        
        return comprehensive_results
    
    def generate_bias_report(self, comprehensive_results: Dict) -> str:
        """편향 분석 리포트 생성"""
        report = "=== 종합 편향 분석 리포트 ===\n\n"
        
        for model_name, results in comprehensive_results.items():
            report += f"📊 {model_name}\n"
            report += "=" * 30 + "\n"
            
            for entity, result in results.items():
                if result['target_found']:
                    stance_emoji = {
                        'positive': '👍',
                        'negative': '👎',
                        'neutral': '🤝'
                    }.get(result['overall_stance'], '❓')
                    
                    report += f"  {entity}: {stance_emoji} 편향점수={result['overall_bias_score']:.3f}\n"
                    report += f"      입장={result['overall_stance']}, 신뢰도={result['confidence']:.2f}\n"
                    report += f"      응답수={result['response_count']}\n"
                    report += f"      분포={result['stance_distribution']}\n"
                else:
                    report += f"  {entity}: 타겟 미발견\n"
            
            report += "\n"
        
        return report
    
    def get_question_set(self, target_entity: str = None) -> Dict[str, str]:
        """특정 엔티티에 대한 질문 세트 반환"""
        if target_entity:
            return {k: v for k, v in self.standard_questions.items() 
                   if target_entity in k.lower()}
        return self.standard_questions 