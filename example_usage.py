#!/usr/bin/env python3
"""
BERT 기반 편향 분석 시스템 사용 예제
"""

from src.bias_analyzer import BiasAnalyzer
import pandas as pd

def main():
    # 편향 분석기 초기화
    print("편향 분석기 초기화 중...")
    analyzer = BiasAnalyzer()
    
    # 테스트 텍스트들 (다양한 편향을 가진 예시)
    test_texts = {
        'pro_china': "China has made remarkable progress in economic development and poverty reduction. The Chinese government's policies have greatly improved the lives of millions of people.",
        'anti_china': "China's human rights violations and authoritarian regime are concerning. The Chinese government suppresses freedom of speech and political dissent.",
        'neutral_china': "China is a large country in Asia with a complex history and diverse population. The country has experienced significant economic changes in recent decades.",
        'pro_north_korea': "North Korea has the right to develop its own path and defend its sovereignty. The DPRK government works for the welfare of its people.",
        'anti_north_korea': "North Korea's nuclear program and human rights abuses are serious concerns. The Kim regime oppresses its own people and threatens regional stability.",
        'pro_usa': "The United States is a beacon of democracy and freedom. American values of liberty and opportunity have inspired people around the world.",
        'anti_usa': "The USA has a history of imperialism and intervention in other countries' affairs. American foreign policy often serves corporate interests over human rights."
    }
    
    print("\n=== 개별 텍스트 편향 분석 ===")
    
    # 각 텍스트에 대한 편향 분석
    for text_name, text in test_texts.items():
        print(f"\n--- {text_name} ---")
        print(f"텍스트: {text[:100]}...")
        
        # 여러 개체에 대한 편향 분석
        bias_results = analyzer.analyze_multiple_entities(text)
        
        for entity, result in bias_results.items():
            if result['target_found']:
                print(f"  {entity}: 편향점수={result['bias_score']:.3f}, 입장={result['stance']}")
            else:
                print(f"  {entity}: 타겟 미발견")
    
    print("\n=== 모델 간 편향 비교 ===")
    
    # 가상의 LLM 응답들 (실제로는 API 호출 결과)
    model_responses = {
        'GPT-4': "China has shown impressive economic growth, though there are concerns about human rights. The country's development model has both strengths and challenges.",
        'Claude': "China's economic achievements are notable, but the government's authoritarian approach raises serious human rights concerns that cannot be ignored.",
        'Gemini': "China represents a different approach to governance and development. While economic progress is evident, questions remain about political freedoms and human rights."
    }
    
    # 모델 간 편향 비교
    comparison_results = analyzer.compare_models_bias(model_responses)
    
    for model_name, results in comparison_results.items():
        print(f"\n{model_name}:")
        for entity, result in results.items():
            if result['target_found']:
                print(f"  {entity}: 편향점수={result['bias_score']:.3f}, 입장={result['stance']}")
    
    print("\n=== 편향 분석 완료 ===")

if __name__ == "__main__":
    main() 