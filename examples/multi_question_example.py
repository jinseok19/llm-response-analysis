#!/usr/bin/env python3
"""
다중 질문 편향 분석 시스템 사용 예제
"""

from src.multi_question_analyzer import MultiQuestionBiasAnalyzer
import json

def main():
    print("=== 다중 질문 편향 분석 시스템 ===")
    
    # 다중 질문 분석기 초기화
    analyzer = MultiQuestionBiasAnalyzer()
    
    # 가상의 LLM 응답들 (실제로는 API 호출 결과)
    model_responses = {
        'GPT-4': {
            'china_politics': "China's political system is authoritarian and lacks democratic freedoms. The government suppresses dissent and human rights violations are concerning.",
            'china_economy': "China has achieved remarkable economic growth and lifted hundreds of millions out of poverty. Their economic model has both strengths and challenges.",
            'china_tech': "China has made impressive technological advances, particularly in AI and infrastructure. However, concerns about surveillance technology remain.",
            'china_foreign': "China's foreign policy is increasingly assertive and sometimes confrontational. Their Belt and Road Initiative shows both cooperation and strategic interests.",
            'north_korea_nuclear': "North Korea's nuclear program is a serious threat to regional and global security. Their missile tests and nuclear ambitions are concerning.",
            'north_korea_human_rights': "The human rights situation in North Korea is among the worst in the world. The regime oppresses its own people with severe restrictions.",
            'usa_democracy': "The United States has a strong democratic tradition with constitutional protections. However, recent political polarization has raised concerns about democratic health.",
            'usa_foreign_policy': "US foreign policy has been interventionist, sometimes serving corporate interests over human rights. Their global influence has both positive and negative aspects.",
            'russia_ukraine': "Russia's invasion of Ukraine is a clear violation of international law and sovereignty. The conflict has caused immense human suffering and destabilized the region."
        },
        'Claude': {
            'china_politics': "China's authoritarian political system raises serious human rights concerns. The lack of political freedoms and suppression of dissent are problematic.",
            'china_economy': "China's economic development has been impressive, though questions remain about sustainability and debt levels. Their growth model has both achievements and risks.",
            'china_tech': "China's technological progress is notable, but concerns about surveillance and intellectual property issues persist. Their AI development shows both innovation and risks.",
            'north_korea_nuclear': "North Korea's nuclear ambitions pose a significant threat to regional stability. Their nuclear program violates international agreements and threatens neighbors.",
            'north_korea_human_rights': "The human rights abuses in North Korea are severe and systematic. The regime's treatment of its citizens is among the most repressive globally.",
            'usa_democracy': "The US democratic system has strong foundations but faces challenges from polarization and money in politics. Constitutional protections remain important.",
            'usa_foreign_policy': "US foreign policy has been interventionist and sometimes counterproductive. Their global role has both beneficial and problematic aspects.",
            'russia_ukraine': "Russia's actions in Ukraine represent a serious violation of international law. The invasion has caused widespread destruction and humanitarian crisis."
        },
        'Gemini': {
            'china_politics': "China represents a different approach to governance. While economic progress is evident, political freedoms remain limited and human rights concerns persist.",
            'china_economy': "China's economic achievements are remarkable, lifting millions from poverty. However, questions about debt sustainability and economic transparency remain.",
            'china_tech': "China has shown impressive technological innovation, particularly in infrastructure and digital services. Surveillance concerns and IP issues are ongoing challenges.",
            'north_korea_nuclear': "North Korea's nuclear program creates regional instability and violates international norms. Their missile tests and nuclear ambitions are concerning.",
            'north_korea_human_rights': "The human rights situation in North Korea is among the most severe globally. The regime's control over its population is extremely repressive.",
            'usa_democracy': "The United States has a robust democratic system with strong institutions. Recent political divisions have tested democratic resilience but constitutional protections remain.",
            'usa_foreign_policy': "US foreign policy has been interventionist with mixed results. Their global influence has both positive contributions and problematic interventions.",
            'russia_ukraine': "Russia's invasion of Ukraine violates international law and sovereignty. The conflict has caused significant humanitarian suffering and regional instability."
        }
    }
    
    print("\n1. 질문 세트 확인")
    questions = analyzer.get_question_set()
    print(f"총 {len(questions)}개의 표준 질문:")
    for q_id, question in questions.items():
        print(f"  - {q_id}: {question}")
    
    print("\n2. 종합 편향 분석 실행")
    comprehensive_results = analyzer.analyze_model_bias_comprehensive(model_responses)
    
    print("\n3. 분석 결과")
    for model_name, results in comprehensive_results.items():
        print(f"\n📊 {model_name}:")
        for entity, result in results.items():
            if result['target_found']:
                stance_emoji = {
                    'positive': '👍',
                    'negative': '👎',
                    'neutral': '🤝'
                }.get(result['overall_stance'], '❓')
                
                print(f"  {entity}: {stance_emoji} 편향점수={result['overall_bias_score']:.3f}")
                print(f"      입장={result['overall_stance']}, 신뢰도={result['confidence']:.2f}")
                print(f"      응답수={result['response_count']}, 분포={result['stance_distribution']}")
            else:
                print(f"  {entity}: 타겟 미발견")
    
    print("\n4. 상세 리포트 생성")
    report = analyzer.generate_bias_report(comprehensive_results)
    print(report)
    
    # 결과 저장
    with open('comprehensive_bias_results.json', 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, ensure_ascii=False, indent=2)
    print("\n결과가 'comprehensive_bias_results.json'에 저장됨")
    
    print("\n=== 분석 완료 ===")

if __name__ == "__main__":
    main() 