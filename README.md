# llm-response-analysis
다양한 Generative AI (LLM, Large Language Models)에 동일한 프롬프트(prompt)를 입력한 뒤, 각 모델의 응답 결과를 수집하고 비교·분석하는 프로젝트입니다.

## 모델
1. gpt
2. claude
3. deepseek
4. gemini



## 분석 방법

#### BERT 기반 편향 정량화 시스템
- **BERT + 감정 분석**: 특정 국가/정권에 대한 감정 극성 측정
- **개체명 인식 (NER)**: 중국, 북한, 미국 등 특정 국가/정권 식별
- **입장 분류 (Stance Classification)**: 친중/반중, 친북/반북 등 입장 분류
- **편향 점수 계산**: 
  - Positive Score: 옹호/지지 정도
  - Negative Score: 비판/반대 정도
  - Neutrality Score: 중립성 정도
- **Cross-model Bias Comparison**: 모델 간 편향 패턴 비교
