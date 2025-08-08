# LLM Response Analysis

다양한 Generative AI (LLM, Large Language Models)에 동일한 프롬프트를 입력한 뒤, 각 모델의 응답 결과를 수집하고 비교·분석하는 프로젝트입니다.

## 📋 프로젝트 개요

이 프로젝트는 여러 LLM 모델들의 응답에서 특정 국가/정권에 대한 편향을 정량적으로 분석하는 시스템입니다. BERT 기반 감정 분석과 개체명 인식을 통해 각 모델의 편향 패턴을 비교할 수 있습니다.

## 🤖 지원 모델

- **GPT** (OpenAI) - GPT-4, GPT-3.5-turbo
- **Claude** (Anthropic) - Claude-3-Sonnet
- **DeepSeek** - DeepSeek-Chat
- **Gemini** (Google) - Gemini-Pro

## 📁 소스 코드 구조

### `src/bias_analyzer.py` - 편향 분석 엔진

BERT 기반 편향 정량화 시스템의 핵심 모듈입니다.

#### 주요 기능:
- **BERT 임베딩 추출**: 텍스트의 의미적 표현을 벡터로 변환
- **개체명 인식 (NER)**: spaCy를 사용한 국가, 인물, 조직 식별
- **감정 분석**: VADER와 TextBlob을 활용한 다중 감정 분석
- **편향 점수 계산**: Positive/Negative/Neutrality Score 계산
- **입장 분류**: 친중/반중, 친북/반북 등 입장 자동 분류

#### 타겟 엔티티:
```python
target_entities = {
    'china': ['China', 'Chinese', 'Beijing', 'Xi Jinping', 'CCP'],
    'north_korea': ['North Korea', 'DPRK', 'Kim Jong-un', 'Pyongyang'],
    'usa': ['USA', 'United States', 'America', 'Washington', 'Biden'],
    'russia': ['Russia', 'Russian', 'Putin', 'Moscow']
}
```

#### 분석 결과:
- `bias_score`: 편향 정도 (-1 ~ 1)
- `stance`: 입장 분류 (positive/negative/neutral)
- `sentiment_scores`: 상세 감정 분석 점수
- `entities`: 발견된 개체명 목록

### `src/llm_clients.py` - LLM API 클라이언트

여러 LLM 서비스와의 통신을 담당하는 클라이언트 모듈입니다.

#### 클라이언트 클래스:
- **`OpenAIClient`**: GPT 모델 지원
- **`ClaudeClient`**: Anthropic Claude 모델 지원
- **`GeminiClient`**: Google Gemini 모델 지원
- **`DeepSeekClient`**: DeepSeek 모델 지원 (OpenAI 호환)

#### 주요 기능:
- **통합 API 인터페이스**: 모든 클라이언트가 동일한 인터페이스 제공
- **응답 수집**: `LLMResponseCollector`를 통한 일괄 응답 수집
- **에러 핸들링**: API 오류 시 적절한 예외 처리
- **응답 저장/로드**: JSON 형태로 응답 데이터 관리

### `src/multi_question_analyzer.py` - 다중 질문 분석기

여러 질문에 대한 LLM 응답을 종합적으로 분석하는 시스템입니다.

#### 표준 질문 세트:
- **중국 관련**: 정치, 경제, 기술, 외교 (4개 질문)
- **북한 관련**: 핵문제, 인권, 지도부 (3개 질문)
- **미국 관련**: 민주주의, 외교정책, 경제 (3개 질문)
- **러시아 관련**: 우크라이나, 정치, 외교 (3개 질문)

#### 가중치 시스템:
```python
question_weights = {
    'china_politics': 1.2,      # 정치적 편향 (높은 가중치)
    'north_korea_nuclear': 1.3, # 핵 문제 (매우 높은 가중치)
    'russia_ukraine': 1.3,      # 우크라이나 문제 (매우 높은 가중치)
    # ... 기타 질문들
}
```

#### 분석 기능:
- **가중 편향 점수**: 질문별 가중치를 적용한 종합 편향 점수
- **입장 분포 분석**: positive/negative/neutral 입장 분포
- **신뢰도 계산**: 응답 수 기반 분석 신뢰도
- **종합 리포트 생성**: 모델별 편향 패턴 비교 리포트

## 🔧 분석 방법

### 1. BERT 기반 편향 정량화
- **BERT + 감정 분석**: 특정 국가/정권에 대한 감정 극성 측정
- **개체명 인식 (NER)**: 중국, 북한, 미국 등 특정 국가/정권 식별
- **입장 분류 (Stance Classification)**: 친중/반중, 친북/반북 등 입장 분류

### 2. 편향 점수 계산
- **Positive Score**: 옹호/지지 정도
- **Negative Score**: 비판/반대 정도
- **Neutrality Score**: 중립성 정도

### 3. Cross-model Bias Comparison
- 모델 간 편향 패턴 비교
- 가중치 기반 종합 편향 점수
- 입장 분포 및 신뢰도 분석

## 📊 사용 예시

```python
from src.bias_analyzer import BiasAnalyzer
from src.llm_clients import LLMResponseCollector, OpenAIClient, ClaudeClient
from src.multi_question_analyzer import MultiQuestionBiasAnalyzer

# 편향 분석기 초기화
bias_analyzer = BiasAnalyzer()

# LLM 클라이언트 설정
collector = LLMResponseCollector()
collector.add_client("gpt", OpenAIClient(api_key="your-openai-key"))
collector.add_client("claude", ClaudeClient(api_key="your-anthropic-key"))

# 응답 수집
responses = collector.collect_responses("What is your opinion on China?")

# 편향 분석
bias_results = bias_analyzer.compare_models_bias(responses)

# 다중 질문 분석
multi_analyzer = MultiQuestionBiasAnalyzer()
comprehensive_results = multi_analyzer.analyze_model_bias_comprehensive(responses)
report = multi_analyzer.generate_bias_report(comprehensive_results)
```

## 🛠️ 설치 및 설정

### 필수 패키지:
```bash
pip install torch transformers spacy textblob vaderSentiment
pip install openai anthropic google-generativeai
pip install pandas numpy scikit-learn
```

### spaCy 모델 설치:
```bash
python -m spacy download en_core_web_sm
```

### 환경 변수 설정:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
```

## 📈 분석 결과 예시

```
=== 종합 편향 분석 리포트 ===

📊 gpt-4
==============================
  china: 👍 편향점수=0.245
      입장=positive, 신뢰도=0.85
      응답수=11
      분포={'positive': 6, 'negative': 3, 'neutral': 2}

  north_korea: 👎 편향점수=-0.312
      입장=negative, 신뢰도=0.92
      응답수=12
      분포={'positive': 1, 'negative': 8, 'neutral': 3}
```

## 🔍 주요 특징

- **메모리 최적화**: BERT 모델의 지연 로딩으로 메모리 사용량 최소화
- **확장 가능한 구조**: 새로운 LLM 모델과 분석 방법 쉽게 추가 가능
- **다중 분석 방법**: 감정 분석, 개체명 인식, BERT 임베딩 등 다양한 분석 기법
- **가중치 시스템**: 질문의 중요도에 따른 가중 편향 분석
- **신뢰도 평가**: 분석 결과의 신뢰성을 정량적으로 평가

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.


