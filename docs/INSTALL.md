# LLM Response Analysis - 설치 가이드

## 1. 환경 설정

### Python 가상환경 생성
```bash
python -m venv llm-analysis-env
source llm-analysis-env/bin/activate  # Linux/Mac
# 또는
llm-analysis-env\Scripts\activate  # Windows
```

### 패키지 설치
```bash
pip install -r requirements.txt
```

### spaCy 모델 설치
```bash
python -m spacy download en_core_web_sm
```

## 2. API 키 설정

### 환경변수 설정
```bash
# Windows
set OPENAI_API_KEY=your_openai_api_key_here
set ANTHROPIC_API_KEY=your_anthropic_api_key_here
set GOOGLE_API_KEY=your_google_api_key_here
set DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_openai_api_key_here
export ANTHROPIC_API_KEY=your_anthropic_api_key_here
export GOOGLE_API_KEY=your_google_api_key_here
export DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### .env 파일 사용 (선택사항)
```bash
# python-dotenv 설치
pip install python-dotenv

# .env 파일 생성
cp env_example.txt .env
# .env 파일에 실제 API 키 입력
```

## 3. 실행 방법

### 기본 사용 예제 실행
```bash
python example_usage.py
```

### API 사용 예제 실행
```bash
python api_example.py
```

### Jupyter Notebook 사용
```bash
jupyter notebook
```

## 4. 프로젝트 구조
```
llm-response-analysis/
├── src/
│   ├── __init__.py
│   ├── bias_analyzer.py      # 핵심 편향 분석 클래스
│   └── llm_clients.py        # LLM API 클라이언트들
├── example_usage.py          # 기본 사용 예제
├── api_example.py            # API 사용 예제
├── requirements.txt          # 의존성 패키지
├── env_example.txt          # API 키 설정 예제
├── README.md                # 프로젝트 설명
└── INSTALL.md               # 설치 가이드
```

## 5. 주요 기능

### 편향 분석
- 특정 국가/정권에 대한 편향 점수 계산
- 개체명 인식 (NER)을 통한 타겟 식별
- 감정 분석을 통한 편향 정량화
- 여러 모델 간 편향 패턴 비교

### 지원하는 LLM
- **OpenAI GPT-4/GPT-3.5**: OpenAI API 사용
- **Anthropic Claude**: Claude API 사용
- **Google Gemini**: Google AI API 사용
- **DeepSeek**: DeepSeek API 사용

### 지원하는 타겟
- 중국 (China, Chinese, Beijing, Xi Jinping, CCP)
- 북한 (North Korea, DPRK, Kim Jong-un, Pyongyang)
- 미국 (USA, United States, America, Washington, Biden)
- 러시아 (Russia, Russian, Putin, Moscow)

## 6. 문제 해결

### spaCy 모델 설치 오류
```bash
python -m spacy download en_core_web_sm
```

### API 키 오류
- 환경변수가 올바르게 설정되었는지 확인
- API 키가 유효한지 확인
- API 사용량 한도를 확인

### CUDA 오류 (GPU 사용 시)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 메모리 부족 오류
- `max_length` 파라미터 조정
- 배치 크기 줄이기
- CPU 모드 사용 