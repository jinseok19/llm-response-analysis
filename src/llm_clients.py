import openai
import anthropic
import google.generativeai as genai
import requests
import json
import os
from typing import Dict, List, Optional

class LLMClient:
    """LLM API 클라이언트 기본 클래스"""
    
    def __init__(self):
        self.api_key = None
        self.client = None
    
    def generate_response(self, prompt: str) -> str:
        """프롬프트에 대한 응답 생성"""
        raise NotImplementedError
    
    def set_api_key(self, api_key: str):
        """API 키 설정"""
        self.api_key = api_key

class OpenAIClient(LLMClient):
    """OpenAI GPT 클라이언트"""
    
    def __init__(self, api_key: str = None):
        super().__init__()
        if api_key:
            self.set_api_key(api_key)
    
    def set_api_key(self, api_key: str):
        super().set_api_key(api_key)
        openai.api_key = api_key
        self.client = openai
    
    def generate_response(self, prompt: str, model: str = "gpt-4") -> str:
        """GPT 응답 생성"""
        try:
            response = self.client.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API 오류: {e}")
            return ""

class ClaudeClient(LLMClient):
    """Anthropic Claude 클라이언트"""
    
    def __init__(self, api_key: str = None):
        super().__init__()
        if api_key:
            self.set_api_key(api_key)
    
    def set_api_key(self, api_key: str):
        super().set_api_key(api_key)
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate_response(self, prompt: str, model: str = "claude-3-sonnet-20240229") -> str:
        """Claude 응답 생성"""
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Claude API 오류: {e}")
            return ""

class GeminiClient(LLMClient):
    """Google Gemini 클라이언트"""
    
    def __init__(self, api_key: str = None):
        super().__init__()
        if api_key:
            self.set_api_key(api_key)
    
    def set_api_key(self, api_key: str):
        super().set_api_key(api_key)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel('gemini-pro')
    
    def generate_response(self, prompt: str) -> str:
        """Gemini 응답 생성"""
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API 오류: {e}")
            return ""

class DeepSeekClient(LLMClient):
    """DeepSeek 클라이언트 (OpenAI 호환)"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com"):
        super().__init__()
        self.base_url = base_url
        if api_key:
            self.set_api_key(api_key)
    
    def set_api_key(self, api_key: str):
        super().set_api_key(api_key)
        openai.api_key = api_key
        openai.api_base = self.base_url
        self.client = openai
    
    def generate_response(self, prompt: str, model: str = "deepseek-chat") -> str:
        """DeepSeek 응답 생성"""
        try:
            response = self.client.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API 오류: {e}")
            return ""

class LLMResponseCollector:
    """여러 LLM에서 응답을 수집하는 클래스"""
    
    def __init__(self):
        self.clients = {}
        self.responses = {}
    
    def add_client(self, name: str, client: LLMClient):
        """클라이언트 추가"""
        self.clients[name] = client
    
    def collect_responses(self, prompt: str) -> Dict[str, str]:
        """모든 클라이언트에서 응답 수집"""
        self.responses = {}
        
        for name, client in self.clients.items():
            print(f"{name}에서 응답 수집 중...")
            response = client.generate_response(prompt)
            self.responses[name] = response
            print(f"{name} 응답 완료")
        
        return self.responses
    
    def save_responses(self, filename: str):
        """응답을 파일로 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.responses, f, ensure_ascii=False, indent=2)
    
    def load_responses(self, filename: str):
        """파일에서 응답 로드"""
        with open(filename, 'r', encoding='utf-8') as f:
            self.responses = json.load(f) 