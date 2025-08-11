#!/usr/bin/env python3
"""
LLM 편향 분석 대시보드
Streamlit을 사용한 웹 인터페이스
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from src.multi_question_analyzer import MultiQuestionBiasAnalyzer
from src.bias_analyzer import BiasAnalyzer

# 페이지 설정
st.set_page_config(
    page_title="LLM 편향 분석 대시보드",
    page_icon="📊",
    layout="wide"
)

def load_results():
    """JSON 결과 파일 로드"""
    try:
        with open('comprehensive_bias_results.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("결과 파일을 찾을 수 없습니다. 먼저 분석을 실행해주세요.")
        return None

def create_bias_score_chart(data):
    """편향 점수 차트 생성"""
    # 데이터 준비
    chart_data = []
    for model_name, results in data.items():
        for entity, result in results.items():
            if result['target_found']:
                chart_data.append({
                    'Model': model_name,
                    'Entity': entity,
                    'Bias Score': result['overall_bias_score'],
                    'Stance': result['overall_stance'],
                    'Confidence': result['confidence']
                })
    
    df = pd.DataFrame(chart_data)
    
    # 히트맵 생성
    fig = px.imshow(
        df.pivot(index='Entity', columns='Model', values='Bias Score'),
        title="LLM별 편향 점수 히트맵",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig.update_layout(height=500)
    return fig

def create_stance_distribution_chart(data):
    """입장 분포 차트 생성"""
    stance_data = []
    for model_name, results in data.items():
        for entity, result in results.items():
            if result['target_found']:
                stance_data.append({
                    'Model': model_name,
                    'Entity': entity,
                    'Stance': result['overall_stance']
                })
    
    df = pd.DataFrame(stance_data)
    
    # 입장별 분포 차트
    fig = px.histogram(
        df, 
        x='Entity', 
        color='Stance',
        title="엔티티별 입장 분포",
        color_discrete_map={
            'positive': '#2E8B57',
            'negative': '#DC143C', 
            'neutral': '#808080'
        }
    )
    fig.update_layout(height=400)
    return fig

def create_model_comparison_chart(data):
    """모델 비교 차트 생성"""
    model_scores = []
    for model_name, results in data.items():
        total_score = 0
        count = 0
        for entity, result in results.items():
            if result['target_found']:
                total_score += abs(result['overall_bias_score'])
                count += 1
        
        if count > 0:
            avg_bias = total_score / count
            model_scores.append({
                'Model': model_name,
                'Average Bias Magnitude': avg_bias,
                'Response Count': count
            })
    
    df = pd.DataFrame(model_scores)
    
    fig = px.bar(
        df,
        x='Model',
        y='Average Bias Magnitude',
        title="모델별 평균 편향 강도",
        color='Response Count',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400)
    return fig

def create_confidence_chart(data):
    """신뢰도 차트 생성"""
    confidence_data = []
    for model_name, results in data.items():
        for entity, result in results.items():
            if result['target_found']:
                confidence_data.append({
                    'Model': model_name,
                    'Entity': entity,
                    'Confidence': result['confidence'],
                    'Response Count': result['response_count']
                })
    
    df = pd.DataFrame(confidence_data)
    
    fig = px.scatter(
        df,
        x='Confidence',
        y='Response Count',
        color='Model',
        size='Confidence',
        title="신뢰도 vs 응답 수",
        hover_data=['Entity']
    )
    fig.update_layout(height=400)
    return fig

def main():
    st.title("🤖 LLM 편향 분석 대시보드")
    st.markdown("---")
    
    # 사이드바
    st.sidebar.title("📋 메뉴")
    page = st.sidebar.selectbox(
        "페이지 선택",
        ["📊 대시보드", "🔍 상세 분석", "⚙️ 설정"]
    )
    
    if page == "📊 대시보드":
        show_dashboard()
    elif page == "🔍 상세 분석":
        show_detailed_analysis()
    elif page == "⚙️ 설정":
        show_settings()

def show_dashboard():
    """대시보드 페이지"""
    st.header("📊 편향 분석 대시보드")
    
    # 결과 로드
    data = load_results()
    if data is None:
        return
    
    # 상단 통계
    col1, col2, col3, col4 = st.columns(4)
    
    total_models = len(data)
    total_entities = len(['china', 'north_korea', 'usa', 'russia'])
    total_analyses = sum(len(results) for results in data.values())
    
    with col1:
        st.metric("분석된 모델", total_models)
    with col2:
        st.metric("분석된 엔티티", total_entities)
    with col3:
        st.metric("총 분석 수", total_analyses)
    with col4:
        avg_confidence = np.mean([
            result['confidence'] 
            for results in data.values() 
            for result in results.values() 
            if result['target_found']
        ])
        st.metric("평균 신뢰도", f"{avg_confidence:.2f}")
    
    # 차트들
    st.subheader("📈 편향 점수 히트맵")
    bias_chart = create_bias_score_chart(data)
    st.plotly_chart(bias_chart, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 입장 분포")
        stance_chart = create_stance_distribution_chart(data)
        st.plotly_chart(stance_chart, use_container_width=True)
    
    with col2:
        st.subheader("⚖️ 모델 비교")
        model_chart = create_model_comparison_chart(data)
        st.plotly_chart(model_chart, use_container_width=True)
    
    st.subheader("📊 신뢰도 분석")
    confidence_chart = create_confidence_chart(data)
    st.plotly_chart(confidence_chart, use_container_width=True)

def show_detailed_analysis():
    """상세 분석 페이지"""
    st.header("🔍 상세 분석")
    
    data = load_results()
    if data is None:
        return
    
    # 모델 선택
    selected_model = st.selectbox("모델 선택", list(data.keys()))
    
    if selected_model:
        model_results = data[selected_model]
        
        st.subheader(f"📊 {selected_model} 상세 분석")
        
        # 엔티티별 상세 정보
        for entity, result in model_results.items():
            with st.expander(f"🌍 {entity.upper()}"):
                if result['target_found']:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("편향 점수", f"{result['overall_bias_score']:.3f}")
                    
                    with col2:
                        stance_emoji = {
                            'positive': '👍',
                            'negative': '👎',
                            'neutral': '🤝'
                        }.get(result['overall_stance'], '❓')
                        st.metric("입장", f"{stance_emoji} {result['overall_stance']}")
                    
                    with col3:
                        st.metric("신뢰도", f"{result['confidence']:.2f}")
                    
                    # 분포 차트
                    if 'stance_distribution' in result:
                        dist_df = pd.DataFrame([
                            {'Stance': k, 'Count': v} 
                            for k, v in result['stance_distribution'].items()
                        ])
                        
                        fig = px.pie(
                            dist_df, 
                            values='Count', 
                            names='Stance',
                            title=f"{entity} 입장 분포"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("타겟 미발견")

def show_settings():
    """설정 페이지"""
    st.header("⚙️ 설정")
    
    st.subheader("📝 새로운 분석 실행")
    
    if st.button("🔄 다중 질문 분석 실행"):
        with st.spinner("분석 중..."):
            try:
                # 분석 실행
                from multi_question_example import main as run_analysis
                run_analysis()
                st.success("분석 완료!")
                st.rerun()
            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")
    
    st.subheader("📁 파일 관리")
    
    if st.button("🗑️ 결과 파일 삭제"):
        import os
        if os.path.exists('comprehensive_bias_results.json'):
            os.remove('comprehensive_bias_results.json')
            st.success("파일 삭제 완료!")
        else:
            st.warning("삭제할 파일이 없습니다.")

if __name__ == "__main__":
    main() 