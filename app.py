import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import numpy as np

# Настройка страницы
st.set_page_config(layout="wide", page_title="✈️ Продвинутый анализ удовлетворённости авиапассажиров", page_icon="✈️")

def safe_load_data(uploaded_file):
    """Улучшенная загрузка данных с поддержкой CSV и Excel"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"Ошибка загрузки: {str(e)}")
        return None

def show_data_overview(data):
    """Расширенный обзор данных"""
    with st.expander("📊 Полный обзор данных", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Основные статистики**")
            st.dataframe(data.describe(include='all').T.style.background_gradient(cmap='Blues'))
            
        with col2:
            st.markdown("**Пропущенные значения**")
            missing = data.isnull().sum().to_frame('Пропуски')
            st.dataframe(missing[missing['Пропуски'] > 0].style.background_gradient(cmap='Reds'))

def analyze_satisfaction(data):
    """Углублённый анализ удовлетворённости"""
    st.header("🔍 Глубокий анализ удовлетворённости", divider='rainbow')
    
    # Распределение по полу и классу
    fig = px.sunburst(data, path=['Gender', 'Class', 'satisfaction'], 
                     color='satisfaction', color_discrete_map={
                         'satisfied': '#2ca02c',
                         'neutral or dissatisfied': '#d62728'
                     })
    st.plotly_chart(fig, use_container_width=True)
    
    # Динамика по возрасту
    age_bins = [0, 18, 30, 45, 60, 100]
    data['Age Group'] = pd.cut(data['Age'], bins=age_bins, 
                              labels=['<18', '18-30', '30-45', '45-60', '60+'])
    
    fig = px.histogram(data, x='Age Group', color='satisfaction', barmode='group',
                      title="Распределение удовлетворённости по возрастным группам")
    st.plotly_chart(fig, use_container_width=True)

def service_analysis(data):
    """Расширенный анализ сервисов"""
    st.header("🛎️ Анализ качества сервисов", divider='rainbow')
    
    services = [
        'Inflight wifi service', 'Food and drink', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Cleanliness'
    ]
    
    # Тепловая карта корреляций
    st.subheader("Корреляция между оценками сервисов")
    corr_matrix = data[services].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                   color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # Сравнение сервисов
    st.subheader("Сравнение средних оценок сервисов")
    service_means = data[services].mean().sort_values()
    fig = px.bar(service_means, orientation='h', 
                labels={'value': 'Средняя оценка', 'index': 'Сервис'},
                color=service_means.values, color_continuous_scale='Teal')
    st.plotly_chart(fig, use_container_width=True)

def delay_analysis(data):
    """Анализ задержек рейсов"""
    st.header("⏱️ Анализ задержек рейсов", divider='rainbow')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(data, y='Departure Delay in Minutes', 
                    points="all", title="Задержка вылета")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(data, x='Departure Delay in Minutes', 
                        y='Arrival Delay in Minutes', color='satisfaction',
                        title="Связь задержек вылета и прилёта")
        st.plotly_chart(fig, use_container_width=True)
    
    # Влияние задержек на удовлетворённость
    data['Delay Impact'] = np.where(
        data['Departure Delay in Minutes'] > 60, 'Большая (>60 мин)',
        np.where(data['Departure Delay in Minutes'] > 30, 'Средняя (30-60 мин)', 'Маленькая (<30 мин)'))
    
    fig = px.histogram(data, x='Delay Impact', color='satisfaction', 
                      barmode='group', title="Влияние задержки на удовлетворённость")
    st.plotly_chart(fig, use_container_width=True)

def customer_segmentation(data):
    """Сегментация клиентов"""
    st.header("👥 Сегментация пассажиров", divider='rainbow')
    
    # RFM-анализ (упрощённый)
    st.subheader("Анализ лояльности клиентов")
    rfm = data.groupby('id').agg({
        'satisfaction': lambda x: (x == 'satisfied').mean(),
        'Flight Distance': 'sum',
        'Age': 'last'
    }).rename(columns={
        'satisfaction': 'LoyaltyScore',
        'Flight Distance': 'TotalDistance',
        'Age': 'Age'
    })
    
    fig = px.scatter(rfm, x='TotalDistance', y='LoyaltyScore', 
                    color='Age', size='TotalDistance',
                    hover_data=['Age'], 
                    title="RFM-анализ (Distance vs Loyalty)")
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("✈️ Продвинутый анализ удовлетворённости авиапассажиров")
    
    # Загрузка файла
    uploaded_file = st.file_uploader("Загрузите ваш файл с данными (CSV или Excel)", 
                                   type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        data = safe_load_data(uploaded_file)
        if data is not None:
            st.success(f"✅ Успешно загружено {len(data):,} записей")
            
            # Основные метрики
            st.subheader("📊 Ключевые метрики")
            cols = st.columns(4)
            with cols[0]:
                st.metric("Довольных клиентов", 
                         f"{data['satisfaction'].value_counts(normalize=True).get('satisfied', 0)*100:.1f}%")
            with cols[1]:
                st.metric("Средний возраст", f"{data['Age'].mean():.1f} лет")
            with cols[2]:
                st.metric("Средняя задержка", f"{data['Departure Delay in Minutes'].mean():.1f} мин")
            with cols[3]:
                st.metric("Лояльных клиентов", 
                         f"{data['Customer Type'].value_counts(normalize=True).get('Loyal Customer', 0)*100:.1f}%")
            
            # Основные разделы анализа
            show_data_overview(data)
            analyze_satisfaction(data)
            service_analysis(data)
            delay_analysis(data)
            customer_segmentation(data)
            
            # Генерация отчёта
            st.download_button(
                label="📩 Скачать отчёт (CSV)",
                data=data.to_csv(index=False).encode('utf-8'),
                file_name='airline_passenger_analysis.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()