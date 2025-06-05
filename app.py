import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def safe_load_data(uploaded_file):
    """Безопасная загрузка данных из Excel"""
    try:
        data = pd.read_excel(uploaded_file)
        if data.empty:
            st.error("Файл не содержит данных")
            return None
        return data
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {str(e)}")
        return None

def analyze_satisfaction(data):
    """Анализ удовлетворенности клиентов"""
    st.header("Анализ удовлетворенности клиентов")
    
    # Распределение satisfaction
    fig, ax = plt.subplots(figsize=(8, 5))
    data['satisfaction'].value_counts().plot(kind='bar', color=['#ff9999','#66b3ff'])
    plt.title("Распределение удовлетворенности клиентов")
    plt.xlabel("Удовлетворенность")
    plt.ylabel("Количество")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()
    
    # Влияние факторов на удовлетворенность
    service_columns = [
        'Inflight wifi service',
        'Departure/Arrival time convenient',
        'Ease of Online booking',
        'Gate location',
        'Food and drink',
        'Online boarding',
        'Seat comfort',
        'Inflight entertainment',
        'On-board service',
        'Leg room service',
        'Baggage handling',
        'Checkin service',
        'Inflight service',
        'Cleanliness'
    ]
    
    st.subheader("Средние оценки по сервисам")
    mean_ratings = data.groupby('satisfaction')[service_columns].mean().T
    fig, ax = plt.subplots(figsize=(12, 8))
    mean_ratings.plot(kind='bar', ax=ax, color=['#ff9999','#66b3ff'])
    plt.title("Средние оценки сервисов")
    plt.xlabel("Категория сервиса")
    plt.ylabel("Средняя оценка")
    plt.xticks(rotation=45)
    plt.legend(title="Удовлетворенность")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def analyze_service_ratings(data):
    """Анализ оценок сервисов"""
    st.header("Анализ оценок сервисов")
    
    service_columns = [
        'Inflight wifi service',
        'Departure/Arrival time convenient',
        'Ease of Online booking',
        'Gate location',
        'Food and drink',
        'Online boarding',
        'Seat comfort',
        'Inflight entertainment',
        'On-board service',
        'Leg room service',
        'Baggage handling',
        'Checkin service',
        'Inflight service',
        'Cleanliness'
    ]
    
    selected_service = st.selectbox("Выберите сервис для анализа", service_columns)
    
    # Распределение оценок
    fig, ax = plt.subplots(figsize=(10, 6))
    data[selected_service].value_counts().sort_index().plot(kind='bar', color='#1f77b4')
    plt.title(f"Распределение оценок для {selected_service}")
    plt.xlabel("Оценка")
    plt.ylabel("Количество")
    st.pyplot(fig)
    plt.close()
    
    # Зависимость от типа клиента (без seaborn)
    fig, ax = plt.subplots(figsize=(10, 6))
    data.boxplot(column=selected_service, by='Customer Type', ax=ax)
    plt.title(f"Распределение оценок {selected_service} по типу клиента")
    plt.suptitle('')
    st.pyplot(fig)
    plt.close()
    
    # Зависимость от класса обслуживания (без seaborn)
    fig, ax = plt.subplots(figsize=(10, 6))
    data.boxplot(column=selected_service, by='Class', ax=ax)
    plt.title(f"Распределение оценок {selected_service} по классу обслуживания")
    plt.suptitle('')
    st.pyplot(fig)
    plt.close()

def main():
    st.title("✈️ Анализатор удовлетворенности авиапассажиров")
    st.markdown("""
    <style>
    .small-font { font-size:12px !important; color:gray; }
    </style>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Загрузите ваш Excel файл", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        data = safe_load_data(uploaded_file)
        if data is not None:
            st.success(f"Успешно загружено {len(data)} записей")
            
            required_columns = ['satisfaction', 'Inflight wifi service', 'Customer Type', 'Class']
            if not all(col in data.columns for col in required_columns):
                st.error("Файл не содержит всех необходимых столбцов")
                st.write("Найдены столбцы:", list(data.columns))
                return
            
            with st.expander("🔍 Просмотр данных"):
                st.write("Первые 5 строк:")
                st.write(data.head())
                st.markdown(f'<p class="small-font">Все столбцы: {list(data.columns)}</p>', 
                          unsafe_allow_html=True)
            
            analyze_satisfaction(data)
            analyze_service_ratings(data)
            
            st.header("Дополнительные метрики")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Средняя задержка вылета (мин)", round(data['Departure Delay in Minutes'].mean(), 1))
                st.metric("Средняя задержка прилета (мин)", round(data['Arrival Delay in Minutes'].mean(), 1))
            
            with col2:
                loyal_percent = round(data['Customer Type'].value_counts(normalize=True)['Loyal Customer']*100, 1)
                st.metric("Доля лояльных клиентов", f"{loyal_percent}%")
                st.metric("Средний возраст пассажиров", round(data['Age'].mean(), 1))

if __name__ == "__main__":
    main()