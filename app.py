import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.express as px

# Настройка страницы
st.set_page_config(
    page_title="Анализ удовлетворенности пассажиров",
    page_icon="✈️",
    layout="wide"
)

# Функция загрузки данных с проверкой обязательных столбцов
@st.cache_data
def load_data_from_csv(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        
        # Проверка обязательных столбцов
        required_columns = {'satisfaction_score', 'flight_id'}
        if not required_columns.issubset(data.columns):
            missing = required_columns - set(data.columns)
            st.error(f"В данных отсутствуют обязательные столбцы: {', '.join(missing)}")
            return None
            
        return data
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {e}")
        return None

# Функция очистки данных с дополнительными проверками
@st.cache_data
def clean_data(data):
    if data is None or data.empty:
        st.error("Нет данных для очистки")
        return None

    # Создаем копию данных для очистки
    data_clean = data.copy()
    
    # Удаление пропущенных значений
    data_clean = data_clean.dropna()

    # Проверка наличия обязательных столбцов после очистки
    if 'satisfaction_score' not in data_clean.columns:
        st.error("После очистки в данных отсутствует столбец 'satisfaction_score'")
        return None
        
    # Фильтрация аномальных значений
    data_clean = data_clean[
        (data_clean['satisfaction_score'] >= 1) &
        (data_clean['satisfaction_score'] <= 5)
    ]
    
    return data_clean

# Улучшенная функция построения распределения
def show_distribution(data, title, color='skyblue'):
    try:
        if data is None or data.empty or 'satisfaction_score' not in data.columns:
            st.error("Нет данных или отсутствует столбец 'satisfaction_score'")
            return
            
        fig, ax = plt.subplots(figsize=(8, 5))
        data['satisfaction_score'].hist(bins=20, color=color, ax=ax)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Уровень удовлетворенности', fontsize=12)
        ax.set_ylabel('Количество пассажиров', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Ошибка построения графика: {str(e)}")

# Основной интерфейс
def main():
    st.title("✈️ Анализ удовлетворенности пассажиров")
    
    # Загрузка данных
    uploaded_file = st.file_uploader(
        "Загрузите CSV-файл с данными",
        type=["csv"],
        help="Файл должен содержать столбцы: satisfaction_score и flight_id"
    )

    if uploaded_file is not None:
        data = load_data_from_csv(uploaded_file)
        if data is not None:
            st.session_state['raw_data'] = data
            st.success(f"Данные успешно загружены! Всего записей: {len(data):,}")

    if 'raw_data' in st.session_state:
        data = st.session_state['raw_data']
        
        # Выбор типа анализа
        analysis_option = st.sidebar.radio(
            "Выберите тип анализа:",
            ["Обзор данных", "Очистка данных", "Регрессионный анализ", "Кластеризация", "Отчет по рейсам"]
        )

        if analysis_option == "Обзор данных":
            st.header("🔍 Обзор данных")
            
            # Проверка данных перед отображением
            if 'satisfaction_score' not in data.columns:
                st.error("В данных отсутствует столбец 'satisfaction_score'")
            else:
                show_distribution(data, 'Распределение satisfaction_score (исходные данные)')
                
                with st.expander("Подробная статистика"):
                    if not data.empty:
                        st.dataframe(data.describe().T.style.background_gradient(cmap='Blues'))
                    else:
                        st.warning("Нет данных для отображения статистики")

        elif analysis_option == "Очистка данных":
            st.header("🧹 Очистка данных")
            
            if 'clean_data' not in st.session_state:
                st.session_state['clean_data'] = clean_data(data)

            clean_data_df = st.session_state['clean_data']
            
            if clean_data_df is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Исходное количество записей", len(data))
                    st.metric("Записей после очистки", len(clean_data_df))
                    st.metric("Удалено записей", len(data) - len(clean_data_df))

                with col2:
                    st.write("**Типы удаленных данных:**")
                    st.write("- Пропущенные значения (NaN)")
                    st.write("- Аномальные значения satisfaction_score (вне диапазона 1-5)")

                show_distribution(clean_data_df, 'Распределение после очистки', 'orange')

if __name__ == "__main__":
    main()