import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.express as px
import io

# Настройка страницы
st.set_page_config(
    page_title="Анализ удовлетворенности пассажиров",
    page_icon="✈️",
    layout="wide"
)

# Кэшируемые функции
@st.cache_data
def load_data_from_csv(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {e}")
        return None

@st.cache_data
def clean_data(data):
    if data is None or data.empty:
        return None

    # Удаление пропущенных значений
    data_clean = data.dropna()

    # Проверка обязательных столбцов
    required_columns = {'satisfaction_score', 'flight_id'}
    if not required_columns.issubset(data_clean.columns):
        missing = required_columns - set(data_clean.columns)
        st.error(f"Отсутствуют обязательные столбцы: {', '.join(missing)}")
        return None
        
    # Фильтрация аномальных значений
    data_clean = data_clean[
        (data_clean['satisfaction_score'] >= 1) &
        (data_clean['satisfaction_score'] <= 5)
    ].copy()
    
    return data_clean

# Функции визуализации
def show_distribution(data, title, color='skyblue'):
    try:
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

def plot_regression(X, y, feature_name):
    try:
        model = LinearRegression()
        model.fit(X, y)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X, y, color='blue', alpha=0.5, label='Данные')
        ax.plot(X, model.predict(X), color='red', linewidth=2, label='Линия регрессии')
        ax.set_title(f'Зависимость удовлетворенности от {feature_name}', fontsize=14)
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel('Уровень удовлетворенности', fontsize=12)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        return model.coef_
    except Exception as e:
        st.error(f"Ошибка регрессионного анализа: {str(e)}")
        return None

def plot_clusters(data, n_clusters=3):
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data)

        fig = px.scatter(
            data, 
            x=data.columns[0], 
            y=data.columns[1],
            color=clusters,
            title=f'Кластеризация (k={n_clusters})',
            labels={'color': 'Кластер'},
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return clusters
    except Exception as e:
        st.error(f"Ошибка кластеризации: {str(e)}")
        return None

def generate_flight_report(data):
    try:
        report = data.groupby('flight_id')['satisfaction_score'].agg(['mean', 'count', 'std'])
        report.columns = ['Средняя удовлетворенность', 'Количество пассажиров', 'Стандартное отклонение']
        report = report.sort_values('Средняя удовлетворенность', ascending=False)

        fig = px.bar(
            report,
            x='Средняя удовлетворенность',
            y=report.index,
            orientation='h',
            error_x='Стандартное отклонение',
            title='Средний уровень удовлетворенности по рейсам',
            color='Количество пассажиров',
            color_continuous_scale='Purples'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return report
    except Exception as e:
        st.error(f"Ошибка генерации отчета: {str(e)}")
        return None

# Основной интерфейс
def main():
    st.title("✈️ Анализ удовлетворенности пассажиров")
    st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    </style>
    """, unsafe_allow_html=True)

    # Загрузка данных
    uploaded_file = st.file_uploader(
        "Загрузите CSV-файл с данными",
        type=["csv"],
        help="Файл должен содержать столбцы: satisfaction_score, flight_id и другие числовые признаки"
    )

    if uploaded_file is not None:
        data = load_data_from_csv(uploaded_file)
        if data is not None:
            st.session_state['raw_data'] = data
            st.success(f"Данные успешно загружены! Всего записей: {len(data):,}")

    if 'raw_data' in st.session_state:
        analysis_option = st.sidebar.radio(
            "Выберите тип анализа:",
            ["Обзор данных", "Очистка данных", "Регрессионный анализ", "Кластеризация", "Отчет по рейсам"],
            index=0
        )

        data = st.session_state['raw_data']

        if analysis_option == "Обзор данных":
            st.header("🔍 Обзор данных")
            show_distribution(data, 'Распределение satisfaction_score (исходные данные)')
            
            with st.expander("Подробная статистика"):
                st.dataframe(data.describe().T.style.background_gradient(cmap='Blues'))

        elif analysis_option == "Очистка данных":
            st.header("🧹 Очистка данных")
            
            if 'clean_data' not in st.session_state:
                st.session_state['clean_data'] = clean_data(data)

            clean_data_df = st.session_state['clean_data']
            
            if clean_data_df is None:
                return

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

        elif analysis_option == "Регрессионный анализ":
            st.header("📈 Регрессионный анализ")

            if 'clean_data' not in st.session_state or st.session_state['clean_data'] is None:
                st.warning("Сначала выполните очистку данных!")
                return

            clean_data_df = st.session_state['clean_data']
            numeric_cols = clean_data_df.select_dtypes(include=['number']).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'satisfaction_score']

            if not numeric_cols:
                st.error("В данных нет числовых признаков для анализа!")
                return

            selected_feature = st.selectbox(
                "Выберите признак для анализа:",
                numeric_cols,
                index=0
            )

            coef = plot_regression(
                clean_data_df[[selected_feature]],
                clean_data_df['satisfaction_score'],
                selected_feature
            )

            if coef is not None:
                st.info(f"""
                **Коэффициент регрессии:** `{coef[0]:.4f}`

                **Интерпретация:**
                - Положительное значение означает, что с ростом '{selected_feature}' растет удовлетворенность
                - Отрицательное значение означает обратную зависимость
                """)

        elif analysis_option == "Кластеризация":
            st.header("🧩 Кластеризация пассажиров")

            if 'clean_data' not in st.session_state or st.session_state['clean_data'] is None:
                st.warning("Сначала выполните очистку данных!")
                return

            clean_data_df = st.session_state['clean_data']
            numeric_cols = clean_data_df.select_dtypes(include=['number']).columns.tolist()

            if len(numeric_cols) < 2:
                st.error("Для кластеризации нужно как минимум 2 числовых признака!")
                return

            col1, col2, col3 = st.columns(3)
            with col1:
                feature1 = st.selectbox("Первый признак", numeric_cols, index=0)
            with col2:
                feature2 = st.selectbox("Второй признак", numeric_cols, index=min(1, len(numeric_cols)-1))
            with col3:
                n_clusters = st.slider("Количество кластеров", 2, 10, 3)

            if st.button("Выполнить кластеризацию"):
                clusters = plot_clusters(
                    clean_data_df[[feature1, feature2]],
                    n_clusters
                )

                if clusters is not None:
                    st.session_state['clusters'] = clusters
                    st.success(f"Пассажиры успешно разделены на {n_clusters} кластера!")
                    st.dataframe(pd.Series(clusters).value_counts().rename("Количество в кластере"))

        elif analysis_option == "Отчет по рейсам":
            st.header("📊 Отчет по рейсам")

            if 'clean_data' not in st.session_state or st.session_state['clean_data'] is None:
                st.warning("Сначала выполните очистку данных!")
                return

            clean_data_df = st.session_state['clean_data']
            report = generate_flight_report(clean_data_df)
            
            if report is not None:
                st.dataframe(
                    report.style.background_gradient(cmap='Purples', subset=['Средняя удовлетворенность'])
                )

                csv = report.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Скачать отчет в CSV",
                    data=csv,
                    file_name='flight_satisfaction_report.csv',
                    mime='text/csv'
                )

if __name__ == "__main__":
    main()