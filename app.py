import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import io

# Функция загрузки данных с обработкой загружаемого файла
def load_data_from_csv(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {e}")
        return None

def show_initial_distribution(data):
    if 'satisfaction_score' not in data.columns:
        st.error("В данных отсутствует столбец 'satisfaction_score'")
        return
        
    fig, ax = plt.subplots(figsize=(8, 5))
    data['satisfaction_score'].hist(bins=20, color='skyblue', ax=ax)
    ax.set_title('Распределение satisfaction_score (исходные данные)', fontsize=14)
    ax.set_xlabel('Уровень удовлетворенности', fontsize=12)
    ax.set_ylabel('Количество пассажиров', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

def clean_data(data):
    if data is None:
        return None

    # Удаление пропущенных значений
    data_clean = data.dropna()

    # Проверка наличия столбца satisfaction_score
    if 'satisfaction_score' not in data_clean.columns:
        st.error("В данных отсутствует столбец 'satisfaction_score'")
        return None
        
    # Удаление аномальных значений
    data_clean = data_clean[
        (data_clean['satisfaction_score'] >= 1) &
        (data_clean['satisfaction_score'] <= 5)
    ]
    return data_clean

def show_cleaned_distribution(data_clean):
    if data_clean is None or 'satisfaction_score' not in data_clean.columns:
        return
        
    fig, ax = plt.subplots(figsize=(8, 5))
    data_clean['satisfaction_score'].hist(bins=20, color='orange', ax=ax)
    ax.set_title('Распределение после очистки', fontsize=14)
    ax.set_xlabel('Уровень удовлетворенности', fontsize=12)
    ax.set_ylabel('Количество пассажиров', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

def perform_regression_analysis(X, y):
    model = LinearRegression()
    model.fit(X, y)

    if X.shape[1] == 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X, y, color='blue', alpha=0.5, label='Данные')
        ax.plot(X, model.predict(X), color='red', linewidth=2, label='Линия регрессии')
        ax.set_title(f'Зависимость удовлетворенности от {X.columns[0]}', fontsize=14)
        ax.set_xlabel(X.columns[0], fontsize=12)
        ax.set_ylabel('Уровень удовлетворенности', fontsize=12)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    return model.coef_

def perform_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)

    if data.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            data.iloc[:, 0],
            data.iloc[:, 1],
            c=clusters,
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        ax.set_title(f'Кластеризация по {data.columns[0]} и {data.columns[1]}', fontsize=14)
        ax.set_xlabel(data.columns[0], fontsize=12)
        ax.set_ylabel(data.columns[1], fontsize=12)
        plt.colorbar(scatter, label='Кластер')
        plt.tight_layout()
        st.pyplot(fig)

    return clusters

def generate_report(data):
    if 'flight_id' not in data.columns or 'satisfaction_score' not in data.columns:
        st.error("Для отчета необходимы столбцы 'flight_id' и 'satisfaction_score'")
        return None
        
    report = data.groupby('flight_id')['satisfaction_score'].agg(['mean', 'count', 'std'])
    report.columns = ['Средняя удовлетворенность', 'Количество пассажиров', 'Стандартное отклонение']

    fig, ax = plt.subplots(figsize=(12, 6))
    report['Средняя удовлетворенность'].sort_values().plot(
        kind='barh',
        color='purple',
        ax=ax,
        xerr=report['Стандартное отклонение']
    )
    ax.set_title('Средний уровень удовлетворенности по рейсам', fontsize=14)
    ax.set_xlabel('Уровень удовлетворенности', fontsize=12)
    ax.set_ylabel('Номер рейса', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    return report

def main():
    st.set_page_config(
        page_title="Анализ удовлетворенности пассажиров",
        page_icon="✈️",
        layout="wide"
    )

    st.title("✈️ Анализ удовлетворенности пассажиров")
    st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    </style>
    """, unsafe_allow_html=True)

    # Загрузка файла
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
            show_initial_distribution(data)

            with st.expander("Подробная статистика"):
                st.dataframe(data.describe().T.style.background_gradient(cmap='Blues'))

        elif analysis_option == "Очистка данных":
            st.header("🧹 Очистка данных")

            if 'clean_data' not in st.session_state:
                st.session_state['clean_data'] = clean_data(data)

            clean_data_df = st.session_state['clean_data']
            
            if clean_data_df is None:
                st.error("Не удалось очистить данные. Проверьте наличие необходимых столбцов.")
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

            show_cleaned_distribution(clean_data_df)

        elif analysis_option == "Регрессионный анализ":
            st.header("📈 Регрессионный анализ")

            if 'clean_data' not in st.session_state:
                st.warning("Сначала выполните очистку данных!")
                st.stop()

            clean_data_df = st.session_state['clean_data']
            
            if clean_data_df is None:
                st.error("Данные не были очищены. Проверьте наличие ошибок на этапе очистки.")
                return
                
            if 'satisfaction_score' not in clean_data_df.columns:
                st.error("В данных отсутствует столбец 'satisfaction_score'")
                return

            numeric_cols = clean_data_df.select_dtypes(include=['number']).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'satisfaction_score']

            if not numeric_cols:
                st.error("В данных нет числовых признаков для анализа!")
                st.stop()

            selected_feature = st.selectbox(
                "Выберите признак для анализа:",
                numeric_cols,
                index=0
            )

            coef = perform_regression_analysis(
                clean_data_df[[selected_feature]],
                clean_data_df['satisfaction_score']
            )

            st.info(f"""
            **Коэффициент регрессии:** `{coef[0]:.4f}`

            **Интерпретация:**
            - Положительное значение означает, что с ростом '{selected_feature}' растет удовлетворенность
            - Отрицательное значение означает обратную зависимость
            """)

        elif analysis_option == "Кластеризация":
            st.header("🧩 Кластеризация пассажиров")

            if 'clean_data' not in st.session_state:
                st.warning("Сначала выполните очистку данных!")
                st.stop()

            clean_data_df = st.session_state['clean_data']
            
            if clean_data_df is None:
                st.error("Данные не были очищены. Проверьте наличие ошибок на этапе очистки.")
                return

            numeric_cols = clean_data_df.select_dtypes(include=['number']).columns.tolist()

            if len(numeric_cols) < 2:
                st.error("Для кластеризации нужно как минимум 2 числовых признака!")
                st.stop()

            col1, col2, col3 = st.columns(3)

            with col1:
                feature1 = st.selectbox("Первый признак", numeric_cols, index=0)

            with col2:
                default_idx = 1 if len(numeric_cols) > 1 else 0
                feature2 = st.selectbox("Второй признак", numeric_cols, index=default_idx)

            with col3:
                n_clusters = st.slider("Количество кластеров", 2, 10, 3)

            if st.button("Выполнить кластеризацию"):
                clusters = perform_clustering(
                    clean_data_df[[feature1, feature2]],
                    n_clusters
                )

                if clusters is not None:
                    st.session_state['clusters'] = clusters
                    st.success(f"Пассажиры успешно разделены на {n_clusters} кластера!")

                    cluster_stats = pd.Series(clusters).value_counts().sort_index()
                    st.dataframe(cluster_stats.rename("Количество в кластере"))

        elif analysis_option == "Отчет по рейсам":
            st.header("📊 Отчет по рейсам")

            if 'clean_data' not in st.session_state:
                st.warning("Сначала выполните очистку данных!")
                st.stop()

            clean_data_df = st.session_state['clean_data']
            
            if clean_data_df is None:
                st.error("Данные не были очищены. Проверьте наличие ошибок на этапе очистки.")
                return

            report = generate_report(clean_data_df)
            
            if report is None:
                return

            st.dataframe(
                report.sort_values('Средняя удовлетворенность', ascending=False)
                .style.background_gradient(cmap='Purples', subset=['Средняя удовлетворенность'])
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