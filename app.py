import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def safe_load_data(uploaded_file):
    """Безопасная загрузка данных с обработкой всех возможных ошибок"""
    try:
        data = pd.read_csv(uploaded_file)
        if data.empty:
            st.error("Файл не содержит данных")
            return None
        return data
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {str(e)}")
        return None

def find_suitable_columns(data):
    """Автоматический поиск подходящих столбцов для анализа"""
    # 1. Попробуем найти столбцы с похожими названиями
    possible_score_columns = [col for col in data.columns 
                           if 'score' in col.lower() or 'satisfaction' in col.lower()]
    
    # 2. Ищем числовые столбцы
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # 3. Ищем столбцы с подходящими значениями (диапазон 1-5 или 1-10)
    suitable_cols = []
    for col in data.columns:
        try:
            unique_vals = pd.to_numeric(data[col].dropna()).unique()
            if all(1 <= x <= 10 for x in unique_vals):
                suitable_cols.append(col)
        except:
            continue
    
    return {
        'possible_scores': possible_score_columns,
        'numeric_cols': numeric_cols,
        'suitable_cols': suitable_cols
    }

def smart_plot(data, column_name):
    """Умное построение графика с автоматической настройкой"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Автоматическое определение типа графика
        if data[column_name].nunique() > 10:
            data[column_name].hist(bins=20, ax=ax, color='#1f77b4')
            ax.set_ylabel('Частота')
        else:
            value_counts = data[column_name].value_counts().sort_index()
            value_counts.plot(kind='bar', ax=ax, color='#2ca02c')
            ax.set_ylabel('Количество')
        
        ax.set_title(f'Распределение {column_name}')
        ax.set_xlabel(column_name)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Не удалось построить график: {str(e)}")

def main():
    st.title("📊 Умный анализатор данных")
    st.markdown("""
    <style>
    .small-font { font-size:12px !important; color:gray; }
    </style>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type=["csv"])
    
    if uploaded_file is not None:
        data = safe_load_data(uploaded_file)
        if data is not None:
            st.success(f"Успешно загружено {len(data)} записей")
            
            # Анализ доступных столбцов
            columns_info = find_suitable_columns(data)
            
            # Показываем информацию о данных
            with st.expander("🔍 Просмотр данных"):
                st.write("Первые 5 строк:")
                st.write(data.head())
                st.markdown(f'<p class="small-font">Все столбцы: {list(data.columns)}</p>', 
                          unsafe_allow_html=True)
            
            # Основной анализ
            st.header("Анализ данных")
            
            # Вариант 1: Нашли идеальный столбец
            if columns_info['suitable_cols']:
                best_col = columns_info['suitable_cols'][0]
                st.info(f"Автоматически выбран столбец для анализа: '{best_col}'")
                smart_plot(data, best_col)
            
            # Вариант 2: Есть похожие столбцы
            elif columns_info['possible_scores']:
                selected_col = st.selectbox(
                    "Выберите столбец для анализа (автоподбор):",
                    columns_info['possible_scores']
                )
                smart_plot(data, selected_col)
            
            # Вариант 3: Есть числовые столбцы
            elif columns_info['numeric_cols']:
                selected_col = st.selectbox(
                    "Выберите числовой столбец для анализа:",
                    columns_info['numeric_cols']
                )
                smart_plot(data, selected_col)
            
            # Вариант 4: Совсем нет подходящих столбцов
            else:
                st.warning("Не найдено подходящих числовых столбцов для анализа")
                
                # Пробуем найти хотя бы один анализируемый столбец
                all_cols = data.columns.tolist()
                if all_cols:
                    st.info("Попробуйте проанализировать любой столбец:")
                    selected_col = st.selectbox("Выберите столбец:", all_cols)
                    
                    try:
                        smart_plot(data, selected_col)
                    except:
                        st.error("Не удалось проанализировать выбранный столбец")
                        st.write("Пример значений:", data[selected_col].head().tolist())

if __name__ == "__main__":
    main()