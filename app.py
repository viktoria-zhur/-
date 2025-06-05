import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys

def check_required_columns(data, required_columns):
    """Проверяет наличие обязательных столбцов в данных"""
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        st.error(f"Отсутствуют обязательные столбцы: {', '.join(missing)}")
        return False
    return True

def show_distribution(data, column_name, title, color='skyblue'):
    """Безопасное отображение распределения с проверками"""
    try:
        if data is None or data.empty:
            st.warning("Нет данных для отображения")
            return
            
        if column_name not in data.columns:
            st.error(f"Столбец '{column_name}' не найден в данных")
            st.write("Доступные столбцы:", list(data.columns))
            return
            
        if not pd.api.types.is_numeric_dtype(data[column_name]):
            st.error(f"Столбец '{column_name}' должен содержать числовые данные")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        data[column_name].hist(bins=20, color=color, ax=ax)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(column_name, fontsize=12)
        ax.set_ylabel('Количество', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Ошибка при построении графика: {str(e)}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        st.write(f"Тип ошибки: {exc_type}")
        st.write(f"Строка: {exc_tb.tb_lineno}")

def main():
    st.title("Анализ данных")
    
    uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state['data'] = data
            st.success(f"Данные загружены успешно! Записей: {len(data)}")
            
            # Показываем список всех столбцов для справки
            with st.expander("Просмотр данных"):
                st.write("Первые 5 строк:")
                st.write(data.head())
                st.write("Все столбцы:", list(data.columns))
                
            # Проверяем наличие нужного столбца
            if 'satisfaction_score' not in data.columns:
                st.warning("Столбец 'satisfaction_score' не найден!")
                
                # Предлагаем выбрать альтернативный столбец
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox(
                        "Выберите столбец для анализа вместо satisfaction_score:",
                        numeric_cols
                    )
                    
                    # Анализируем выбранный столбец
                    if st.button("Проанализировать выбранный столбец"):
                        show_distribution(
                            data, 
                            selected_col, 
                            f"Распределение {selected_col}", 
                            'purple'
                        )
                else:
                    st.error("В данных нет числовых столбцов для анализа")
            else:
                # Если нужный столбец есть - анализируем его
                show_distribution(
                    data, 
                    'satisfaction_score', 
                    'Распределение satisfaction_score'
                )
                
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {str(e)}")

if __name__ == "__main__":
    main()