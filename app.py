import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

def safe_load_data(uploaded_file):
    """Безопасная загрузка данных с обработкой всех форматов"""
    try:
        # Проверяем тип файла
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            try:
                return pd.read_excel(uploaded_file, engine='openpyxl')
            except ImportError:
                st.error("""Для чтения Excel файлов требуется библиотека openpyxl.
                        Установите её командой: `pip install openpyxl`""")
                return None
        else:
            st.error("Неподдерживаемый формат файла. Загрузите CSV или Excel файл.")
            return None
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {str(e)}")
        return None

def main():
    st.title("✈️ Анализатор данных авиапассажиров")
    
    st.markdown("""
    ### Инструкция:
    1. Загрузите файл в формате CSV или Excel
    2. Для Excel файлов убедитесь, что установлен openpyxl:
       ```
       pip install openpyxl
       ```
    """)
    
    uploaded_file = st.file_uploader("Выберите файл", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        data = safe_load_data(uploaded_file)
        if data is not None:
            st.success(f"Успешно загружено {len(data)} записей")
            
            # Показываем первые строки
            st.subheader("Предпросмотр данных")
            st.write(data.head())
            
            # Анализ данных
            st.subheader("Анализ данных")
            
            # Проверяем наличие нужных столбцов
            if 'satisfaction' in data.columns:
                st.write("Распределение удовлетворенности:")
                fig, ax = plt.subplots()
                data['satisfaction'].value_counts().plot(kind='bar', ax=ax)
                st.pyplot(fig)
            
            if 'Age' in data.columns:
                st.write("Распределение возраста:")
                fig, ax = plt.subplots()
                data['Age'].plot(kind='hist', bins=20, ax=ax)
                st.pyplot(fig)

if __name__ == "__main__":
    main()