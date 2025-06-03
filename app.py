import base64

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor

st.sidebar.title("РГР по дисциплине «Машинное обучение и большие данные»")
page = st.sidebar.radio("Выберите страницу", ("О разработчике", "О наборе данных", "Визуализация", "Предсказание"))

if page == "О разработчике":
    st.title("Информация о разработчике")
    st.image(
        "photo.jpg",
        caption="Фото разработчика",
        width=200,
        output_format="JPEG" 
    )
    st.markdown("**ФИО:** Завьялова Анастасия Николаевна")
    st.markdown("**Группа:** ФИТ-232/2")
    st.markdown(
        "**Тема РГР:** «Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных»"
    )

elif page == "О наборе данных":
    st.title("Информация о наборе данных")
    st.markdown("### Описание предметной области")
    st.markdown(
        "Этот классический набор данных содержит информацию о ценах и других характеристиках почти 54 000 бриллиантов."
    )
    st.markdown("### Признаки набора данных")
    st.markdown(
        """- **unnamed: 0** — столбец без названия, содержащий индексы, его удалим  
- **carat** — Вес бриллианта в каратах  
- **cut** — качество огранки бриллианта. Качество в порядке возрастания: Fair, Good, Very Good, Premium, Ideal  
- **color** — Цвет бриллианта, где D - лучший, а J - худший  
- **clarity** — Насколько заметны включения в бриллианте (от лучшего к худшему: FL, IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, I3)  
- **depth** — Глубина %: высота алмаза, измеренная от вершины до основания, деленная на средний диаметр его основания  
- **table** — ширина таблицы алмаза, выраженная в процентах от его среднего диаметра  
- **price** — цена бриллианта  
- **x** — длина, мм  
- **y** — ширина, мм  
- **z** — глубина, мм"""
    )
    st.markdown("### Особенности предобработки и EDA")
    st.markdown(
        """- Очистка данных и устранение выбросов  
- Визуализация распределений признаков  
- Корреляционный анализ и анализ зависимостей между переменными"""
    )
elif page == "Визуализация":
    st.title("Визуализация зависимостей в наборе данных")
    st.markdown(
        "Ниже представлены 5 изображений, демонстрирующих визуальные аспекты набора данных. "
    )
    st.image(
        "correlation_heatmap.png",
        use_container_width=True,
        output_format="PNG" 
    )
    st.image(
        "pairplot.png",
        use_container_width=True,
        output_format="JPEG"  
    )
    st.image(
        "boxplot_price_vs_color.png",
        use_container_width=True,
        output_format="PNG" 
    )
    st.image(
        "before_outliers.png",
        caption="До удаления выбросов",
        use_container_width=True,
        output_format="PNG"  
    )
    st.image(
        "after_outliers.png",
        caption="После удаления выбросов",
        use_container_width=True,
        output_format="PNG" 
    )


elif page == "Предсказание":
    st.title("Интерфейс предсказания модели ML")
    st.header("Получите предсказание с помощью загруженных или вручную введённых данных")

    st.subheader("Выберите модель для предсказания:")
    model_choice = st.selectbox("Модель",
                                ["ML1: Классическая модель",
                                 "ML2: Ансамблевая модель (Бустинг)",
                                 "ML3: Продвинутый градиентный бустинг (CatBoost)",
                                 "ML4: Ансамблевая модель (Бэггинг)",
                                 "ML5: Ансамблевая модель (Стэкинг)",
                                 "ML6: Глубокая полносвязная нейронная сеть"])

    model_files = {
        "ML1: Классическая модель": "ridge_model.pkl",
        "ML2: Ансамблевая модель (Бустинг)": "best_gbr_model.pkl",
        "ML3: Продвинутый градиентный бустинг (CatBoost)": "catboost_model.cbm",
        "ML4: Ансамблевая модель (Бэггинг)": "best_bag_reg_model.pkl",
        "ML5: Ансамблевая модель (Cтэкинг)": "stack_reg_model.pkl",
        "ML6: Глубокая полносвязная нейронная сеть": "best_model_hyperopt.pkl"
    }
    model_file = model_files.get(model_choice)

    if model_choice == "ML3: Продвинутый градиентный бустинг (CatBoost)":
        model = CatBoostRegressor()
        try:
            model.load_model(model_file)
        except Exception as e:
            st.error(f"Ошибка при загрузке модели CatBoost из файла {model_file}: {e}")
            st.stop()
    else:
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            st.error(f"Ошибка при загрузке модели из файла {model_file}: {e}")
            st.stop()

    scaler_file = "scaler.pkl"
    try:
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Ошибка при загрузке scaler из файла {scaler_file}: {e}")
        st.stop()

    feature_names = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']

    st.subheader("1. Загрузка данных через CSV")
    uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=['csv'])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Загруженные данные:")
            st.write(data.head())

            missing_cols = [col for col in feature_names if col not in data.columns]
            if missing_cols:
                st.error(f"В загруженном файле отсутствуют колонки: {missing_cols}")
            else:
                input_data = data[feature_names]
                input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=feature_names)

                if model_choice == "ML1: Классическая модель" or model_choice == "ML5: Ансамблевая модель (Стэкинг)":
                    predictions_log = model.predict(input_data_scaled)
                    predictions = np.exp(predictions_log)
                else:
                    predictions = model.predict(input_data_scaled)

                formatted_predictions = [f"${pred:,.2f}" for pred in predictions]
                data['Предсказание'] = formatted_predictions
                st.write("Результаты предсказания:")
                st.write(data[['Предсказание']])
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")

    st.markdown("---")
    st.subheader("2. Ручной ввод данных")
    st.info("Если CSV-файл не загружен, введите необходимые данные вручную:")

    with st.form("manual_input_form"):
        user_data = {}
        for feature in feature_names:
            user_data[feature] = st.text_input(f"Введите значение для {feature}")
        submitted = st.form_submit_button("Получить предсказание")

        if submitted:
            def validate_input(data):
                errors = {}
                validated_data = {}
                for key, value in data.items():
                    try:
                        validated_data[key] = float(value)
                    except ValueError:
                        errors[key] = f"Значение '{value}' не является числом."
                return validated_data, errors

            validated_data, errors = validate_input(user_data)
            if errors:
                for key, error in errors.items():
                    st.error(f"Ошибка в поле '{key}': {error}")
            else:
                input_df = pd.DataFrame([validated_data])
                input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)
                try:
                    if model_choice == "ML1: Классическая модель" or model_choice == "ML5: Ансамблевая модель (Стэкинг)":
                        prediction_log = model.predict(input_df_scaled)[0]
                        prediction = np.exp(prediction_log)
                    else:
                        prediction = model.predict(input_df_scaled)[0]
                    formatted_prediction = f"${prediction:,.2f}"
                    st.success(f"Предсказанная цена: {formatted_prediction}")
                except Exception as e:
                    st.error(f"Ошибка при предсказании: {e}")

st.markdown("---")
st.markdown("### Полезные ссылки:")
st.markdown("[GitHub-репозиторий](https://github.com/Anastasiiyyaa/ML_RGR)")
st.markdown("[Web-сервис на Hugging Face](https://huggingface.co/spaces/Anastasiya3838/RGRML)")
