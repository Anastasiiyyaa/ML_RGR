import base64
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor

st.sidebar.title("РГР по дисциплине «Машинное обучение и большие данные»")
page = st.sidebar.radio("Выберите страницу", 
                         ("О разработчике", "О наборе данных", "Визуализация", "Предсказание"))

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
    st.markdown("**Тема РГР:** «Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных»")

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
- **cut** — Качество огранки бриллианта. Порядок: Fair, Good, Very Good, Premium, Ideal  
- **color** — Цвет бриллианта, где D - лучший, а J - худший  
- **clarity** — Чистота бриллианта (от лучшего к худшему: FL, IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, I3)  
- **depth** — Глубина %  
- **table** — Ширина таблицы алмаза в процентах  
- **price** — Цена бриллианта  
- **x** — Длина, мм  
- **y** — Ширина, мм  
- **z** — Глубина, мм"""
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
        "Ниже представлены 5 изображений, демонстрирующих визуальные аспекты набора данных."
    )
    st.image("correlation_heatmap.png", use_container_width=True, output_format="PNG")
    st.image("pairplot.png", use_container_width=True, output_format="JPEG")
    st.image("boxplot_price_vs_color.png", use_container_width=True, output_format="PNG")
    st.image("before_outliers.png", caption="До удаления выбросов", use_container_width=True, output_format="PNG")
    st.image("after_outliers.png", caption="После удаления выбросов", use_container_width=True, output_format="PNG")

elif page == "Предсказание":
    st.title("Интерфейс предсказания модели ML")
    st.header("Получите предсказание с помощью загруженных или вручную введённых данных")

    st.subheader("Выберите модели для предсказания:")
    model_choices = st.multiselect("Модели", 
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
        "ML5: Ансамблевая модель (Стэкинг)": "stack_reg_model.pkl",
        "ML6: Глубокая полносвязная нейронная сеть": "best_model_hyperopt.pkl"
    }

    loaded_models = {}
    for choice in model_choices:
        model_file = model_files.get(choice)
        if choice == "ML3: Продвинутый градиентный бустинг (CatBoost)":
            model_loaded = CatBoostRegressor()
            try:
                model_loaded.load_model(model_file)
                loaded_models[choice] = model_loaded
            except Exception as e:
                st.error(f"Ошибка при загрузке модели CatBoost из файла {model_file}: {e}")
        else:
            try:
                with open(model_file, 'rb') as f:
                    model_loaded = pickle.load(f)
                    loaded_models[choice] = model_loaded
            except Exception as e:
                st.error(f"Ошибка при загрузке модели из файла {model_file}: {e}")
    
    scaler_file = "scaler.pkl"
    try:
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Ошибка при загрузке scaler из файла {scaler_file}: {e}")
        st.stop()
    
    feature_names = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']

    categorical_options = {
        'cut': {
            'options': ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
            'mapping': {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
        },
        'color': {
            'options': ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
            'mapping': {'J': 7, 'I': 6, 'H': 5, 'G': 4, 'F': 3, 'E': 2, 'D': 1}
        },
        'clarity': {
            'options': ['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1', 'I2', 'I3'],
            'mapping': {'FL': 0, 'IF': 1, 'VVS1': 2, 'VVS2': 3, 'VS1': 4, 'VS2': 5, 'SI1': 6, 'SI2': 7, 'I1': 8, 'I2': 9, 'I3': 10}
        }
    }

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
                
                for cat_feature in categorical_options.keys():
                    if cat_feature in input_data.columns:
                        input_data[cat_feature] = input_data[cat_feature].apply(
                            lambda x: categorical_options[cat_feature]['mapping'].get(x, x)
                        )
       
                input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=feature_names)
      
                results = {}
                for model_name, model_obj in loaded_models.items():
                    if model_name == "ML1: Классическая модель" or model_name == "ML5: Ансамблевая модель (Стэкинг)":
                        predictions_log = model_obj.predict(input_data_scaled)
                        predictions = np.exp(predictions_log)
                    else:
                        predictions = model_obj.predict(input_data_scaled)
                    results[model_name] = [f"${pred:,.2f}" for pred in predictions]
                    
                for model_name, preds in results.items():
                    data[f"Предсказание ({model_name})"] = preds
                
                st.write("Результаты предсказания:")
                st.write(data)
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")

    st.markdown("---")
    st.subheader("2. Ручной ввод данных")
    st.info("Если CSV-файл не загружен, введите необходимые данные вручную:")

    slider_ranges = {
        'carat': (0.1, 5.0),   
        'depth': (40.0, 80.0),    
        'table': (40.0, 100.0),  
        'x': (0.0, 10.0),         
        'y': (0.0, 10.0),        
        'z': (0.0, 10.0)      
    }

    with st.form("manual_input_form"):
        user_data = {}
        for feature in feature_names:
            if feature in categorical_options:
                user_data[feature] = st.selectbox(f"Выберите значение для {feature}", categorical_options[feature]['options'])
            else:
                if feature in slider_ranges:
                    min_val, max_val = slider_ranges[feature]
                    user_data[feature] = st.slider(f"Выберите значение для {feature}", min_val, max_val, value=min_val, step=0.1)
                else:
                    user_data[feature] = st.text_input(f"Введите значение для {feature}")
        submitted = st.form_submit_button("Получить предсказание")
        
        if submitted:
            def validate_input(data):
                errors = {}
                validated_data = {}
                for key, value in data.items():
                    if key in categorical_options:
                        validated_data[key] = categorical_options[key]['mapping'].get(value, None)
                        if validated_data[key] is None:
                            errors[key] = f"Неверное значение {value} для {key}."
                    else:
                        validated_data[key] = value
                return validated_data, errors

            validated_data, errors = validate_input(user_data)
            if errors:
                for key, error in errors.items():
                    st.error(f"Ошибка в поле '{key}': {error}")
            else:
                input_df = pd.DataFrame([validated_data])
                input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)
                results = {}
                for model_name, model_obj in loaded_models.items():
                    try:
                        if model_name == "ML1: Классическая модель" or model_name == "ML5: Ансамблевая модель (Стэкинг)":
                            prediction_log = model_obj.predict(input_df_scaled)[0]
                            prediction = np.exp(prediction_log)
                        else:
                            prediction = model_obj.predict(input_df_scaled)[0]
                        results[model_name] = f"${prediction:,.2f}"
                    except Exception as e:
                        results[model_name] = f"Ошибка: {e}"
                for model_name, pred in results.items():
                    st.success(f"Предсказание ({model_name}): {pred}")

    
st.markdown("---")
st.markdown("### Полезные ссылки:")
st.markdown("[GitHub-репозиторий](https://github.com/Anastasiiyyaa/ML_RGR)")
st.markdown("[Web-сервис на Hugging Face](https://huggingface.co/spaces/Anastasiya3838/RGRML)")
