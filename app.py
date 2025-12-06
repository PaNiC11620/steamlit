import streamlit as st
from PIL import Image
from model import predict, models_dict
from translator import get_translation

st.set_page_config(
    page_title="Розпізнавання зображень — AI Classifier",
    layout="wide"
)

st.title("Розпізнавання зображень за допомогою нейронних мереж")
st.markdown("---")


with st.sidebar:
    st.header("⚙️ Налаштування")

    model_name = st.selectbox(
        "Оберіть модель нейронної мережі:",
        options=list(models_dict.keys()),
        index=1  
    )

    model_info = {
        "EfficientNetB0": "Ефективна модель з балансом між точністю та швидкістю",
        "MobileNetV2": "Легка модель, оптимізована для мобільних пристроїв",
        "ResNet50": "Глибока мережа з високою точністю розпізнавання"
    }

    st.info(f"**Про модель:** {model_info[model_name]}")

    top_k = st.slider(
        "Кількість найімовірніших класів:",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

    show_original = st.checkbox("Показувати англійські назви", value=True)

col1, col2 = st.columns([1, 1])

with col1:
        st.subheader("Завантаження зображення")
        uploaded_file = st.file_uploader(
            "Оберіть зображення (JPG, JPEG або PNG)",
            type=["jpg", "jpeg", "png"],
            help="Підтримуються формати: JPG, JPEG, PNG",
            key="classification_uploader"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Завантажене зображення", use_container_width=True)

            st.caption(f"Розмір: {image.size[0]} x {image.size[1]} пікселів | Формат: {image.format}")

with col2:
        st.subheader("Результати розпізнавання")

        if uploaded_file:
            if st.button("Розпізнати зображення", type="primary", use_container_width=True):
                with st.spinner(f"Виконується класифікація за допомогою {model_name}..."):
                    preds = predict(image, model_name=model_name, top=top_k)

                    st.success("✅ Розпізнавання завершено!")
                    st.markdown("---")

                    for i, (class_id, label, prob) in enumerate(preds, 1):
                        translated_label = get_translation(label, show_original=show_original)

                        if prob > 0.5:
                            bar_color = "🟢"
                        elif prob > 0.2:
                            bar_color = "🟡"
                        else:
                            bar_color = "🔴"

                        st.markdown(f"### {i}. {translated_label}")
                        st.progress(float(prob))
                        st.markdown(f"{bar_color} **Ймовірність: {prob*100:.2f}%**")
                        st.markdown("---")
        else:
            st.info("Завантажте зображення, щоб почати розпізнавання")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Створено з використанням TensorFlow/Keras та Streamlit</p>
        <p>Моделі: EfficientNetB0, MobileNetV2, ResNet50</p>
    </div>
    """,
    unsafe_allow_html=True
)