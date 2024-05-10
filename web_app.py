import streamlit as st
import os
from PIL import Image
from py_modules.model import ClassificationModel
from py_modules.inference import predict
from py_modules.utils.generate_grad_cam import generate_grad_cam
import torch
import config


def main():
    st.title("Document Classification ")

    st.sidebar.title("Upload Image")
    uploaded_image = st.sidebar.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    st.sidebar.title("Select Model")
    selected_model = st.sidebar.selectbox("Choose a model", ["ResNet50"])

    if selected_model == "ResNet50":
        checkpoint_paths = os.path.join(
            "Checkpoints",
            "May-10_14-43-13",
            "model_38.pth",
        )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")

        if selected_model and uploaded_image:
            if selected_model == "ResNet50":
                model = ClassificationModel(num_classes=config.num_classes)
            # elif selected_model == "Model 2":
            #     model = Model2()

            checkpoint_path = checkpoint_paths

            model.load_state_dict(torch.load(checkpoint_path))
            # Classify the image
            class_index, confidence = predict(model, image)

            st.write(f"Class Index: {class_index}")
            st.write(f"Confidence: {confidence:.2f}")

            # Option to view CAM
            if st.sidebar.checkbox("View Class Activation Map (CAM)"):
                # Generate CAM
                cam = generate_grad_cam(model, image, class_index)
                # Apply heatmap
                st.image(
                    cam,
                    caption="Class Activation Map (CAM)",
                    use_column_width=True,
                )

        else:
            st.write("Please select a model to proceed.")

    else:
        st.write("Please upload an image to classify.")


if __name__ == "__main__":
    main()
