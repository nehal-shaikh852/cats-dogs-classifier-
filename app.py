import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from model import CNN


st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐱",
    layout="centered"
)

st.title("🐱 Dogs vs Cats Classifier")
st.write("Upload an image and the model will predict whether it is a cat or a dog.")

device = torch.device("cpu")

model = CNN()
model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=device
    )
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():

        outputs = model(img_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    classes = ["Cat", "Dog"]

    st.success(
        f"Prediction: {classes[predicted.item()]}"
    )

    st.info(
        f"Confidence: {confidence.item()*100:.2f}%"
    )