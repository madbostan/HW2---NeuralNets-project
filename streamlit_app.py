import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# --- Определение архитектуры (должно совпадать с обученной моделью) ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Загрузка обученной модели ---
@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load('./cifar_net.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --- Интерфейс Streamlit ---
st.title("🧠 Классификация изображений CIFAR-10")
uploaded_file = st.file_uploader("Загрузите изображение (jpg/png)", type=["jpg", "png"])

if uploaded_file is not None:
    # открываем изображение
    image = Image.open(uploaded_file).convert('RGB')

    # центрируем и оформляем изображение
    st.markdown("### Загруженное изображение:")
    st.image(image, caption="Ваше изображение", use_column_width=False, width=200)

    # преобразуем изображение для модели
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_tensor = transform(image).unsqueeze(0)

    # предсказание
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    label = classes[predicted[0]]

    # аккуратный вывод
    st.markdown("---")
    st.markdown(f"## Предсказание модели: **{label.upper()}**")
    st.info("Модель обучена на датасете CIFAR-10 (10 классов изображений).") 
