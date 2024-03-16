import torch
from PIL import Image
from torchvision import transforms


def test() -> None:
    with open("model.pth", "rb") as file:
        model = torch.load(file)
    
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    
    for i in range(1, 7):
        image_path = f"dataset/test/test{i}.jpeg"
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
        
        _, predicted = torch.max(output, 1)
        
        sign_name = classes[predicted.item()]
        
        print(f"Prediction for image {i}: {sign_name}")
