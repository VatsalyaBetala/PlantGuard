import json
from pathlib import Path

from PIL import Image

from src.model_catalog import get_backend_name, plant_model_name
from src.model_registry import get_model


def main():
    backend = get_backend_name()
    model_name = plant_model_name(backend)

    image_path = Path("scripts/_smoke_test.jpg")
    if not image_path.exists():
        Image.new("RGB", (224, 224), color=(128, 128, 128)).save(image_path)

    model = get_model(model_name)
    tensor = model.preprocess(str(image_path))
    output = model.predict(tensor)
    result = model.postprocess(output)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
