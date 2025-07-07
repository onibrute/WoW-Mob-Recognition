import os
import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Setări
MODEL_PATH = r"C:\Users\Administrator\Desktop\TIA proiect\best_model.keras"  # Calea către modelul antrenat
IMAGE_FOLDER = r"C:\Users\Administrator\Desktop\TIA proiect\test"  # Folderul cu imagini
IMAGE_SIZE = (224, 224)  # Dimensiunea de intrare a modelului
LABELS = [
    'ardo_dirtpaw', 'blackrock_battleworg', 'blackrock_champion', 'blackrock_drake', 'blackrock_drakerider',
    'blackrock_gladiator', 'blackrock_guard', 'blackrock_hunter', 'blackrock_overseer', 'blackrock_renegade',
    'blackrock_shadowcaster', 'blackrock_warden', 'blackrock_worg_captain', 'canyon_ettin', 'cow', 'dire_condor',
    'gathllzog', 'general_fangcore', 'gnollfeaster', 'great_goretusk', 'impaled_blackrock_orc', 'kazon',
    'mountain_cottontail', 'murloc_any', 'overlord_barbarius', 'redridge_any', 'redridge_citizen_female',
    'redridge_citizen_male', 'redridge_fox', 'redridge_garrison_watchman', 'redridge_mystic', 'seeker_aqualon',
    'shadowhide_darkweaver', 'shadowhide_gnoll', 'shadowhide_warrior', 'snareflare', 'squiddic', 'tarantula',
    'yowler(redridge boss)'
]

# Încarcă modelul
try:
    model = load_model(MODEL_PATH)
    print("Model încărcat cu succes.")
except Exception as e:
    print(f"Eroare la încărcarea modelului: {e}")
    exit()

def predict_image(image_path):
    """Preprocesează imaginea și returnează predicția modelului."""
    try:
        # Încarcă și preprocesează imaginea
        image = load_img(image_path, target_size=IMAGE_SIZE)
        image_array = img_to_array(image) / 255.0  # Normalizează imaginea la intervalul [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Adaugă dimensiunea batch

        print(f"Dimensiunea imaginii preprocesate: {image_array.shape}")

        # Generează predicții
        start_time = time.time()
        predictions = model.predict(image_array, verbose=0)
        elapsed_time = time.time() - start_time

        # Determină clasa prezisă și încrederea
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]

        return LABELS[predicted_class], confidence, elapsed_time
    except Exception as e:
        print(f"Eroare la prelucrarea imaginii {image_path}: {e}")
        return None, None, None

def main():
    """Funcția principală pentru procesarea imaginilor."""
    # Verifică dacă folderul cu imagini există
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Folderul {IMAGE_FOLDER} nu există.")
        return

    # Parcurge toate fișierele din folderul de imagini
    for root, _, files in os.walk(IMAGE_FOLDER):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, file)

                # Obține predicțiile pentru fiecare imagine
                label, confidence, elapsed_time = predict_image(image_path)

                # Afișează rezultatele pentru fiecare imagine
                if label is not None:
                    print(f"Imagine: {image_path}\n"
                          f"Eticheta prezisă: {label}\n"
                          f"Precizie: {confidence:.2f}\n"
                          f"Timp: {elapsed_time:.2f}s\n")

if __name__ == "__main__":
    main()
