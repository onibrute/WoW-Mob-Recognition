import numpy as np
import tensorflow as tf
import pyautogui
import time
import tkinter as tk
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont
import os
from loot import loot_table

# Calea către modelul salvat
model_path = r"C:\Users\Administrator\Desktop\TIA proiect\best_model.keras"

# Încărcarea modelului
model = tf.keras.models.load_model(model_path)

# Dimensiunea imaginii
image_size = (224, 224)

# Numele claselor
class_names = ['ardo dirtpaw', 'blackrock battleworg', 'blackrock champion', 'blackrock drake', 'blackrock drakerider',
               'blackrock gladiator',
               'blackrock guard', 'blackrock hunter', 'blackrock overseer', 'blackrock renegade',
               'blackrock shadowcaster', 'blackrock warden',
               'blackrock worg captain', 'canyon ettin', 'cow', 'dire condor', 'gathllzog', 'general fangcore',
               'gnollfeaster', 'great goretusk',
               'impaled blackrock orc', 'kazon', 'mountain cottontail', 'murloc any', 'overlord barbarius',
               'redridge any', 'redridge citizen female',
               'redridge citizen male', 'redridge fox', 'redridge garrison watchman', 'redridge mystic',
               'seeker aqualon', 'shadowhide darkweaver',
               'shadowhide gnoll', 'shadowhide warrior', 'snareflare', 'squiddic', 'tarantula', 'yowler(redridge boss)']


# Preprocesarea imaginii în formatul corect pentru model
def preprocess_image(img_path):
    """Preproceses the image to the correct format for the model."""
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = tf.image.convert_image_dtype(img_array, tf.float32)  # Asigură-te că tipul de date este float32
    img_array = tf.expand_dims(img_array, 0)  # Creează dimensiunea batch-ului
    return img_array



# Funcția de predicție
def predict_image(img_array):
    """Încarcă o imagine, o preprocesează și face o predicție."""
    predictions = model(img_array)  # Folosește model(img_array) în loc de model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"Predicted class: {predicted_class} with confidence {confidence:.2f}")
    return predicted_class, confidence


# Capturarea zonei de ecran
def capture_screen(region):
    """Captează o zonă specifică a ecranului"""
    screenshot = pyautogui.screenshot(region=region)
    screenshot.save(r"C:\Users\Administrator\Desktop\TIA proiect\Poza.png")  # Salvează imaginea pe desktop
    return r"C:\Users\Administrator\Desktop\TIA proiect\Poza.png"


# Funcția pentru a desena pătratul și textul pe overlay
def get_color_for_rarity(rarity):
    """Returnează culoarea corespunzătoare rarității itemului."""
    rarity_colors = {
        'common': 'white',
        'uncommon': 'green',
        'rare': 'blue',
        'epic': 'purple',
        'legendary': 'orange'
    }
    return rarity_colors.get(rarity, 'gray')  # Dacă raritatea nu este cunoscută, folosim gri


def draw_square_and_text_on_overlay(overlay_canvas, predicted_class, confidence):
    """Desenează un pătrat verde și textul pe overlay"""
    # Dimensiunea ecranului
    screen_width, screen_height = pyautogui.size()

    # Calculăm latura pătratului ca procent din dimensiunea minimă a ecranului
    side_length = int(min(screen_width, screen_height) * 0.3)

    # Poziționăm pătratul în centrul ecranului
    x = (screen_width - side_length) * 0.5
    y = (screen_height - side_length) * 0.4

    # Desenăm pătratul
    overlay_canvas.create_rectangle(x, y, x + side_length, y + side_length, outline="green", width=2)

    # Afișăm doar textul dacă confidența este mai mare sau egală cu 0.75
    if confidence >= 0.50:
        text = f"Detected: {predicted_class} ({confidence:.2f})"
        overlay_canvas.create_text(x + side_length // 2, y - 10, text=text.upper(), fill="red", font=('Arial Black', 15, 'bold'))
    else:
        return  # Dacă confidența este sub 0.75, nu mai facem nimic și ieșim din funcție

    # Verificăm dacă mob-ul are loot
    loot = loot_table.get(predicted_class, None)
    if not loot:  # Dacă mob-ul nu are loot
        loot_text = "No loot"
        overlay_canvas.create_text(x + side_length // 2, y + side_length + 20, text=loot_text, fill="black",
                                   font=('Arial Black', 12))
        return

    # Afișăm loot-ul pentru mob în grupuri de câte 2
    loot_texts = []
    for i in range(0, len(loot), 2):  # Împărțim loot-ul în grupuri de câte 2
        loot_group = loot[i:i+2]
        loot_text = '  |  '.join([f"{item['item']} ({item['rarity']})" for item in loot_group])
        loot_texts.append(loot_text)

    # Afișăm fiecare grup de loot pe canvas
    y_offset = y + side_length + 20
    for loot_text in loot_texts:
        # Pentru fiecare item din loot_text, vom extrage raritatea și culoarea corespunzătoare
        items_in_group = loot_text.split('  |  ')  # Împărțim loot_text în iteme individuale

        for item_text in items_in_group:
            item_name, item_rarity = item_text.split(' (')  # Extragem numele itemului și raritatea
            item_rarity = item_rarity.strip(')')  # Eliminăm parantezele

            # Determinăm culoarea corespunzătoare rarității
            item_color = get_color_for_rarity(item_rarity)

            # Adăugăm textul pentru fiecare item cu culoarea corespunzătoare
            overlay_canvas.create_text(x + side_length // 2, y_offset, text=item_text, fill=item_color, font=('Arial Black', 15))
            y_offset += 20  # Mărimăm distanța pentru următorul item




# Configurarea zonei patratului (exemplu la 40% din dimensiunea ecranului)
screen_width, screen_height = pyautogui.size()
side_length = int(min(screen_width, screen_height) * 0.3)
x = int((screen_width - side_length) * 0.5)
y = int((screen_height - side_length) * 0.5)
region = (x, y, side_length, side_length)

# Creăm fereastra de overlay
root = tk.Tk()
root.attributes("-topmost", True)  # Asigurăm că overlay-ul este deasupra
root.attributes("-transparentcolor", "white")  # Setăm fundalul transparent
root.overrideredirect(True)  # Fereastra nu are margini

overlay_canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="white")
overlay_canvas.pack()

# Bucla principală pentru capturarea ecranului și analiza continuă
while True:
    # Capturăm zona de ecran
    img_path = capture_screen(region)

    # Preprocesăm imaginea și facem predicția
    img_array = preprocess_image(img_path)
    predicted_class, confidence = predict_image(img_array)

    # Ștergem conținutul anterior de pe canvas
    overlay_canvas.delete("all")

    # Desenăm patratul și textul pe overlay
    draw_square_and_text_on_overlay(overlay_canvas, predicted_class, confidence)

    # Actualizăm display-ul
    root.update()

    # Așteptăm 1 secunde înainte de următoarea captură
    time.sleep(1)
