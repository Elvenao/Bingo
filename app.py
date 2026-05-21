import cv2
import os
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from tkinter import simpledialog
from ultralytics import YOLO
from cnn import CNN_Numeros
import torch
from procesarEntrada import procesar_entrada
from torchvision import transforms as T
import json

IMG_W, IMG_H = 64, 64

def detectar_camaras():
    disponibles = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            disponibles.append(i)
            cap.release()
    return disponibles   

class CameraApp:
    def __init__(self, root, camara_index=0):
        self.root = root
        self.current_number = 0
        self.root.title("Recognize Bingo Cards")
        self.root.geometry("700x900")
        self.languages = ["English","日本五", "Spanish"]
        self.toggle_next = 0 #This select the next language next to the first one which is 0 (English)
        self.root.resizable(False, False)
        self.root.bind("<space>", self.key_capture)
        self.cap = cv2.VideoCapture(camara_index)
        self.output_dir = None
        self.device = self.get_device()
        self.modelo = CNN_Numeros(num_clases=75).to(self.device)
        self.error_label_text = ""
        self.registered_players_text = "0 player(s) registered"
        self.modelo.load_state_dict(
            torch.load("modelo_cnn.pth",
            map_location=self.device,
            weights_only=True)
        )
        self.modelo.eval()
        self.registered_cards_file = "registered_cards.json"
        if not os.path.exists(self.registered_cards_file):
            with open(self.registered_cards_file, "w") as f:
                json.dump([], f)
        else: 
            with open(self.registered_cards_file, "w") as f:
                json.dump([], f)
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.current_error = ""
        self.current_frame = "camera"
        self.dictionary = [
            {
                "title": {
                    "camera":"Bingo Cards Registration",
                    "bingo" : "Bingo Game"
                },
                "language_button": "日本語",
                "start_button": "Start Bingo",
                "bingo_start_button": {
                    0 : "Press to start!",
                    1 : self.current_number},
                "last_text": "Last",
                "next_number_button": "Next Number",
                "registered_players" : " player(s) registered",
                "almost_winners" : {
                    0 : "card(s)",
                    1 : "have",
                    2 : "numbers left\n"

                },
                "error_label_text": {
                    "missing_file": "The required file doesn't exist anymore, try to capture the cards again",
                    "not_enough_players": "The game requires at least 2 players",
                    "no_card_detected": "No card detected, try taking another picture",
                    "": "",
                    "repeated_card" : "Repeated Card!",
                }
            },

            {
                "title": {
                    "camera": "ビンゴカード登録",
                    "bingo": "ビンゴゲーム"
                },
                "language_button": "Español",
                "start_button": "ビンゴ開始",
                "bingo_start_button": {
                    0 : "押して開始",
                    1 : self.current_number
                },
                "last_text": "前回",
                "next_number_button": "次の番号",
                "registered_players" : "人登録済み",
                "almost_winners" : {
                    0 : "枚のカードが",
                    1 : "あと",
                    2 : "個の番号でビンゴです\n"
                },
                "error_label_text": {
                    "missing_file": "必要なファイルが存在しません。もう一度カードを撮影してください",
                    "not_enough_players": "ゲームには最低2人のプレイヤーが必要です",
                    "no_card_detected": "カードが検出されませんでした。もう一度撮影してください",
                    "": "",
                    "repeated_card": "同じカードです！",
                }
            },

            {
                "title": {
                    "camera":"Registro de Cartas de Bingo",
                    "bingo" : "Juego de Bingo"
                },
                "language_button": "English",
                "start_button": "Empezar Bingo",
                "bingo_start_button": {
                    0 : "Presiona para empezar",
                    1 : self.current_number
                },
                "last_text": "Último",
                "next_number_button": "Siguiente Número",
                "registered_players" : " jugador(es) registrados",
                "almost_winners" : {
                    0 : "tarjeta(s)",
                    1 : "tienen",
                    2 : "numero(s) restante(s)"
                },
                "error_label_text": {
                    "missing_file": "El archivo requerido ya no existe, intenta capturar las cartas otra vez",
                    "not_enough_players": "El juego requiere al menos 2 jugadores",
                    "no_card_detected": "No se detectó ninguna carta, intenta tomar otra foto",
                    "": "",
                    "repeated_card" : "¡Carta Repetida!",
                }
            }
        ]

        # UI
        self.camera_frame = tk.Frame(root)
        self.camera_frame.pack()

        self.video_label = tk.Label(self.camera_frame)
        self.video_label.pack()

        self.start_button = tk.Button(
            self.camera_frame,
            text=self.dictionary[self.toggle_next]["start_button"],
            command=self.start_bingo,
            font=("Arial", 20)
        )
        self.start_button.pack(pady=20)

        self.language_button = tk.Button(
            self.camera_frame,
            text=self.dictionary[self.toggle_next]["language_button"],
            command=self.toggle_language,
            font=("Arial", 20)
        )
        self.language_button.pack(pady=20)

        self.error_label = tk.Label(
            self.camera_frame,
            text=self.error_label_text,
            justify="center",
            wraplength=600,
            font=("Arial",18)
        )
        self.error_label.pack(pady=5)

        self.registered_players = tk.Label(
            self.camera_frame,
            text=self.registered_players_text,
            justify="center",
            wraplength=600,
            font=("Arial",18)
        )
        self.registered_players.pack(pady=5)

        # ======================================
        # FRAME BINGO
        # ======================================

        self.bingo_frame = tk.Frame(root)

        self.remaining_label = tk.Label(
            self.bingo_frame,
            text="",
            font=("Arial", 18),
            justify="left",
            fg="blue"
        )

        self.remaining_label.pack(pady=10)
        # Numero actual
        self.number_label = tk.Label(
            self.bingo_frame,
            text=self.dictionary[self.toggle_next]["bingo_start_button"][0],
            wraplength=600,
            font=("Arial", 50, "bold")
        )
        self.number_label.pack(pady=20)

        # Ultimos 5 numeros
        self.last_numbers_label = tk.Label(
            self.bingo_frame,
            text=self.dictionary[self.toggle_next]["last_text"],
            font=("Arial", 18)
        )
        self.last_numbers_label.pack()

        # Historial completo
        self.history_text = tk.Text(
            self.bingo_frame,
            height=10,
            width=50,
            font=("Arial", 14)
        )

        self.history_text.pack(pady=10)

        self.history_text.config(state="disabled")

        # Ganador
        self.winner_label = tk.Label(
            self.bingo_frame,
            text="",
            font=("Arial", 30),
            fg="green"
        )
        self.winner_label.pack(pady=20)

        # Boton siguiente
        self.next_button = tk.Button(
            self.bingo_frame,
            text=self.dictionary[self.toggle_next]["next_number_button"],
            command=self.generate_number,
            font=("Arial", 20)
        )
        self.next_button.pack(pady=10)

        self.language_button2 = tk.Button(
            self.bingo_frame,
            text=self.dictionary[self.toggle_next]["language_button"],
            command=self.toggle_language,
            font=("Arial", 20)
        )
        self.language_button2.pack(pady=20)

        self.available_numbers = list(range(1, 76))
        self.called_numbers = []

        self.update_video()

 
        
    def toggle_language(self):
        if self.toggle_next < len(self.languages) - 1: self.toggle_next += 1
        else: self.toggle_next = 0
        self.root.title(self.dictionary[self.toggle_next]["title"][self.current_frame])
        self.language_button.config(text=self.dictionary[self.toggle_next]["language_button"])
        self.language_button2.config(text=self.dictionary[self.toggle_next]["language_button"])
        self.start_button.config(text=self.dictionary[self.toggle_next]["start_button"])
        self.error_label.config(text=self.dictionary[self.toggle_next]["error_label_text"][self.current_error])
        self.registered_players_text = str(self.get_index() - 1) + self.dictionary[self.toggle_next]["registered_players"]
        self.registered_players.config(text=self.registered_players_text)
        self.next_button.config(text=self.dictionary[self.toggle_next]["next_number_button"])
        if(self.current_number != 0):
            self.number_label.config(text=self.dictionary[self.toggle_next]["bingo_start_button"][1])
        else:
            self.number_label.config(text=self.dictionary[self.toggle_next]["bingo_start_button"][0])
        self.last_numbers_label.config(text=self.dictionary[self.toggle_next]["last_text"])

        self.winner_label.config(text=self.current_number)

    def get_bingo_letter(self, num):
        if 1 <= num <= 15:
            return "B"

        elif 16 <= num <= 30:
            return "I"

        elif 31 <= num <= 45:
            return "N"

        elif 46 <= num <= 60:
            return "G"

        return "O"
    def check_winner(self):
        try:
            with open(self.registered_cards_file, "r") as f:
                cards = json.load(f)

        except:
            return None

        llamados = set(self.called_numbers)

        for card in cards:

            # FREE SPACE centro
            # Tu indice 12 es el centro

            numeros_sin_centro = list(card["numbers"])

            numeros_sin_centro.pop(12)

            numeros_sin_centro = set(numeros_sin_centro)

            # Bingo completo
            if numeros_sin_centro.issubset(llamados):

                return card["card_number"]

        return None
    
    def check_remaining_cards(self):

        try:
            with open(self.registered_cards_file, "r") as f:
                cards = json.load(f)

        except:
            return ""

        llamados = set(self.called_numbers)

        conteo_restantes = {}

        for card in cards:

            numeros = list(card["numbers"])

            # quitar centro FREE
            numeros.pop(12)

            restantes = 0

            for n in numeros:

                if n not in llamados:
                    restantes += 1

            # Solo mostrar cartones con menos de 6
            if restantes <= 5:

                if restantes not in conteo_restantes:
                    conteo_restantes[restantes] = 0

                conteo_restantes[restantes] += 1

        self.texto_restantes = ""

        for restantes in sorted(conteo_restantes):

            cantidad = conteo_restantes[restantes]
            
            self.texto_restantes += (
                f"{cantidad} {self.dictionary[self.toggle_next]['almost_winners'][0]}"
                f"{self.dictionary[self.toggle_next]['almost_winners'][1]}{restantes} {self.dictionary[self.toggle_next]['almost_winners'][2]}"
            )
    def start_bingo(self):
        if not os.path.exists(self.registered_cards_file):
            print("No existe el archivo necesario")
            self.current_error = "missing_file"
            self.error_label.config(text= self.languages[self.toggle_next]["error_label_text"][self.current_error])
            
            return
        else:
            with open(self.registered_cards_file, "r") as f:
                data = json.load(f)
                if len(data) < 2:
                    self.current_error = "not_enough_players"
                    self.error_label.config(text=self.dictionary[self.toggle_next]["error_label_text"][self.current_error])
                    print("El juego debe tener al menos dos participantes")
                    return
        self.camera_frame.pack_forget()

        self.bingo_frame.pack()
        self.current_frame = "bingo"
        self.root.title(self.dictionary[self.toggle_next]["title"][self.current_frame])
        self.cap.release()

    def generate_number(self):

        if len(self.available_numbers) == 0:
            self.number_label.config(text="FIN")
            return

        numero = np.random.choice(self.available_numbers)

        self.available_numbers.remove(numero)

        self.called_numbers.append(numero)

        letra = self.get_bingo_letter(numero)

        # Numero grande actual
        self.number_label.config(
            text=f"{letra} - {numero}"
        )
        self.current_number = f"{letra} - {numero}"
        

        # Ultimos 5
        ultimos = self.called_numbers[-5:]

        texto_ultimos = self.dictionary[self.toggle_next]["last_text"]

        for n in ultimos:
            l = self.get_bingo_letter(n)
            texto_ultimos += f"{l}-{n}  "

        self.last_numbers_label.config(text=texto_ultimos)

        # Historial completo
        self.history_text.config(state="normal")

        self.history_text.delete("1.0", tk.END)

        for i, n in enumerate(self.called_numbers):

            l = self.get_bingo_letter(n)

            self.history_text.insert(
                tk.END,
                f"{l}-{n:02d}  "
            )

            # salto de linea cada 8 numeros
            if (i + 1) % 10 == 0:
                self.history_text.insert(tk.END, "\n")

        self.history_text.config(state="disabled")

        self.check_remaining_cards()

        self.remaining_label.config(text=self.texto_restantes)

        # Revisar ganador
        ganador = self.check_winner()

        if ganador is not None:

            self.winner_label.config(
                text=f"#{ganador} won"
            )

            self.next_button.config(state="disabled")

    
    
    def get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    def get_index(self):
        with open(self.registered_cards_file) as f:
            data = json.load(f)

        return len(data) + 1
    
    def key_capture(self, event):
        self.take_photo()

    def normalizar_para_predict(self, img: Image.Image) -> Image.Image:

        img = img.convert("RGB")
        w, h = img.size
        escala = min(IMG_W / w, IMG_H / h)
        nw, nh = max(1, int(w * escala)), max(1, int(h * escala))
        scaled = img.resize((nw, nh), Image.LANCZOS)
        lienzo = Image.new("RGB", (IMG_W, IMG_H), color=(0, 0, 0))  # padding negro (fondo bolas)
        lienzo.paste(scaled, ((IMG_W - nw) // 2, (IMG_H - nh) // 2))
        return lienzo
    
    def cnn_predict(self, crop):
        img = crop.convert("RGB")

        img_norm = self.normalizar_para_predict(img)

        tensor = self.transform(img_norm).unsqueeze(0).to(self.device)

        logits = self.modelo(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        idx = probs.argmax()

        numero = idx.item() + 1

        probabilidad = probs[idx].item()

        print(f"{numero:02d} ", end="", flush=True)
        #print(f"  {numero:02d} -> {probabilidad:.4f}", end="", flush=True)
        return numero 
    
    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            #frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_video)
    
    def check_repeated_card(self, numbers):
        if os.path.exists(self.registered_cards_file):
            with open(self.registered_cards_file, "r") as f:
               data = json.load(f)
        else:
            with open(self.registered_cards_file, "w") as f:
               json.dump([], f)
            return False

        for c in data:
            if sorted(numbers) == sorted(c["numbers"]):
                return True
        
        return False


    def take_photo(self):
        ret, frame = self.cap.read()
        if ret:
            crops = procesar_entrada(frame)
            #Checks if no card or circle was detected
            if crops == 0:
                self.current_error = "no_card_detected" 
                self.error_label.config(text=self.dictionary[self.toggle_next]["error_label_text"][self.current_error])
                return
            else:
                self.error_label.config(text="")
            i = 0

            if not os.path.exists(self.registered_cards_file):
                with open(self.registered_cards_file, "w") as f:
                    json.dump([], f)

            try:
                with open(self.registered_cards_file, "r") as f:
                    data = json.load(f)
            except:
                data = []

            numbers = []
            for crop in crops:
                
                if i%5==0:
                    print("\n")
                if i == 12:
                    i+=1
                    print("-- ", end="", flush=True)
                numero = self.cnn_predict(crop)
                i += 1
                numbers.append(numero)
            print("\n\n---------------")

            if self.check_repeated_card(numbers):
                self.current_error = "repeated_card" 
                self.error_label.config(text=self.dictionary[self.toggle_next]["error_label_text"][self.current_error])
                return
            else:
                self.current_error = "" 
                self.error_label_text = self.dictionary[self.toggle_next]["error_label_text"][""]
                self.error_label.config(text=self.error_label_text)
            #Guardar numeros
            data.append({
                "card_number" : self.get_index(),
                "numbers" : numbers,
            })
            
            self.registered_players_text = str(self.get_index()) + self.dictionary[self.toggle_next]["registered_players"]
            self.registered_players.config(text=self.registered_players_text)
            with open(self.registered_cards_file, "w") as f:
                json.dump(data,f, indent=4)


    
    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    camaras = detectar_camaras()

    if not camaras:
        import tkinter.messagebox as mb
        mb.showerror("Error", "No se encontró ninguna cámara.")
        root.destroy()
    else:
        if len(camaras) == 1:
            camara_elegida = camaras[0]
        else:
            # Aquí el usuario elige, por ejemplo índice 0, 1, 2...
            import tkinter.simpledialog as sd
            eleccion = sd.askstring(
                "Seleccionar cámara",
                "Cámaras disponibles: " + str(camaras) + "\n\nEscribe el número:",
                initialvalue=str(camaras[0])
            )
            try:
                camara_elegida = int(eleccion)
            except:
                camara_elegida = camaras[0]

        root.deiconify()
        app = CameraApp(root, camara_elegida)
        root.mainloop()