import cv2
import paho.mqtt.client as mqtt
import threading

# ✅ MQTT mesajı için global değişken
current_alert = ""

# MQTT callback fonksiyonları
def on_connect(client, userdata, flags, rc):
    print("✅ MQTT bağlı:", rc)
    client.subscribe("stream/alerts")

def on_message(client, userdata, msg):
    global current_alert
    current_alert = msg.payload.decode()
    print("📥 MQTT mesajı:", current_alert)

# MQTT bağlantısı (ayrı thread'te)
def start_mqtt():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("localhost", 1883, 60)
    client.loop_forever()

mqtt_thread = threading.Thread(target=start_mqtt)
mqtt_thread.daemon = True
mqtt_thread.start()

# ✅ RTSP yayını başlat
rtsp_url = "rtsp://127.0.0.1:8554/mystream"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ Yayına bağlanamadı!")
    exit()

print("🎥 Yayın başladı...")

# Ana döngü
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame alınamadı.")
        break

    # ❗ Uyarı metnini üst köşeye yaz
    if current_alert:
        cv2.putText(frame, f"ALERT: {current_alert}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    # Görüntüyü göster
    cv2.imshow("RTSP Yayını + MQTT", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
