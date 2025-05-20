import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print("✅ MQTT bağlandı:", rc)
    client.subscribe("stream/alerts")

def on_message(client, userdata, msg):
    print(f"📥 Gelen MQTT mesajı: {msg.payload.decode()}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("localhost", 1883, 60)
client.loop_start()

try:
    while True:
        pass
except KeyboardInterrupt:
    print("Çıkış yapılıyor...")
    client.loop_stop()
