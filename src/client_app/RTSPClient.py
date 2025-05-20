import cv2

rtsp_url = "rtsp://127.0.0.1:8554/mystream"

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ RTSP yayınına bağlanılamadı!")
    exit()

print("✅ Yayına bağlandı, gösteriliyor...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Yayından frame alınamıyor.")
        break

    cv2.imshow("RTSP Yayını", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
