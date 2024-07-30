import cv2
import numpy as np

def load_image(file_path):
    return cv2.imread(file_path)

def load_video(file_path):
    video_capture = cv2.VideoCapture(file_path)
    frames = []

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)

    video_capture.release()
    return frames

def edge_detection(frame, parametre1, parametre2): 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, parametre1, parametre2)
    return edges

def corner_detection(frame): 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
    return frame

def segmentation(frame): #K-means kümeleme
    reshaped_frame = frame.reshape((-1, 3))
    reshaped_frame = np.float32(reshaped_frame)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 8 
    _, labels, centers = cv2.kmeans(reshaped_frame, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    segmented_frame = centers[labels.flatten()]
    segmented_frame = segmented_frame.reshape(frame.shape)
    return segmented_frame

def dehazing(frame): #Sis giderme için Dark Channel Prior algoritmasını kullanma
    dark_channel = np.min(frame, axis=2)
    atmospheric_light = np.percentile(dark_channel, 95)
    transmission = 1 - 0.95 * (dark_channel / atmospheric_light)
    transmission = np.clip(transmission, 0.1, 1.0)
    dehazed_frame = np.zeros(frame.shape, dtype=np.uint8)
    
    for channel in range(3):
        dehazed_channel = (frame[:, :, channel] - atmospheric_light) / transmission + atmospheric_light
        dehazed_frame[:, :, channel] = np.clip(dehazed_channel, 0, 255)
    return dehazed_frame

def main(): #Kullanıcıya bir menü sunulur ve seçimine göre ilgili fonksiyon çağırılır
    while True:
        print('------------------------')
        print("1. Fotoğraf yükle ve kenarları tespit et")
        print("2. Fotoğraf yükle ve köşeleri tespit et")
        print("3. Fotoğraf yükle ve segmentasyon yap")
        print("4. Fotoğraf yükle ve sis gider")
        print("5. Video yükle ve kenarları tespit et")
        print("6. Video yükle ve köşeleri tespit et")
        print("7. Video yükle ve segmentasyon yap")
        print("8. Video yükle ve sis gider")
        print("9. Çıkış")
        print('------------------------')

        choice = input("Seçiminizi yapın (1-9): ")

        if choice == '1':
            image_file = input("Fotoğraf dosyasının yolunu girin: ")
            image = load_image(image_file)
            cv2.imshow("Original Image", image)
            
            param1 = int(input("Canny parametresi 1 girin: "))
            param2 = int(input("Canny parametresi 2 girin: "))
            edges = edge_detection(image, param1, param2)
            cv2.imshow("Edges", edges)
            
            cv2.waitKey(0)
            break
            cv2.destroyAllWindows()

        elif choice == '2':
            image_file = input("Fotoğraf dosyasının yolunu girin: ")
            image = load_image(image_file)
            cv2.imshow("Original Image", image)
            
            corners = corner_detection(image)
            cv2.imshow("Corners", corners)
            
            cv2.waitKey(0)
            break
            cv2.destroyAllWindows()

        elif choice == '3':
            image_file = input("Fotoğraf dosyasının yolunu girin: ")
            image = load_image(image_file)
            cv2.imshow("Original Image", image)
            
            segmented_frame = segmentation(image)
            cv2.imshow("Segmented Frame", segmented_frame)
           
            cv2.waitKey(0)
            break
            cv2.destroyAllWindows()

        elif choice == '4':
            image_file = input("Fotoğraf dosyasının yolunu girin: ")
            image = load_image(image_file)
            cv2.imshow("Original Image", image)
            
            dehazed_frame = dehazing(image)
            cv2.imshow("Dehazed Frame", dehazed_frame)
            
            cv2.waitKey(0)
            break
            cv2.destroyAllWindows()

        elif choice == '5':
            param1 = int(input("Canny parametresi 1 girin: "))
            param2 = int(input("Canny parametresi 2 girin: "))
            video_file = input("Video dosyasının yolunu girin: ")
            frames = load_video(video_file)
            for frame in frames:
                edges = edge_detection(frame, param1, param2)
                cv2.imshow("Edges", edges)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()

        elif choice == '6':
            video_file = input("Video dosyasının yolunu girin: ")
            frames = load_video(video_file)
            for frame in frames:
                corners = corner_detection(frame)
                cv2.imshow("Corners", corners)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()

        elif choice == '7':
            video_file = input("Video dosyasının yolunu girin: ")
            frames = load_video(video_file)
            for frame in frames:
                segmented_frame = segmentation(frame)
                cv2.imshow("Segmented Frame", segmented_frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()

        elif choice == '8':
            video_file = input("Video dosyasının yolunu girin: ")
            frames = load_video(video_file)
            for frame in frames:
                dehazed_frame = dehazing(frame)
                cv2.imshow("Dehazed Frame", dehazed_frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()

        elif choice == '9':
            break

        else:
            print("Geçersiz seçim.")

if __name__ == "__main__":
    main()