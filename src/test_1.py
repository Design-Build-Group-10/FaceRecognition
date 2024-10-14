from chroma_client import face_collection
import numpy as np
import asyncio
import websockets
import cv2

from face_analysis import FaceAnalysis


faceAnalysis = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
faceAnalysis.prepare(ctx_id=0, det_size=(640, 640))


def process_frame(frame):
    original_frame = frame.copy()
    faces = faceAnalysis.get(frame)

    key_points_image = np.ones_like(frame) * 255
    key_points_image = faceAnalysis.draw_on(key_points_image, faces)

    processed_faces = []
    unknown_faces = []
    unknown_embeddings = []

    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.normed_embedding.tolist()
        min_dist = float('inf')
        identity = 'unknown'
        confidence = 0

        gender = 'Male' if face.gender == 1 else 'Female'
        age = face.age

        results = face_collection.query(query_embeddings=[embedding], n_results=1)
        if results['ids'] and results['ids'][0]:
            nearest_id = results['ids'][0][0]
            min_dist = results['distances'][0][0]
            if min_dist < 1.0:
                identity = nearest_id
                confidence = min(1 - min_dist, 1)

        color = (0, 0, 255)
        if confidence > 0.5:
            color = (0, 255, 0)
        elif confidence > 0.3:
            color = (0, 255, 255)

        # 收集处理后的信息
        face_data = {
            'identity': identity,
            'confidence': confidence,
            'gender': gender,
            'age': int(age),
            'embedding': embedding
        }
        processed_faces.append(face_data)

        if identity == 'unknown':
            face_image = original_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            unknown_faces.append(face_image)
            unknown_embeddings.append(embedding)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, f'ID: {identity} ({confidence:.2f})', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    color, 2)
        cv2.putText(frame, f'Gender: {gender}, Age: {int(age)}', (bbox[0], bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2)

    return {
        'frame': frame,
        'key_points_image': key_points_image,
        'processed_faces': processed_faces,
        'unknown_faces': unknown_faces,
        'unknown_embeddings': unknown_embeddings
    }


async def send_image():
    # 连接到 WebSocket 服务
    uri = "wss://service.design-build.site/api/ws/transmit/SR-2024X-7B4D-QP98"

    # 打开本地摄像头（设备ID通常是0，表示默认的摄像头）
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    async with websockets.connect(uri) as websocket:
        a = 0

        while True:
            # 从摄像头读取帧
            ret, frame = cap.read()

            if not ret:
                print("无法从摄像头读取帧")
                break

            result = process_frame(frame)

            processed_faces = result['frame']

            # 将帧编码为JPEG格式的二进制数据
            ret, buffer = cv2.imencode('.jpg', processed_faces)

            if not ret:
                print("无法编码图像")
                break

            # 将图像的二进制数据发送到 WebSocket 服务
            await websocket.send(buffer.tobytes())
            # print("Image from camera sent successfully.")

            # 接收服务器返回的数据
            response = await websocket.recv()
            print(a)
            a += 1
            # print(response)

    # 释放摄像头资源
    cap.release()

# 运行异步任务
asyncio.get_event_loop().run_until_complete(send_image())
