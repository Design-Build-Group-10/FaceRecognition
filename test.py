import asyncio
import websockets
import cv2


async def send_image():
    # 连接到 WebSocket 服务
    uri = "ws://62.234.168.154/api/ws/camera/SR-2024X-7B4D-QP98"

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

            # 将帧编码为JPEG格式的二进制数据
            ret, buffer = cv2.imencode('.jpg', frame)

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
