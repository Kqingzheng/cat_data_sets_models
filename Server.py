
import json
import six
from socket import *

from keras.preprocessing import image
from tensorflow.python.keras.models import load_model

def predict_cat():
    path = "./whichcat.jpeg"
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img) / 255.
    x = x.reshape((1,) + x.shape)
    model = load_model('model1.hdf5')
    xx = model.predict_classes(x)
    print(str(xx[0]+1))
    return str(xx[0]+1)


if __name__ == '__main__':
    # 1 定义域名和端口号
    HOST, POST = '', 6666
    # 2 定义缓冲区大小 缓存输入或输出的大小，为了解决速度不匹配的问题
    BUFFER_SIZE = 4096
    ADDR = (HOST, POST)
    result_temp = 66
    # 3 创建服务器的套接字 AF_INET:IPV4 SOCK_STREAM:协议
    tcpServerSocket = socket(AF_INET, SOCK_STREAM)
    # 4 绑定域名和端口号
    tcpServerSocket.bind(ADDR)
    # 5 监听连接，最大连接数一般默认5，如果服务器高并发则增大
    tcpServerSocket.listen(5)  # 被动等待连接
    print('服务器创建成功，等待客户端连接。。。。。')
    while True:
        # 6.1 打开一个客户端对象 同意你连接
        tcpCilentSocket, addr = tcpServerSocket.accept()
        print('连接服务器的客户端对象', addr)
        # 6.2循环过程
        file = open("d:/MyDownloads/cat_data_sets_models/whichcat.jpeg", "wb")
        while True:
            # 6.3拿到数据recv()从缓冲区读取指定长度的数据
            # decode(）解码bytes——>str  encode()——>编码 str——>bytes
            data = tcpCilentSocket.recv(BUFFER_SIZE)
            data0 = str(data)[-4:-1]
            if data0 =="END":
                data = data[:len(data)-3]
                file.write(data)
                file.flush()
                file.close()
                print("正在识别。。。。。")
                result = predict_cat()
                tcpCilentSocket.send(result.encode())
                print("识别完成")
                break
            file.write(data)
            file.flush()

            # print('data=', data)
            # result = predict(str(data))

            # 6.4 发送时间还有信息
        # 7 关闭资源

        tcpCilentSocket.close()
    tcpServerSocket.close()