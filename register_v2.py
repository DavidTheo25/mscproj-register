import socket
import time
import os
import json
import hashlib
from learn_face import LearnFace
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES, PKCS1_OAEP


def encrypt_request(request, public_key_path):
    encoded_req = request.encode('utf-8')
    public_key = RSA.import_key(open(public_key_path).read())
    session_key = get_random_bytes(16)

    # encrypt session key with public key
    cipher_rsa = PKCS1_OAEP.new(public_key)
    enc_session_key = cipher_rsa.encrypt(session_key)

    # encrypt data with aes session key
    cipher_aes = AES.new(session_key, AES.MODE_EAX)
    cipher_text, tag = cipher_aes.encrypt_and_digest(encoded_req)

    return str((enc_session_key, cipher_aes.nonce, tag, cipher_text)).encode()


def encrypt_file(file_path, public_key_path):
    clear_file = open(file_path).read().encode("utf-8")
    enc_file_path = file_path
    file_out = open(enc_file_path, "wb")
    recipient_key = RSA.import_key(open(public_key_path).read())
    session_key = get_random_bytes(16)

    # Encrypt the session key with the public RSA key
    cipher_rsa = PKCS1_OAEP.new(recipient_key)
    enc_session_key = cipher_rsa.encrypt(session_key)

    # Encrypt the data with the AES session key
    cipher_aes = AES.new(session_key, AES.MODE_EAX)
    ciphertext, tag = cipher_aes.encrypt_and_digest(clear_file)
    [file_out.write(x) for x in (enc_session_key, cipher_aes.nonce, tag, ciphertext)]
    file_out.close()
    return enc_file_path


dataset = "temp/dataset"
proto = "face_detection_model/deploy.prototxt"
model = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
embeddings = "output/embeddings.pickle"
recognizer = "output/recognizer.pickle"
le = "output/le.pickle"
embedding_model = "face_detection_model/openface_nn4.small2.v1.t7"

TCP_IP = 'localhost'
# TCP_IP = "192.168.56.103"  # IP address of the virtual machine
TCP_PORT = 9001
BUFFER_SIZE = 1024

email = input("email address: ")
pswd = input("password: ")
# email = "td37@hw.ac.uk"
# pswd = "toto"
public_key_p = "public1.pem"
email = hashlib.sha256(email.encode()).hexdigest()
pswd = hashlib.sha256(pswd.encode()).hexdigest()
req = "register,%s,%s" % (email, pswd)
req_to_send = encrypt_request(req, public_key_p)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((TCP_IP, TCP_PORT))
    # s.sendall(req.encode('utf-8'))
    s.sendall(req_to_send)
    s.sendall(b"EOR")
    data = s.recv(BUFFER_SIZE)
    data = json.loads(data.decode('utf-8'))
    # print('Received', repr(data))
    if "success" in data:
        if data["success"]:
            print("[INFO] Success !")
            lf = LearnFace(dataset, proto, model, embeddings, recognizer, le, embedding_model)
            zip_name = lf.learn_face(5, email)
            # TODO Encrypt file using the public key
            # enc_zip_name = encrypt_file(zip_name, "public1.pem")
            enc_zip_name = zip_name
            with open(enc_zip_name, "a") as myfile:
                # add token at the end of the file
                myfile.write(data["file_token"])
            # send the zip file
            with open(enc_zip_name, "rb") as myfile:
                buffer = myfile.read(1024)
                counter = 0
                while buffer:
                    counter += 1
                    s.send(buffer)
                    # print('Sent ', repr(buffer))
                    buffer = myfile.read(1024)
                # print("counter = ", counter)
                s.send(b"EOF")
            # remove the zipfile after it has been sent
            os.remove(enc_zip_name)
            # print(enc_zip_name)
            print('[INFO] Face recognition data has been sent to server')
            feedback = s.recv(BUFFER_SIZE * 10)
            feedback = json.loads(feedback.decode("utf-8"))
            # print(repr(feedback))
        else:
            print("[INFO] failed: ", data["reason"])
    else:
        print("[ERROR] This should not ever happen")

# s.close()
print('[INFO] Connection closed')
