import requests

m_files = [
    ('imageR', ('R_3277.jpg', open('R_3277.jpg', 'rb'), 'image/jpg')),
    ('imageO', ('O_1815.jpg', open('O_1815.jpg', 'rb'), 'image/jpg')),
]

response = requests.post('http://0.0.0.0:8080/send_image', files=m_files)

print(response.text)
