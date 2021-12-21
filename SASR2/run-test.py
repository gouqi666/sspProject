import requests

print(requests.post('http://localhost:5001/text', data={"pinyin": "['nǐ', 'hǎo']"}).json())
