import os.path
from flask import Flask, request
from AM.api import speech2pinyin
from LM.api import pinyin2text

app = Flask(__name__)


@app.route("/", methods=['POST'])
def speech2text():
    audio = request.files.get("audio")
    file_path = os.path.join("cache", audio.filename)
    audio.save(file_path)
    text = pinyin2text(speech2pinyin(file_path))
    os.remove(file_path)
    return {"data": text}


@app.route("/hello")
def hello():
    return "hello"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9108, debug=False)
