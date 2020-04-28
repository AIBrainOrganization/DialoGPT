from flask import Flask, request
from bot import Bot
from config import vocab_path, model_path, reverse_model_path

import json


bot = Bot(vocab_path, model_path, reverse_model_path)
emotion_map = {10001: 'Happiness', 10002: 'Anger', 10003: 'Disgust',
               10004: 'Fear', 10005: 'Neutral', 10006: 'Sadness',
               10007: 'Surprise'}


app = Flask(__name__)


@app.route('/', methods=['GET'])
def server():
  utext = request.args.get('utext')
  emotion = request.args.get('emotion')
  print(utext)
  uid = request.args.get('uid')
  ret_text, ret_emotion = bot.reply(utext, emotion_map[int(emotion)], uid)
  for key, emotion in emotion_map.items():
    if emotion == ret_emotion:
      ret_emotion = key
      break
  ret = (ret_text, ret_emotion)
  return json.dumps({'text': ret[0], 'emotion': ret[1]},
                    ensure_ascii=False).encode('utf8')


if __name__ == '__main__':
  app.run(host="0.0.0.0")
