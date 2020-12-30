from interact import load, append_messages, generate_message


class Bot:
  def __init__(self, vocab_path, model_path, reverse_model_path):
    self.vocab, self.model, self.reverse_model, self.end_token = load(
        vocab_path, model_path, reverse_model_path)
    self.messages = {}

  def reply(self, message, emotion, id):
    if id not in self.messages:
      self.messages[id] = []
    append_messages(self.messages[id], [message], self.vocab, self.end_token)
    response = generate_message(
        self.messages[id], self.model, self.reverse_model, self.vocab, False)
    append_messages(self.messages[id], [response], self.vocab, self.end_token)

    return response, 'Neutral'
