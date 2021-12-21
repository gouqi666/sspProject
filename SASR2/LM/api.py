from .trainer import LMTrainer

trainer = LMTrainer()


def pinyin2text(pinyin):
    return trainer.predict(pinyin)
