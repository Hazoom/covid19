from sqlalchemy import Column, Integer, String

from db import Base


class Sentence(Base):
    """Sentence model (text + metadata).

    Usage example
    -------------

    session = Session()
    sents = session.query(Sentence).filter(Sentence.id.in_([0, 1])).all()
    print(sents[0])
    session.close()
    """
    __tablename__ = 'sentences'

    id = Column(Integer, name='sentence_id', primary_key=True)
    sentence = Column(String)
    paper_id = Column(String, name='paper_id')
    cord_uid = Column(String, name='cord_uid')
    publish_time = Column(String, name='publish_time')

    def __repr__(self):
        return f"<Sentence(id={self.id}, sentence=\"{self.sentence}\")>"

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.sentence,
            'paper_id': self.paper_id,
            'cord_uid': self.cord_uid,
            'publish_time': self.publish_time
        }
