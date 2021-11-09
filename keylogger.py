import asyncio
import enum
import json
from pynput.keyboard import Key, KeyCode, Listener
from slice_audio import *
from time import time_ns

class EventType(enum.Enum):
    """ Returns the type of the keystroke event currently being captured. """

    KEY_DOWN = 0
    KEY_UP = 1

    def __repr__(self):
        if self == EventType.KEY_DOWN:
            return 'KEY_DOWN'
        elif self == EventType.KEY_UP:
            return 'KEY_UP'

    def __str__(self):
        return repr(self)

class Event:

    def __init__(self, event_type, char, delta_time):
        self.event_type = event_type
        self.char = char
        self.delta_time = delta_time

    def __repr__(self):
        return f'type: {self.event_type}, char: {self.char}, delta_time: {self.delta_time}'

    @staticmethod
    def from_json(json):
        j = json.loads(json)
        return Event(EventType.from_json(j['type']), j['char'], j['delta_time'])

    def to_dict(self):
        return {'event_type': str(self.event_type),
                'char': self.char,
                'delta_time': self.delta_time}

class EventCollection:

    def __init__(self):
        self.events = []

    def __repr__(self):
        return str(self.events)

    def to_dict(self, dot_dict):
        if dot_dict:
            return [DotDict(event.to_dict()) for event in self.events]
        else:
            return [event.to_dict() for event in self.events]

class KeyloggerSession:
    """ Creates a new keylogger session for recording keystrokes. """

    def __init__(self):
        self.event_collection = EventCollection()
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

    async def get_start_time(self):
        self.start_time_unix = time_ns()

    async def start_logging(self):
        print("\nStarted Logging. Press F7 to stop logging.")
        await self.get_start_time()
        self.listener.start()
        self.listener.join()

    def stop_logging(self):
        print("\nStopped Keyboard Logging.")
        self.listener.stop()
        self.end_time_unix = time_ns()

    def on_press(self, key):
        """ Capture KEY_DOWN events. """
        if key == Key.f7:
            self.stop_logging()
        elif str(key) != '<63>':
            event = Event(EventType.KEY_DOWN, str(key), time_ns() - self.start_time_unix)
            self.log_event(event)

    def on_release(self, key):
        """ Capture KEY_UP events. """
        if str(key) != '<63>':
            event = Event(EventType.KEY_UP, str(key), time_ns() - self.start_time_unix)
            self.log_event(event)

    def log_event(self, event):
        self.event_collection.events.append(event)

    def json_to_file(self, out_f, fmode='w'):
        f = open(out_f, fmode)
        f.write(self.to_json())
        f.close()

    def to_json(self):
        """ Write keylogger session as a string in JSON format. """
        return json.dumps({"start_time_unix" : self.start_time_unix,
                           "end_time_unix" : self.end_time_unix,
                           "events" : self.event_collection.to_dict(dot_dict=False)},
                           indent = 4)

    def import_json(self, out_f, dot_dict=True):
        j = json.load(open(out_f))
        


        if dot_dict:
            j['events'] = [DotDict(event) for event in j['events']]
            return DotDict(j)
        return d
        

    def to_dict(self, dot_dict=True):
        d = {
             'start_time_unix' : self.start_time_unix,
             'end_time_unix' : self.end_time_unix,
             'events' : self.event_collection.to_dict(dot_dict)
            }
        if dot_dict:
            return DotDict(d)
