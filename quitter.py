from pynput.keyboard import Key, Listener
from pynput import keyboard
from time import sleep as wait

class Quitter:
    def __init__(self):
        self.listener = keyboard.Listener(
            on_release=self.on_release)
        self.listener.start()

    def on_release(self, key):
        wait(0.1)
        print('\nQuitting... ')
        if key == Key.esc:
            # Stop listener
            return False



if __name__ == "__main__":
    listener = Quitter()
    while listener.listener.running:
        print("Running")
        pass