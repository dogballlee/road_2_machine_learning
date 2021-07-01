# -*- coding: utf-8 -*-
import ctypes
import time


SendInput = ctypes.windll.user32.SendInput

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

M = 0x32
J = 0x24
K = 0x25
LSHIFT = 0x2A
R = 0x13  # 用R代替识破
V = 0x2F

Q = 0x10
U = 0x16
I = 0x17
O = 0x18
P = 0x19
C = 0x2E
F = 0x21

up = 0xC8
down = 0xD0
left = 0xCB
right = 0xCD

esc = 0x01

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def attack():
    PressKey(U)
    time.sleep(0.05)
    ReleaseKey(U)
    # time.sleep(0.1)


def go_left():
    PressKey(A)
    time.sleep(0.4)
    ReleaseKey(A)


def go_right():
    PressKey(D)
    time.sleep(0.4)
    ReleaseKey(D)


def jump():
    PressKey(I)
    time.sleep(0.1)
    ReleaseKey(I)
    # time.sleep(0.1)


def dodge():
    PressKey(S)
    time.sleep(0.1)
    ReleaseKey(S)
    # time.sleep(0.1)


def press_esc():
    PressKey(esc)
    time.sleep(0.3)
    ReleaseKey(esc)


if __name__ == '__main__':
    time.sleep(5)
    time1 = time.time()
    while (True):
        if abs(time.time() - time1) > 5:
            break
        else:
            PressKey(U)
            time.sleep(0.1)
            ReleaseKey(U)
            time.sleep(0.2)

    PressKey(D)
    time.sleep(0.4)
    ReleaseKey(D)
    time.sleep(1)

    PressKey(I)
    time.sleep(0.1)
    ReleaseKey(I)
    time.sleep(1)


