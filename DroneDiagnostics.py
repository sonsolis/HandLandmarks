from djitellopy import tello
import KeyPressModule as kp

kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())

def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if kp.getKey("LEFT"): lr = -speed
    elif kp.getKey("RIGHT"): lr = speed
    if kp.getKey("UP"): fb = speed
    elif kp.getKey("DOWN"): fb = -speed

    if kp.getKey("w"): ud = speed
    elif kp.getKey("s"): ud = -speed

    if kp.getKey("a"): yv = speed
    elif kp.getKey("d"): yv = -speed

    if kp.getKey("q"): me.land()
    elif kp.getKey("e"): me.takeoff()

    print(f"Current Speed: {speed}")

#    if kp.getKey('p'):
#        print("Image Captured")
#        cv2.imwrite(f'./Resources/Images/{time.time()}.jpg', img)
#        time.sleep(0.3)

    return [lr, fb, ud, yv]

while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    
    if kp.getKey("h"): me.emergency()

    # print(f"Is flying: {me.is_flying}")
    print(f"UDP Object: {me.get_own_udp_object()}")

