# Neon Hand Yoyo Effect

A real-time, interactive hand-tracking yoyo/particle effect using OpenCV and MediaPipe.  
Features glowing comet orbs, neon edge outlines, and dynamic group physics—all controlled by your hand gestures!

![Demo gif](Images/magic-hands.gif)

---

## Features

- Real-time hand tracking (MediaPipe)
- Neon edge outlines of all objects (Canny edge detection)
- Glowing comet orbs that detach and return with physics
- Group catch/release dynamics (all orbs move together)
- Smooth transitions and visual effects

---

## Requirements

- Python 3.8+
- opencv-python
- mediapipe
- numpy

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python3 yoyo.py
```

- Press `q` to quit.
- Set `FANCY_MODE = True` at the top of `yoyo.py` for the neon/edge effect, or `False` for the regular camera view.

---

## Controls & Gestures

- **Throw:** Open your hand quickly to fling the orbs away.
- **Catch:** Bring your fingers together to zip all orbs back and merge into a single glowing red ball.
- **Orbs:** Neon blue when floating, neon red when caught.


*Note: Works best when hand is 2-3 feet from camera.*

---

## Customization

- Tweak `FANCY_MODE`, colors, and physics parameters in `yoyo.py` for different effects.
- Try different hand gestures for creative results!

---

## Credits

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- Inspired by creative coding and AR/vision art.

---

## License

MIT License

---

*Made with ❤️ and computer vision magic!* 
