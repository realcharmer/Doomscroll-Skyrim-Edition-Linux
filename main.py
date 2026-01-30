import argparse
import cv2
import time
import subprocess
from pathlib import Path
import mediapipe as mp


# MediaPipe Tasks setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
Image = mp.Image
ImageFormat = mp.ImageFormat


# Video player (mpv)
class LinuxVideoPlayer:
    def __init__(self):
        self.process = None

    def play(self, video_path: Path) -> None:
        if self.process is not None:
            return

        self.process = subprocess.Popen(
            [
                "mpv",
                "--ontop",
                "--no-border",
                "--geometry=360x780+25+45",
                "--loop",
                str(video_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def close(self, video_path: Path) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process = None


# Draw warning overlay
def draw_warning(frame, text="lock in twin"):
    h, w = frame.shape[:2]
    box_w, box_h = 500, 70
    x1 = (w - box_w) // 2
    y1 = 24
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 0, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (80, 255, 160), 4)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 160), 2)

    cv2.putText(
        frame,
        text.upper(),
        (x1 + 26, y1 + 48),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )


# Main
def main(timer, looking_threshold, debounce_threshold):
    skyrim_video = Path("./assets/skyrim-skeleton.mp4").resolve()
    model_file = Path("./assets/face_landmarker_full.task").resolve()

    if not skyrim_video.exists() or not model_file.exists():
        print("Missing skyrim-skeleton.mp4 or face_landmarker_full.task in ./assets/")
        return

    video_player = LinuxVideoPlayer()

    # Initialize FaceLandmarker
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_file)),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    face_landmarker = FaceLandmarker.create_from_options(options)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open webcam")
        return

    doomscroll = None
    video_playing = False

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # Convert to MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)

        result = face_landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        if result.face_landmarks and len(result.face_landmarks) > 0:
            lm = result.face_landmarks[0]

            left = [lm[145], lm[159]]
            right = [lm[374], lm[386]]

            lx = int((left[0].x + left[1].x) / 2 * width)
            ly = int((left[0].y + left[1].y) / 2 * height)
            rx = int((right[0].x + right[1].x) / 2 * width)
            ry = int((right[0].y + right[1].y) / 2 * height)

            box = 50
            cv2.rectangle(
                frame, (lx - box, ly - box), (lx + box, ly + box), (10, 255, 0), 2
            )
            cv2.rectangle(
                frame, (rx - box, ry - box), (rx + box, ry + box), (10, 255, 0), 2
            )

            l_iris = lm[468]
            r_iris = lm[473]

            l_ratio = (l_iris.y - left[1].y) / (left[0].y - left[1].y + 1e-6)
            r_ratio = (r_iris.y - right[1].y) / (right[0].y - right[1].y + 1e-6)
            avg_ratio = (l_ratio + r_ratio) / 2.0

            if video_playing:
                is_looking_down = avg_ratio < debounce_threshold
            else:
                is_looking_down = avg_ratio < looking_threshold

            if is_looking_down:
                if doomscroll is None:
                    doomscroll = time.time()
                if (time.time() - doomscroll) >= timer:
                    if not video_playing:
                        video_player.play(skyrim_video)
                        video_playing = True
            else:
                doomscroll = None
                if video_playing:
                    video_player.close(skyrim_video)
                    video_playing = False
        else:
            doomscroll = None
            if video_playing:
                video_player.close(skyrim_video)
                video_playing = False

        if video_playing:
            draw_warning(frame, "doomscrolling alarm")

        cv2.imshow("lock in", frame)
        if cv2.waitKey(1) == 27:
            break

    if video_playing:
        video_player.close(skyrim_video)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doomscrolling alarm with MediaPipe.")
    parser.add_argument("--timer", type=float, default=2.0, help="Seconds before video plays when looking down")
    parser.add_argument("--looking_threshold", type=float, default=0.25, help="Threshold for initial look-down detection")
    parser.add_argument("--debounce_threshold", type=float, default=0.45, help="Threshold for continuing look-down detection when video is playing")

    args = parser.parse_args()
    main(args.timer, args.looking_threshold, args.debounce_threshold)
