import cv2
import os

def extract_frames(
    video_path,
    output_folder,
    target_fps=5,
    resize_width=None
):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(round(video_fps / target_fps)), 1)

    frame_idx = 0
    saved_idx = 0

    print(f"Video FPS: {video_fps:.2f}")
    print(f"Saving ~{target_fps} FPS (every {frame_interval} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if resize_width is not None:
                h, w = frame.shape[:2]
                scale = resize_width / w
                frame = cv2.resize(
                    frame,
                    (resize_width, int(h * scale)),
                    interpolation=cv2.INTER_AREA
                )

            filename = os.path.join(
                output_folder, f"frame_{saved_idx:06d}.png"
            )
            cv2.imwrite(filename, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved_idx} frames to {output_folder}")


if __name__ == "__main__":
    extract_frames(
        video_path="needle_train.mkv",
        output_folder="data/label_frames",
        target_fps=5,
        resize_width=960
    )
