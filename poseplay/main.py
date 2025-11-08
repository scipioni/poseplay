"""Main entry point for poseplay application."""

import sys

import cv2

from .config import parse_args, Config
from .image_grabber import ImageGrabberFactory, RTSPGrabber


def display_frame(frame, window_name: str = "PosePlay"):
    """Display frame in OpenCV window."""
    cv2.imshow(window_name, frame)
    return cv2.waitKey(1) & 0xFF


def grab_and_display_loop(config: Config):
    """Main loop for grabbing and displaying frames."""
    try:
        grabber = ImageGrabberFactory.create(
            config.source, **config.to_grabber_kwargs()
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    paused = False
    running = True

    print(f"Starting image grabber for: {config.source}")
    print(
        "Controls: 'q' or ESC to quit"
        + (
            ", 'p' or SPACE to pause/resume, 'r' to reset"
            if not isinstance(grabber, RTSPGrabber)
            else ""
        )
    )

    try:
        while running:
            if not paused:
                frame = grabber.get_frame()
                if frame is None:
                    if config.loop and not isinstance(grabber, RTSPGrabber):
                        print("Reached end of source, restarting...")
                        grabber.close()
                        try:
                            grabber = ImageGrabberFactory.create(
                                config.source, **config.to_grabber_kwargs()
                            )
                            continue
                        except ValueError as e:
                            print(f"Restart failed: {e}")
                            break
                    else:
                        print("Reached end of source")
                        break

                key = display_frame(frame)
                if key == ord("q") or key == 27:  # q or ESC
                    break
                elif key == ord("p") or key == 32:  # p or space
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord("r") and not isinstance(grabber, RTSPGrabber):
                    grabber.close()
                    try:
                        grabber = ImageGrabberFactory.create(
                            config.source, **config.to_grabber_kwargs()
                        )
                        print("Reset grabber")
                    except ValueError as e:
                        print(f"Reset failed: {e}")
                        break
            else:
                # When paused, still check for keyboard input
                key = cv2.waitKey(100) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("p") or key == 32:
                    paused = not paused
                    print("Paused" if paused else "Resumed")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        grabber.close()
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    try:
        config = parse_args()
        grab_and_display_loop(config)
    except SystemExit:
        pass  # argparse handles help and errors
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
