from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vips_data_analysis.actuator_bend import process_video


VIDEO_PATH = REPO_ROOT / "data" / "04-15_Actuator_Test.mp4"
OUTPUT_DIR = REPO_ROOT / "analysis" / "actuator_bend"


def main() -> None:
    process_video(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        show_preview=True,
        max_frames=None,
        smoothing_window=7,
    )


if __name__ == "__main__":
    main()
