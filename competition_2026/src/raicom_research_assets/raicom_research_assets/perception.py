from ament_index_python.packages import get_package_share_path
from yellow_track_avoidance import AvoidancePipeline, load_config

VIEWS = ('bev', 'yellow_mask', 'raw_overlay', 'disc_overlay', 'plan')


def competition_profile():
    share = get_package_share_path('raicom_research_assets')
    return share / 'config/vision/region4_competition_20260819.yaml'


class TrackAnalysis:
    def __init__(self, profile):
        self.config = load_config(str(profile))
        self.config['control']['enable_motion'] = False
        self.pipeline = AvoidancePipeline(self.config)

    def analyze(self, image, stamp):
        height, width = image.shape[:2]
        cam = self.config['camera']
        if (width, height) != (cam['expected_width'], cam['expected_height']):
            raise ValueError(f'Image is {width}x{height}. This calibration expects '
                             f'{cam["expected_width"]}x{cam["expected_height"]}.')
        result = self.pipeline.process(image, stamp, debug_image_names=VIEWS, run_planning=True)
        stats = {
            'calibrated': bool(result.diagnostics['calibrated']),
            'boundary_quality': float(result.diagnostics.get('boundary_quality', 0)),
            'left_boundary': bool(result.diagnostics.get('left_detected', False)),
            'right_boundary': bool(result.diagnostics.get('right_detected', False)),
            'detected_discs': len(result.raw_disks),
            'tracked_discs': len(result.tracked_disks),
            'selected_path': result.diagnostics.get('selected_path'),
        }
        return result.debug_images, stats
