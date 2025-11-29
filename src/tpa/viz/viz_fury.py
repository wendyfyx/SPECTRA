

def get_cam_settings_fury(cam_name='Axial'):
    EYE_SCALE=2
    CAM_SETTINGS = {
        'Axial': {'view_up': (0,1,0), 'focal_point': (0,0,0), 'position': (0,0,EYE_SCALE) },
        'Sagittal_L': {'view_up': (0,0,1), 'focal_point': (0,0,0),'position': (-1*EYE_SCALE, 0, 0) },
        'Sagittal_R': {'view_up': (0,0,1), 'focal_point': (0,0,0),'position': (EYE_SCALE, 0, 0) },
        'Coronal_A': {'view_up': (0,0,1), 'focal_point': (0,0,0), 'position': (0, EYE_SCALE, 0)},
        'Coronal_P': {'view_up': (0,0,1), 'focal_point': (0,0,0), 'position': (0, -1*EYE_SCALE, 0)}
        }
    return CAM_SETTINGS[cam_name]