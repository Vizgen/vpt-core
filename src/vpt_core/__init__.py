import os
from pathlib import Path

import numpy as np
import dotenv

if not hasattr(np, "float_"):
    np.float_ = np.float64

AWS_PROFILE_NAME_VAR = "VPT_AWS_PROFILE"
AWS_ACCESS_KEY_VAR = "VPT_AWS_ACCESS_KEY_ID"
AWS_SECRET_KEY_VAR = "VPT_AWS_SECRET_ACCESS_KEY"
GCS_SERVICE_ACCOUNT_KEY_VAR = "VPT_GCS_SERVICE_ACCOUNT_KEY"

envPath = os.path.join(Path.home(), ".vptenv")

if os.path.exists(envPath):
    dotenv.load_dotenv(envPath)
