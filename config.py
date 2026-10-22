"""Central configuration. Override with environment variables if needed."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# UPLOADED-EHR database: the only patient database the running application reads or writes. It starts empty and is
# filled only by CSV imports. Created on first start.
UPLOADED_EHR_DB_PATH = os.environ.get("EHR_UPLOAD_DB_PATH", os.path.join(BASE_DIR, "data", "uploaded_ehr.db"))

# MANUFACTURER-UPLOADED EHR database: same schema as the clinician one, separate file, filled only by manufacturer CSV uploads.
MFR_UPLOAD_DB_PATH = os.environ.get("MFR_UPLOAD_DB_PATH", os.path.join(BASE_DIR, "data", "manufacturer_ehr.db"))

# TRAINING-ERA databases. Used only by the offline build/training scripts in scripts/; the running application never
# opens them. Kept untouched because they are the data the saved model was trained on.
DB_PATH = os.environ.get("EHR_DB_PATH", os.path.join(BASE_DIR, "data", "ehr.db"))
ML_DB_PATH = os.environ.get("EHR_ML_DB_PATH", os.path.join(BASE_DIR, "data", "v4_2", "ehr.db"))

# Source datasets (original files, untouched)
RAW_DIR = os.environ.get("EHR_RAW_DIR", os.path.join(BASE_DIR, "data", "raw"))
SCHEMA_PATH = os.path.join(BASE_DIR, "app", "schema.sql")

SOURCE_FILES = {
    "patients": os.path.join(RAW_DIR, "dataset1_Patients.csv"),
    "medication_history": os.path.join(RAW_DIR, "dataset2_Medication_History.xlsx"),
    "cognitive_behavioral": os.path.join(RAW_DIR, "dataset3_Cognitive_Behavioral.xlsx"),
    "clinical_events": os.path.join(RAW_DIR, "dataset4_Clinical_Events.xlsx"),
    "drug_knowledge": os.path.join(RAW_DIR, "dataset5_Drug_Knowledge_Base.xlsx"),
}

# Application database (doctor reviews + logged AI predictions). Kept separate from the
# read-only EHR databases so no dataset is ever modified. Created on first use.
APP_DB_PATH = os.environ.get("COGNIOS_APP_DB_PATH", os.path.join(BASE_DIR, "data", "cognios_app.db"))


# DEMO-ONLY LOGIN. Both roles (clinician, manufacturer) share this one fixed password, compared as plain text on the server
# (no hashing) because this project is a local demo. It is never sent to the browser or logged. NOT suitable for any
# public or real deployment. The single place to change it is here (or the COGNIOS_DEMO_PASSWORD environment variable).
DEMO_LOGIN_PASSWORD = os.environ.get("COGNIOS_DEMO_PASSWORD", "Vedant@123")
