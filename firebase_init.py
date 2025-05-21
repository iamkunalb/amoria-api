import firebase_admin
from firebase_admin import credentials, firestore

# ✅ Use Cloud Run's built-in identity
cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred)

db = firestore.client()
