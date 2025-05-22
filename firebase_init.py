import firebase_admin
from firebase_admin import credentials, firestore

# cred = credentials.Certificate("/secrets/firebase.json")
cred = credentials.Certificate("/secrets/firebase-service-key")
firebase_admin.initialize_app(cred)

db = firestore.client()
