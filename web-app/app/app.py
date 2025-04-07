# Connect to MongoDB
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)  # Adjust the connection string if necessary
db = client["sentiment"]  # Database name
collection = db["texts"]  # Collection name