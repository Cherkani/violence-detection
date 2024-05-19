import hashlib
password="abc"
# Convert password to bytes
password_bytes = password.encode('utf-8')
    
    # Create SHA-256 hash object
hash_object = hashlib.sha256()
    
    # Update hash object with password bytes
hash_object.update(password_bytes)
    
    # Get the hexadecimal representation of the hash
hashed_password = hash_object.hexdigest()
    
print(hashed_password)