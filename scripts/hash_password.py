"""Generate a CHAINLIT_USERS record for a new user, with a freshly generated password.

Usage (from repo root):
    python scripts/hash_password.py

Prompts for a username, generates a random password, and prints both the
password (to hand to the user, once) and a "username:salt_hex:hash_hex"
fragment to append (joined with "|") to CHAINLIT_USERS in .env.
"""

import hashlib
import secrets

PBKDF2_ITERATIONS = 600_000  # current OWASP-recommended minimum for PBKDF2-SHA256
PASSWORD_BYTES = 12  # ~16 base64url chars of entropy


def hash_password(password: str, salt: bytes) -> bytes:
    """Derive a PBKDF2-SHA256 hash for `password` using `salt`. Shared by app.py's login check."""
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)


def main() -> None:
    """Prompt for a username, generate a random password, print both the password and the record."""
    username = input("Username: ").strip()
    password = secrets.token_urlsafe(PASSWORD_BYTES)

    salt = secrets.token_bytes(16)
    digest = hash_password(password, salt)

    print(f"\nPassword (share once, not stored anywhere else): {password}")
    print(f"CHAINLIT_USERS record: {username}:{salt.hex()}:{digest.hex()}")


if __name__ == "__main__":
    main()
