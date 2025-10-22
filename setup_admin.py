#!/usr/bin/env python3
"""
Setup script to create default admin user for JV Cinelytics
Run this once to create the initial admin account
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'auth'))

from auth_manager import auth_manager

def create_admin_user():
    """Create default admin user"""
    print("Setting up JV Cinelytics Admin User...")
    print("=" * 40)
    
    # Check if admin already exists
    if "admin" in auth_manager.users:
        print("Admin user already exists!")
        return
    
    # Create admin user
    result = auth_manager.register_user(
        username="admin",
        password="admin123",
        email="admin@jvcinelytics.com",
        role="admin"
    )
    
    if result["success"]:
        print("✅ Admin user created successfully!")
        print("Username: admin")
        print("Password: admin123")
        print("\n⚠️  IMPORTANT: Change the admin password after first login!")
    else:
        print(f"❌ Error creating admin user: {result['message']}")

if __name__ == "__main__":
    create_admin_user()
