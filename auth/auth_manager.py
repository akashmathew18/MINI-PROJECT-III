import hashlib
import json
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import streamlit as st

class AuthManager:
    def __init__(self, users_file: str = "auth/users.json", sessions_file: str = "auth/sessions.json"):
        self.users_file = users_file
        self.sessions_file = sessions_file
        self.ensure_directories()
        self.load_data()
    
    def ensure_directories(self):
        """Create auth directory if it doesn't exist"""
        os.makedirs("auth", exist_ok=True)
    
    def load_data(self):
        """Load users and sessions from files"""
        # Load users
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        else:
            self.users = {}
            self.save_users()
        
        # Load sessions
        if os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'r') as f:
                self.sessions = json.load(f)
        else:
            self.sessions = {}
            self.save_sessions()
    
    def save_users(self):
        """Save users to file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def save_sessions(self):
        """Save sessions to file"""
        with open(self.sessions_file, 'w') as f:
            json.dump(self.sessions, f, indent=2)
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt"""
        salt = "jv_cinelytics_salt_2024"
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def register_user(self, username: str, password: str, email: str = "", role: str = "user") -> Dict[str, Any]:
        """Register a new user"""
        if username in self.users:
            return {"success": False, "message": "Username already exists"}
        
        if len(password) < 6:
            return {"success": False, "message": "Password must be at least 6 characters"}
        
        hashed_password = self.hash_password(password)
        self.users[username] = {
            "password_hash": hashed_password,
            "email": email,
            "role": role,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "login_count": 0
        }
        
        self.save_users()
        return {"success": True, "message": "User registered successfully"}
    
    def login_user(self, username: str, password: str) -> Dict[str, Any]:
        """Login user and create session"""
        if username not in self.users:
            return {"success": False, "message": "Invalid username or password"}
        
        hashed_password = self.hash_password(password)
        if self.users[username]["password_hash"] != hashed_password:
            return {"success": False, "message": "Invalid username or password"}
        
        # Update user stats
        self.users[username]["last_login"] = datetime.now().isoformat()
        self.users[username]["login_count"] = self.users[username].get("login_count", 0) + 1
        self.save_users()
        
        # Create session
        session_id = hashlib.sha256(f"{username}{datetime.now()}".encode()).hexdigest()
        self.sessions[session_id] = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        self.save_sessions()
        
        return {"success": True, "message": "Login successful", "session_id": session_id}
    
    def logout_user(self, session_id: str) -> bool:
        """Logout user by removing session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.save_sessions()
            return True
        return False
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session and return user info if valid"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        expires_at = datetime.fromisoformat(session["expires_at"])
        
        if datetime.now() > expires_at:
            del self.sessions[session_id]
            self.save_sessions()
            return None
        
        return {
            "username": session["username"],
            "role": self.users[session["username"]]["role"]
        }
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics for admin dashboard"""
        total_users = len(self.users)
        active_sessions = len(self.sessions)
        
        # Count users by role
        role_counts = {}
        for user in self.users.values():
            role = user.get("role", "user")
            role_counts[role] = role_counts.get(role, 0) + 1
        
        # Recent logins (last 7 days)
        recent_logins = 0
        week_ago = datetime.now() - timedelta(days=7)
        for user in self.users.values():
            if user.get("last_login"):
                last_login = datetime.fromisoformat(user["last_login"])
                if last_login > week_ago:
                    recent_logins += 1
        
        return {
            "total_users": total_users,
            "active_sessions": active_sessions,
            "role_counts": role_counts,
            "recent_logins": recent_logins
        }
    
    def get_user_activity(self) -> Dict[str, Any]:
        """Get detailed user activity for analytics"""
        activities = []
        
        for session_id, session in self.sessions.items():
            user = self.users.get(session["username"], {})
            activities.append({
                "username": session["username"],
                "role": user.get("role", "user"),
                "login_time": session["created_at"],
                "login_count": user.get("login_count", 0),
                "last_login": user.get("last_login", "Never")
            })
        
        return {
            "current_sessions": activities,
            "total_sessions_today": len([s for s in self.sessions.values() 
                                       if datetime.fromisoformat(s["created_at"]).date() == datetime.now().date()])
        }
    
    def add_user(self, username: str, password: str, email: str = "", role: str = "user") -> Dict[str, Any]:
        """Add a new user (admin function)"""
        if username in self.users:
            return {"success": False, "message": "Username already exists"}
        
        if len(password) < 6:
            return {"success": False, "message": "Password must be at least 6 characters"}
        
        if role not in ["user", "admin"]:
            return {"success": False, "message": "Invalid role. Must be 'user' or 'admin'"}
        
        hashed_password = self.hash_password(password)
        self.users[username] = {
            "password_hash": hashed_password,
            "email": email,
            "role": role,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "login_count": 0,
            "created_by": "admin"  # Track who created this user
        }
        
        self.save_users()
        return {"success": True, "message": f"User '{username}' created successfully"}
    
    def edit_user(self, username: str, email: str = None, role: str = None) -> Dict[str, Any]:
        """Edit user information (admin function)"""
        if username not in self.users:
            return {"success": False, "message": "User not found"}
        
        if username == "admin":
            return {"success": False, "message": "Cannot edit the main admin account"}
        
        if email is not None:
            self.users[username]["email"] = email
        
        if role is not None:
            if role not in ["user", "admin"]:
                return {"success": False, "message": "Invalid role. Must be 'user' or 'admin'"}
            self.users[username]["role"] = role
        
        self.users[username]["last_modified"] = datetime.now().isoformat()
        self.save_users()
        return {"success": True, "message": f"User '{username}' updated successfully"}
    
    def change_user_password(self, username: str, new_password: str) -> Dict[str, Any]:
        """Change user password (admin function)"""
        if username not in self.users:
            return {"success": False, "message": "User not found"}
        
        if len(new_password) < 6:
            return {"success": False, "message": "Password must be at least 6 characters"}
        
        hashed_password = self.hash_password(new_password)
        self.users[username]["password_hash"] = hashed_password
        self.users[username]["password_changed"] = datetime.now().isoformat()
        self.save_users()
        return {"success": True, "message": f"Password for '{username}' changed successfully"}
    
    def update_own_profile(self, username: str, email: str = None) -> Dict[str, Any]:
        """Allow users to update their own profile information"""
        if username not in self.users:
            return {"success": False, "message": "User not found"}
        
        if email is not None:
            self.users[username]["email"] = email
        
        self.users[username]["last_modified"] = datetime.now().isoformat()
        self.save_users()
        return {"success": True, "message": "Profile updated successfully"}
    
    def change_own_password(self, username: str, current_password: str, new_password: str) -> Dict[str, Any]:
        """Allow users to change their own password"""
        if username not in self.users:
            return {"success": False, "message": "User not found"}
        
        # Verify current password
        hashed_current = self.hash_password(current_password)
        if self.users[username]["password_hash"] != hashed_current:
            return {"success": False, "message": "Current password is incorrect"}
        
        if len(new_password) < 6:
            return {"success": False, "message": "New password must be at least 6 characters"}
        
        hashed_password = self.hash_password(new_password)
        self.users[username]["password_hash"] = hashed_password
        self.users[username]["password_changed"] = datetime.now().isoformat()
        self.save_users()
        return {"success": True, "message": "Password changed successfully"}
    
    def delete_user(self, username: str) -> Dict[str, Any]:
        """Delete a user (admin function)"""
        if username not in self.users:
            return {"success": False, "message": "User not found"}
        
        if username == "admin":
            return {"success": False, "message": "Cannot delete the main admin account"}
        
        # Remove user from users
        del self.users[username]
        
        # Remove any active sessions for this user
        sessions_to_remove = []
        for session_id, session in self.sessions.items():
            if session["username"] == username:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        
        self.save_users()
        self.save_sessions()
        return {"success": True, "message": f"User '{username}' deleted successfully"}
    
    def get_user_details(self, username: str) -> Optional[Dict[str, Any]]:
        """Get detailed user information"""
        if username not in self.users:
            return None
        
        user_data = self.users[username].copy()
        # Don't return the password hash
        user_data.pop("password_hash", None)
        user_data["username"] = username
        return user_data
    
    def search_users(self, query: str = "", role_filter: str = "all") -> List[Dict[str, Any]]:
        """Search and filter users"""
        results = []
        
        for username, user_data in self.users.items():
            # Apply role filter
            if role_filter != "all" and user_data.get("role") != role_filter:
                continue
            
            # Apply text search
            if query:
                query_lower = query.lower()
                if (query_lower not in username.lower() and 
                    query_lower not in user_data.get("email", "").lower()):
                    continue
            
            user_info = {
                "username": username,
                "email": user_data.get("email", ""),
                "role": user_data.get("role", "user"),
                "created_at": user_data.get("created_at", ""),
                "last_login": user_data.get("last_login", "Never"),
                "login_count": user_data.get("login_count", 0)
            }
            results.append(user_info)
        
        # Sort by username
        results.sort(key=lambda x: x["username"])
        return results
    
    def bulk_update_users(self, usernames: List[str], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Bulk update multiple users"""
        if not usernames:
            return {"success": False, "message": "No users specified"}
        
        updated_count = 0
        errors = []
        
        for username in usernames:
            if username == "admin":
                errors.append(f"Cannot modify admin account")
                continue
            
            if username not in self.users:
                errors.append(f"User '{username}' not found")
                continue
            
            try:
                if "role" in updates:
                    if updates["role"] not in ["user", "admin"]:
                        errors.append(f"Invalid role for '{username}'")
                        continue
                    self.users[username]["role"] = updates["role"]
                
                if "email" in updates:
                    self.users[username]["email"] = updates["email"]
                
                self.users[username]["last_modified"] = datetime.now().isoformat()
                updated_count += 1
            except Exception as e:
                errors.append(f"Error updating '{username}': {str(e)}")
        
        self.save_users()
        
        if errors:
            return {
                "success": updated_count > 0,
                "message": f"Updated {updated_count} users. Errors: {'; '.join(errors)}"
            }
        else:
            return {
                "success": True,
                "message": f"Successfully updated {updated_count} users"
            }
    
    def bulk_delete_users(self, usernames: List[str]) -> Dict[str, Any]:
        """Bulk delete multiple users"""
        if not usernames:
            return {"success": False, "message": "No users specified"}
        
        deleted_count = 0
        errors = []
        
        for username in usernames:
            if username == "admin":
                errors.append("Cannot delete admin account")
                continue
            
            if username not in self.users:
                errors.append(f"User '{username}' not found")
                continue
            
            try:
                del self.users[username]
                
                # Remove sessions
                sessions_to_remove = []
                for session_id, session in self.sessions.items():
                    if session["username"] == username:
                        sessions_to_remove.append(session_id)
                
                for session_id in sessions_to_remove:
                    del self.sessions[session_id]
                
                deleted_count += 1
            except Exception as e:
                errors.append(f"Error deleting '{username}': {str(e)}")
        
        self.save_users()
        self.save_sessions()
        
        if errors:
            return {
                "success": deleted_count > 0,
                "message": f"Deleted {deleted_count} users. Errors: {'; '.join(errors)}"
            }
        else:
            return {
                "success": True,
                "message": f"Successfully deleted {deleted_count} users"
            }

# Initialize global auth manager
auth_manager = AuthManager()

def get_current_user():
    """Get current user from session state"""
    if "session_id" in st.session_state:
        return auth_manager.validate_session(st.session_state.session_id)
    return None

def is_authenticated():
    """Check if user is authenticated"""
    return get_current_user() is not None

def require_auth():
    """Decorator to require authentication for pages"""
    if not is_authenticated():
        st.error("Please log in to access this page.")
        st.stop()
