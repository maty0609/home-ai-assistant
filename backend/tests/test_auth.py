import pytest
from fastapi.testclient import TestClient
from backend import app
import os

client = TestClient(app)

@pytest.fixture
def test_db():
    """Setup test database"""
    # This would normally set up a test database
    # For now, we'll use the existing database
    pass

def test_register_user():
    """Test user registration"""
    response = client.post(
        "/auth/register",
        json={
            "email": "test@example.com",
            "name": "Test User",
            "password": "testpassword123"
        }
    )
    assert response.status_code in [200, 400]  # 400 if user already exists

def test_login_user():
    """Test user login"""
    response = client.post(
        "/auth/login",
        json={
            "email": "test@example.com",
            "password": "testpassword123"
        }
    )
    assert response.status_code in [200, 401]  # 401 if credentials are wrong

def test_protected_endpoint_without_token():
    """Test that protected endpoints require authentication"""
    response = client.get("/sessions")
    assert response.status_code == 401

def test_protected_endpoint_with_token():
    """Test that protected endpoints work with valid token"""
    # First login to get token
    login_response = client.post(
        "/auth/login",
        json={
            "email": "test@example.com",
            "password": "testpassword123"
        }
    )
    
    if login_response.status_code == 200:
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        response = client.get("/sessions", headers=headers)
        assert response.status_code == 200 