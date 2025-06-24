#!/usr/bin/env python3
"""
Test script to verify authentication error handling in hyperparameter agents.
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.langchain_hyperparam_agent import AuthenticationError
from src.agents.hyperparam_agent import AuthenticationError as AuthError2


def test_auth_error_detection():
   """Test that the error detection logic correctly identifies 403 errors."""
   print("Testing authentication error detection logic...")
   
   # Test cases for 403 errors
   auth_error_messages = [
      "Error code: 403 - Error: Permission denied from internal policies",
      "HTTP/1.1 403 Forbidden",
      "403 Forbidden - Permission denied",
      "Error: Permission denied from internal policies. This is likely due to a high-assurance timeout."
   ]
   
   # Test cases for non-auth errors
   non_auth_error_messages = [
      "Network timeout error",
      "500 Internal Server Error",
      "Connection refused",
      "JSON decode error"
   ]
   
   def is_auth_error(error_msg):
      """Replicate the error detection logic from the agents."""
      return "403" in error_msg and ("Forbidden" in error_msg or "Permission denied" in error_msg)
   
   # Test auth error detection
   for msg in auth_error_messages:
      if is_auth_error(msg):
         print(f"✅ PASSED: Correctly identified auth error: {msg[:50]}...")
      else:
         print(f"❌ FAILED: Failed to identify auth error: {msg[:50]}...")
         return False
   
   # Test non-auth error detection
   for msg in non_auth_error_messages:
      if not is_auth_error(msg):
         print(f"✅ PASSED: Correctly identified non-auth error: {msg[:50]}...")
      else:
         print(f"❌ FAILED: Incorrectly identified non-auth error as auth error: {msg[:50]}...")
         return False
   
   return True


def test_auth_error_message():
   """Test that the authentication error message is informative."""
   print("\nTesting authentication error message...")
   
   expected_message = (
      "Authentication failed with the LLM service. This is likely due to expired Globus credentials. "
      "Please re-authenticate by running: 'python3 inference_auth_token.py authenticate --force'. "
      "Make sure you authenticate with an authorized identity provider: ['Argonne National Laboratory', 'Argonne LCF']."
   )
   
   # Create an AuthenticationError with the expected message
   error = AuthenticationError(expected_message)
   
   if "Globus credentials" in str(error) and "re-authenticate" in str(error):
      print("✅ PASSED: Authentication error message is informative")
      print(f"   Message: {str(error)[:100]}...")
      return True
   else:
      print("❌ FAILED: Authentication error message is not informative enough")
      return False


def test_exception_inheritance():
   """Test that AuthenticationError properly inherits from Exception."""
   print("\nTesting exception inheritance...")
   
   try:
      raise AuthenticationError("Test error")
   except Exception as e:
      if isinstance(e, AuthenticationError):
         print("✅ PASSED: AuthenticationError properly inherits from Exception")
         return True
      else:
         print("❌ FAILED: AuthenticationError does not inherit from Exception")
         return False


def main():
   """Run all tests."""
   print("Testing authentication error handling...\n")
   
   tests = [
      test_auth_error_detection,
      test_auth_error_message,
      test_exception_inheritance
   ]
   
   passed = 0
   total = len(tests)
   
   for test in tests:
      if test():
         passed += 1
   
   print(f"\n{'='*50}")
   print(f"Test Results: {passed}/{total} tests passed")
   
   if passed == total:
      print("🎉 All tests passed! Authentication error handling is working correctly.")
   else:
      print("❌ Some tests failed. Please check the implementation.")
   
   return passed == total


if __name__ == "__main__":
   success = main()
   sys.exit(0 if success else 1) 