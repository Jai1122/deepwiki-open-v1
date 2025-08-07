#!/usr/bin/env python3
"""
Debug script to test Bitbucket HTTP access token authentication
"""
import sys
import os
import subprocess
import requests
from urllib.parse import urlparse, quote

def test_bitbucket_api(repo_url, access_token):
    """Test Bitbucket API access with HTTP access token"""
    print(f"🔍 Testing Bitbucket API access for: {repo_url}")
    
    try:
        parsed_url = urlparse(repo_url)
        path_parts = parsed_url.path.strip('/').split('/')
        workspace, repo_slug = path_parts[0], path_parts[1]
        
        # Test API access
        api_url = f"https://api.bitbucket.org/2.0/repositories/{workspace}/{repo_slug}"
        headers = {'Authorization': f'Bearer {access_token}'}
        
        print(f"📡 Making API request to: {api_url}")
        print(f"🔐 Using Bearer token (length: {len(access_token)})")
        
        response = requests.get(api_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            repo_info = response.json()
            print(f"✅ API access successful!")
            print(f"   Repository: {repo_info.get('name', 'Unknown')}")
            print(f"   Language: {repo_info.get('language', 'Unknown')}")
            print(f"   Private: {repo_info.get('is_private', 'Unknown')}")
            return True
        else:
            print(f"❌ API access failed!")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed with exception: {e}")
        return False

def test_git_clone(repo_url, access_token, test_dir="/tmp/bitbucket_test"):
    """Test git clone with HTTP access token"""
    print(f"\n🔍 Testing git clone for: {repo_url}")
    
    try:
        # Clean up previous test
        if os.path.exists(test_dir):
            subprocess.run(['rm', '-rf', test_dir], check=True)
        
        # Prepare clone URL with token
        parsed_url = urlparse(repo_url)
        encoded_token = quote(access_token, safe='')
        clone_url = f"https://x-token-auth:{encoded_token}@{parsed_url.netloc}{parsed_url.path}"
        
        print(f"🔐 Using clone URL format: https://x-token-auth:[TOKEN]@{parsed_url.netloc}{parsed_url.path}")
        print(f"📁 Cloning to: {test_dir}")
        
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', clone_url, test_dir],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✅ Git clone successful!")
            # Count files
            if os.path.exists(test_dir):
                file_count = sum(len(files) for _, _, files in os.walk(test_dir))
                print(f"   Files cloned: {file_count}")
            return True
        else:
            print(f"❌ Git clone failed!")
            print(f"   Return code: {result.returncode}")
            print(f"   stderr: {result.stderr}")
            print(f"   stdout: {result.stdout}")
            
            # Analyze common errors
            error_output = (result.stderr + result.stdout).lower()
            if "authentication failed" in error_output:
                print(f"🔍 Authentication issue detected. Check your HTTP access token.")
            elif "unable to update url base from redirection" in error_output:
                print(f"🔍 Redirection error detected. This might be related to token encoding.")
            elif "repository not found" in error_output:
                print(f"🔍 Repository not found. Check the URL and token permissions.")
                
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Git clone timed out (60s)")
        return False
    except Exception as e:
        print(f"❌ Git clone failed with exception: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            subprocess.run(['rm', '-rf', test_dir], check=False)

def main():
    if len(sys.argv) != 3:
        print("Usage: python debug_bitbucket.py <repo_url> <access_token>")
        print("Example: python debug_bitbucket.py https://bitbucket.org/username/repo your_http_token")
        sys.exit(1)
    
    repo_url = sys.argv[1]
    access_token = sys.argv[2]
    
    print("🚀 Bitbucket HTTP Access Token Debug Tool")
    print("=" * 50)
    
    # Validate inputs
    if not repo_url.startswith('https://bitbucket.org/'):
        print(f"❌ Invalid repo URL. Must start with https://bitbucket.org/")
        sys.exit(1)
    
    if len(access_token) < 10:
        print(f"❌ Access token seems too short. HTTP access tokens are typically longer.")
        sys.exit(1)
    
    print(f"Repository: {repo_url}")
    print(f"Token length: {len(access_token)} characters")
    print(f"Token preview: {access_token[:8]}...{access_token[-4:]}")
    
    # Test API access
    api_success = test_bitbucket_api(repo_url, access_token)
    
    # Test git clone
    clone_success = test_git_clone(repo_url, access_token)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"   API Access: {'✅ SUCCESS' if api_success else '❌ FAILED'}")
    print(f"   Git Clone: {'✅ SUCCESS' if clone_success else '❌ FAILED'}")
    
    if api_success and clone_success:
        print("🎉 All tests passed! Your HTTP access token should work with DeepWiki.")
    else:
        print("⚠️  Some tests failed. Check your token permissions and repository access.")
        print("\nTroubleshooting tips:")
        print("1. Ensure your HTTP access token has 'Repository read' permissions")
        print("2. Verify the repository URL is correct")
        print("3. Check if the repository is private and you have access")
        print("4. Try creating a new HTTP access token")

if __name__ == "__main__":
    main()